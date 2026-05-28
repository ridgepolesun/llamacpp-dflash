// DFlash speculative decoding demo
// ===================================
// Implements speculative decoding using the DFlash draft model (Qwen3DFLASH).
//
// Usage:
//   llama-dflash -m target.gguf --model-draft draft.gguf -p "prompt" -n 128

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ── helpers ──────────────────────────────────────────────────────────────────

// Add a float embedding row to a batch (batch must have embd allocated).
static void batch_embd_add(llama_batch & batch,
                            const float * embd, int32_t n_embd,
                            llama_pos pos, llama_seq_id seq, bool want_logits) {
    memcpy(batch.embd + (size_t)batch.n_tokens * n_embd, embd, n_embd * sizeof(float));
    batch.pos     [batch.n_tokens]    = pos;
    batch.n_seq_id[batch.n_tokens]    = 1;
    batch.seq_id  [batch.n_tokens][0] = seq;
    batch.logits  [batch.n_tokens]    = want_logits ? 1 : 0;
    batch.n_tokens++;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    // ── Extract DFlash-specific args before common_params_parse ──────────────
    // --draft-ctx-max N : cap the draft model's context window to N tokens (0 = unlimited)
    int32_t arg_draft_ctx_max = 0;
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--draft-ctx-max") == 0) {
            arg_draft_ctx_max = std::max(0, atoi(argv[i + 1]));
            // Remove both argv[i] and argv[i+1] so common_params_parse doesn't see them
            for (int j = i; j < argc - 2; j++) argv[j] = argv[j + 2];
            argc -= 2;
            break;
        }
    }

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    const bool no_draft = params.speculative.mparams_dft.path.empty();

    llama_backend_init();
    llama_numa_init(params.numa);

    // ── load target model ───────────────────────────────────────────────────
    auto tgt_init    = common_init_from_params(params);
    llama_model   * model_tgt = tgt_init->model();
    llama_context * ctx_tgt   = tgt_init->context();
    if (!model_tgt || !ctx_tgt) {
        fprintf(stderr, "%s: failed to load target model\n", __func__);
        return 1;
    }

    // ── load draft model ────────────────────────────────────────────────────
    std::unique_ptr<common_init_result> dft_init;
    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;
    if (!no_draft) {
        common_params params_dft = params;
        params_dft.model.path = params.speculative.mparams_dft.path;
        // Draft model uses bidirectional (non-causal) attention: the entire batch
        // [ctx_tokens + noise_tokens] must fit in a single micro-batch.
        // Set n_batch = n_ubatch = n_ctx so n_dft never exceeds n_ubatch.
        params_dft.n_ctx      = params.n_ctx;
        params_dft.n_batch    = params.n_ctx;
        params_dft.n_ubatch   = params.n_ctx;
        dft_init  = common_init_from_params(params_dft);
        model_dft = dft_init->model();
        ctx_dft   = dft_init->context();
        if (!model_dft || !ctx_dft) {
            fprintf(stderr, "%s: failed to load draft model\n", __func__);
            return 1;
        }
    }

    // ── DFlash parameters ────────────────────────────────────────────────────
    int32_t n_noise    = 1;
    int32_t n_tgt_lyrs = 0;
    std::vector<int32_t> tgt_layer_ids;
    if (!no_draft) {
        n_noise    = (int32_t)llama_model_dflash_block_size(model_dft);
        n_tgt_lyrs = (int32_t)llama_model_dflash_num_target_layers(model_dft);
        if (n_noise <= 0 || n_tgt_lyrs <= 0) {
            fprintf(stderr, "%s: draft model is not a DFlash model "
                    "(block_size=%d num_target_layers=%d)\n",
                    __func__, n_noise, n_tgt_lyrs);
            return 1;
        }
        tgt_layer_ids.resize(n_tgt_lyrs);
        llama_model_dflash_target_layer_ids(model_dft, tgt_layer_ids.data(), n_tgt_lyrs);
        fprintf(stderr, "%s: DFlash  n_noise=%d  n_target_layers=%d\n",
                __func__, n_noise, n_tgt_lyrs);
        fprintf(stderr, "%s: target_layer_ids: [", __func__);
        for (int i = 0; i < n_tgt_lyrs; i++)
            fprintf(stderr, "%d%s", tgt_layer_ids[i], i+1<n_tgt_lyrs?", ":"");
        fprintf(stderr, "]\n");
    }

    const int32_t n_embd    = llama_model_n_embd(model_tgt);
    const int32_t n_vocab   = (int32_t)llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
    const int32_t n_predict = params.n_predict < 0 ? 64 : params.n_predict;
    const int32_t n_cands   = n_noise - 1;  // candidates per block
    fprintf(stderr, "%s: mode=%s  n_embd=%d  n_vocab=%d  n_predict=%d\n",
            __func__, no_draft ? "AR-baseline" : "DFlash", n_embd, n_vocab, n_predict);

    // ── Mask token embedding ──────────────────────────────────────────────────
    std::vector<float> mask_embd(n_embd, 0.0f);
    if (!no_draft) {
        const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);
        const llama_token mask_token_id = llama_vocab_mask(vocab_dft);
        if (mask_token_id < 0) {
            fprintf(stderr, "%s: draft model has no mask token\n", __func__);
            return 1;
        }
        fprintf(stderr, "%s: mask_token_id=%d\n", __func__, mask_token_id);
        if (!llama_model_get_token_embd(model_tgt, mask_token_id, mask_embd.data(), n_embd)) {
            fprintf(stderr, "%s: failed to get mask token embedding from target\n", __func__);
            return 1;
        }
    }

    // ── GPU lm_head ───────────────────────────────────────────────────────────
    llama_lm_head_gpu * lmh_gpu = nullptr;
    if (!no_draft) {
        // apply_output_norm=false: draft embeddings already have output_norm applied
        lmh_gpu = llama_lm_head_gpu_create(model_tgt, n_cands, false);
        if (!lmh_gpu) {
            fprintf(stderr, "%s: GPU lm_head unavailable, using CPU fallback\n", __func__);
        }
    }

    // ── tokenize ──────────────────────────────────────────────────────────────
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    std::vector<llama_token> tokens = common_tokenize(ctx_tgt, params.prompt, true, true);
    if (tokens.empty()) {
        fprintf(stderr, "%s: empty token list\n", __func__);
        return 1;
    }
    fprintf(stderr, "%s: %d prompt tokens\n", __func__, (int)tokens.size());

    // ── sampling ──────────────────────────────────────────────────────────────
    auto smpl_tgt = common_sampler_init(model_tgt, params.sampling);

    // ── prefill target model ─────────────────────────────────────────────────
    const int32_t n_prompt = (int32_t)tokens.size();
    std::vector<float> ctx_proj;       // [n_ctx * n_embd]
    std::vector<float> ctx_hidden_tmp;

    if (!no_draft) {
        llama_set_hidden_capture_layers(ctx_tgt, tgt_layer_ids.data(), n_tgt_lyrs);
    }
    {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (int32_t i = 0; i < n_prompt; i++) {
            common_batch_add(batch, tokens[i], (llama_pos)i, {0},
                             i == n_prompt - 1);
        }
        if (llama_decode(ctx_tgt, batch) != 0) {
            fprintf(stderr, "%s: target prefill failed\n", __func__);
            llama_batch_free(batch);
            return 1;
        }
        llama_batch_free(batch);
    }
    if (!no_draft) {
        ctx_hidden_tmp.resize((size_t)n_prompt * n_tgt_lyrs * n_embd);
        for (int32_t li = 0; li < n_tgt_lyrs; li++) {
            const float * h = llama_get_layer_hidden(ctx_tgt, tgt_layer_ids[li]);
            if (!h) {
                fprintf(stderr, "%s: hidden layer %d unavailable after prefill\n",
                        __func__, tgt_layer_ids[li]);
                return 1;
            }
            for (int32_t t = 0; t < n_prompt; t++) {
                memcpy(ctx_hidden_tmp.data() + (size_t)(t * n_tgt_lyrs + li) * n_embd,
                       h + (size_t)t * n_embd, n_embd * sizeof(float));
            }
        }
        ctx_proj.resize((size_t)n_prompt * n_embd);
        if (!llama_model_dflash_project(model_dft, ctx_hidden_tmp.data(), ctx_proj.data(), n_prompt)) {
            fprintf(stderr, "%s: dflash projection failed during prefill\n", __func__);
            return 1;
        }
        llama_set_hidden_capture_layers(ctx_tgt, nullptr, 0);

        // Debug: verify hidden states and ctx_proj are non-zero after prefill
        if (getenv("DFLASH_DEBUG_HIDDEN")) {
            // Check raw hidden states (last layer, last token)
            double hidden_sum = 0.0, hidden_sq = 0.0;
            int hidden_nonzero = 0;
            const int dbg_tok = n_prompt - 1;  // last prompt token
            for (int li = 0; li < n_tgt_lyrs; li++) {
                for (int d = 0; d < n_embd; d++) {
                    float v = ctx_hidden_tmp[(size_t)(dbg_tok * n_tgt_lyrs + li) * n_embd + d];
                    hidden_sum += v;
                    hidden_sq  += (double)v * v;
                    if (v != 0.0f) hidden_nonzero++;
                }
            }
            fprintf(stderr, "%s: [debug] hidden_concat last_tok=%d: sum=%.4f rms=%.6f nonzero=%d/%d\n",
                    __func__, dbg_tok, hidden_sum,
                    sqrtf((float)(hidden_sq / (n_tgt_lyrs * n_embd))),
                    hidden_nonzero, n_tgt_lyrs * n_embd);
            // Check ctx_proj (last prompt token, first 8 values)
            double proj_sum = 0.0;
            int proj_nonzero = 0;
            int proj_nan = 0;
            for (int d = 0; d < n_embd; d++) {
                float v = ctx_proj[(size_t)dbg_tok * n_embd + d];
                if (std::isnan(v)) proj_nan++;
                else if (v != 0.0f) { proj_sum += v; proj_nonzero++; }
            }
            fprintf(stderr, "%s: [debug] ctx_proj last_tok=%d: sum=%.4f nonzero=%d/%d nan=%d\n",
                    __func__, dbg_tok, proj_sum, proj_nonzero, n_embd, proj_nan);
            // Sample a few raw values from first layer, last token
            fprintf(stderr, "%s: [debug] hidden layer=%d tok=%d first8:",
                    __func__, tgt_layer_ids[0], dbg_tok);
            for (int d = 0; d < 8 && d < n_embd; d++) {
                float v = ctx_hidden_tmp[(size_t)(dbg_tok * n_tgt_lyrs + 0) * n_embd + d];
                fprintf(stderr, " %.4f", v);
            }
            fprintf(stderr, "\n");
            fprintf(stderr, "%s: [debug] ctx_proj tok=%d first8:", __func__, dbg_tok);
            for (int d = 0; d < 8 && d < n_embd; d++) {
                float v = ctx_proj[(size_t)dbg_tok * n_embd + d];
                fprintf(stderr, " %.4f", v);
            }
            fprintf(stderr, "\n");
        }
    }

    // ── GPU dflash proj (initialized after n_prompt is known) ─────────────────
    llama_dflash_proj_gpu * proj_gpu = nullptr;
    if (!no_draft) {
        const int32_t proj_batch = std::max(n_prompt, n_noise);
        // Share lm_head's backend to avoid creating a second CUDA context (Jetson OOM)
        ggml_backend_t shared_be = llama_lm_head_gpu_get_backend(lmh_gpu);
        proj_gpu = llama_dflash_proj_gpu_create(model_dft, proj_batch, shared_be);
        if (!proj_gpu) {
            fprintf(stderr, "%s: GPU dflash proj unavailable, using CPU fallback\n", __func__);
        }
    }

    // ── Hybrid model snapshot/replay support ────────────────────────────────
    // Hybrid models (e.g. Qwen3.5 with Gated Delta Net + SWA) use
    // llama_memory_hybrid_iswa which cannot handle partial seq_rm for recurrent layers.
    // Fix: save recurrent-only snapshot (PARTIAL_ONLY) before each verification batch.
    // On rejection, restore recurrent state + trim attention KV + replay accepted prefix.
    //
    // Detection: after prefill, compare PARTIAL_ONLY size vs full size.
    //   - hybrid: PARTIAL_ONLY skips attention KV → sz_recr << sz_full (fixed small size)
    //   - pure-attention: PARTIAL_ONLY == full state → sz_recr == sz_full (large, grows)
    // Only save snapshots for hybrid models; pure-attention models never need them.
    bool use_recr_snap = false;
    llama_gpu_snapshot * gpu_snap = nullptr;
    size_t recr_snap_size = 0;
    std::vector<uint8_t> recr_snap;
    size_t recr_snap_written = 0;
    llama_token tok_vb = LLAMA_TOKEN_NULL;
    if (!no_draft) {
        const size_t sz_full = llama_state_seq_get_size(ctx_tgt, 0);
        const size_t sz_recr = llama_state_seq_get_size_ext(ctx_tgt, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        use_recr_snap = (sz_full > 0 && sz_recr < sz_full);
        if (use_recr_snap) {
            gpu_snap = llama_gpu_snapshot_create(ctx_tgt);
            if (gpu_snap) {
                fprintf(stderr, "%s: hybrid model — GPU snapshot enabled\n", __func__);
            } else {
                recr_snap_size = sz_recr;
                recr_snap.resize(sz_recr);
                fprintf(stderr, "%s: hybrid model — CPU snapshot fallback (recr=%zu full=%zu)\n",
                        __func__, sz_recr, sz_full);
            }
        }
    }

    // ── Draft ctx sliding window ──────────────────────────────────────────────
    // ctx_proj grows by (1+n_accepted) each iteration → draft batch grows → O(n²) cost.
    // Cap ctx_proj at max_ctx_window tokens: slide the window when exceeded.
    // Set via --draft-ctx-max N (or DFLASH_MAX_CTX env var as fallback). 0 = unlimited.
    int32_t max_ctx_window = arg_draft_ctx_max;
    if (max_ctx_window == 0) {
        const char * e = getenv("DFLASH_MAX_CTX");
        if (e) max_ctx_window = std::max(0, atoi(e));
    }
    if (max_ctx_window > 0) {
        fprintf(stderr, "%s: draft ctx window capped at %d tokens\n", __func__, max_ctx_window);
    }

    // Pre-allocate scratch buffers to avoid hot-path heap allocations.
    // Max per-iter hidden tmp: n_noise tokens (full vb) * n_tgt_lyrs * n_embd.
    std::vector<float> hidden_tmp_scratch(no_draft ? 0 : (size_t)n_noise * n_tgt_lyrs * n_embd);
    // Max per-iter projection output: n_noise tokens * n_embd.
    std::vector<float> proj_scratch(no_draft ? 0 : (size_t)n_noise * n_embd);

    // Sample first token from prefill logits
    llama_token tok = common_sampler_sample(smpl_tgt, ctx_tgt, -1);
    common_sampler_accept(smpl_tgt, tok, true);
    printf("%s", common_token_to_piece(ctx_tgt, tok).c_str());
    fflush(stdout);

    // n_past = tokens already in KV (prompt tokens). tok is NOT yet decoded.
    int32_t n_past   = n_prompt;
    int32_t n_ctx    = no_draft ? 0 : n_prompt;
    int32_t n_gen    = 1;
    int32_t n_accept = 0;

    const int64_t t_start = ggml_time_us();

    // ── AR baseline mode ──────────────────────────────────────────────────────
    if (no_draft) {
        while (n_gen < n_predict) {
            llama_batch b = llama_batch_init(1, 0, 1);
            common_batch_add(b, tok, (llama_pos)n_past, {0}, true);
            if (llama_decode(ctx_tgt, b) != 0) {
                llama_batch_free(b);
                fprintf(stderr, "\n%s: decode failed\n", __func__);
                break;
            }
            llama_batch_free(b);
            n_past++;
            tok = common_sampler_sample(smpl_tgt, ctx_tgt, -1);
            common_sampler_accept(smpl_tgt, tok, true);
            printf("%s", common_token_to_piece(ctx_tgt, tok).c_str());
            fflush(stdout);
            n_gen++;
            if (llama_vocab_is_eog(vocab, tok)) break;
        }
        printf("\n");
        const double t_sec = (ggml_time_us() - t_start) * 1e-6;
        fprintf(stderr, "\n%s: AR baseline  generated=%d  %.2f tok/s\n",
                __func__, n_gen, n_gen / t_sec);
        common_sampler_free(smpl_tgt);
        llama_backend_free();
        return 0;
    }

    // ── DFlash decode loop ────────────────────────────────────────────────────
    //
    // State at loop entry:
    //   - tok: next token to decode (NOT yet in KV cache)
    //   - n_past: # tokens in KV (prompt + previously accepted/replayed)
    //   - n_ctx: total accumulated ctx tokens (= n_past); GROWS across iterations
    //   - ctx_proj: projected hidden states for ALL tokens [0..n_ctx-1]
    //
    // ctx_proj ACCUMULATES across iterations, matching Python use_cache=True:
    //   the draft model attends to all past context via full [ctx|noise] batch.
    //   ctx positions are always [0..n_ctx-1]; noise positions are [n_ctx..n_ctx+n_noise-1].
    //
    // Each iteration:
    //   1. Build noise_embd = [embed(tok), mask, ..., mask]
    //   2. Run draft: full [ctx | noise] batch, bidirectional attention
    //   3. Get candidates from draft output (last n_cands noise positions)
    //   4. Decode vb = [tok @ n_past | c0..c_{n_cands-1}] with hidden capture ON
    //   5. Accept/reject: logit[ni] predicts candidates[ni]
    //   6. On rejection at ni:
    //      - Extract hidden of [tok | c0..c_{ni-1}] → project → APPEND to ctx_proj
    //      - correction = argmax(logit[ni]); crop KV; n_past += 1+ni; tok = correction
    //   7. On full acceptance:
    //      - Extract hidden of [tok | c0..c_{n_cands-1}] → project → APPEND to ctx_proj
    //      - bonus = argmax(logit[n_cands]); n_past += 1+n_cands; tok = bonus

    std::vector<float> tok_embd_buf(n_embd);
    std::vector<float> noise_embd((size_t)n_noise * n_embd, 0.0f);
    std::vector<float> draft_embd((size_t)n_cands * n_embd);
    std::vector<float> draft_logits((size_t)n_cands * n_vocab);
    std::vector<llama_token> candidates(n_cands);

    // === Per-phase timing accumulators =========================================
    int64_t t_draft_us    = 0;  // draft model forward (llama_decode)
    int64_t t_lmhead_us   = 0;  // target lm_head applied to draft embeddings
    int64_t t_verify_us   = 0;  // target model verification forward
    int64_t t_project_us  = 0;  // hidden-state projection (fc + norm)
    int64_t n_iters       = 0;  // loop iterations

    while (n_gen < n_predict) {
        bool ok = true;
        n_iters++;

        // === Generate draft candidates =========================================
        // Build noise_embd: [embed(tok), mask, ..., mask]
        if (!llama_model_get_token_embd(model_tgt, tok, tok_embd_buf.data(), n_embd)) {
            fprintf(stderr, "\n%s: failed to get tok embedding\n", __func__);
            break;
        }
        memcpy(noise_embd.data(), tok_embd_buf.data(), n_embd * sizeof(float));
        for (int32_t ni = 1; ni < n_noise; ni++) {
            memcpy(noise_embd.data() + (size_t)ni * n_embd, mask_embd.data(), n_embd * sizeof(float));
        }

        // Draft batch: [ctx_0..ctx_{n_ctx-1} | noise_0..noise_{n_noise-1}]
        // ctx positions: absolute [0..n_ctx-1] (accumulated from all past iterations)
        // noise positions: absolute [n_ctx..n_ctx+n_noise-1] (= [n_past..n_past+n_noise-1])
        // Python: position_ids[:, 0 : start + block_size] with full ctx each call
        const int32_t n_dft = n_ctx + n_noise;

        {
            llama_batch dft_b = llama_batch_init(n_dft, n_embd, 1);
            dft_b.n_tokens = 0;
            // Context tokens at absolute positions [0..n_ctx-1]
            for (int32_t t = 0; t < n_ctx; t++) {
                batch_embd_add(dft_b, ctx_proj.data() + (size_t)t * n_embd, n_embd,
                               (llama_pos)t, 0, true);
            }
            // Noise tokens at absolute positions [n_ctx..n_ctx+n_noise-1]
            for (int32_t ni = 0; ni < n_noise; ni++) {
                batch_embd_add(dft_b, noise_embd.data() + (size_t)ni * n_embd, n_embd,
                               (llama_pos)(n_ctx + ni), 0, true);
            }

            llama_set_embeddings(ctx_dft, true);
            // Clear draft KV state: the graph uses full recomputation each call
            // (bidirectional attention, no KV cache), so positions always restart from 0.
            llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, -1, -1);
            const int64_t t0_draft = ggml_time_us();
            if (llama_decode(ctx_dft, dft_b) != 0) {
                fprintf(stderr, "\n%s: draft decode failed\n", __func__);
                llama_batch_free(dft_b);
                ok = false;
            }
            t_draft_us += ggml_time_us() - t0_draft;
            llama_batch_free(dft_b);
        }
        if (!ok) break;

        // Collect draft output embeddings: last n_cands noise outputs
        // (skip ctx tokens and noise[0], take noise[1..n_cands])
        for (int32_t ni = 0; ni < n_cands && ok; ni++) {
            const float * e = llama_get_embeddings_ith(ctx_dft, n_ctx + 1 + ni);
            if (!e) {
                fprintf(stderr, "\n%s: draft embd[%d] unavailable\n", __func__, ni);
                ok = false;
            } else {
                memcpy(draft_embd.data() + (size_t)ni * n_embd, e, n_embd * sizeof(float));
            }
        }
        llama_set_embeddings(ctx_dft, false);
        if (!ok) break;

        // Apply target lm_head to draft embeddings → greedy argmax
        {
            bool lmh_ok;
            const bool need_logits = (n_gen == 1 && getenv("DFLASH_DUMP_DRAFT_LOGITS"));
            const int64_t t0_lmh = ggml_time_us();
            if (lmh_gpu && !need_logits) {
                lmh_ok = llama_lm_head_gpu_apply_argmax(lmh_gpu, draft_embd.data(), candidates.data(), n_cands);
            } else if (lmh_gpu) {
                lmh_ok = llama_lm_head_gpu_apply(lmh_gpu, draft_embd.data(), draft_logits.data(), n_cands);
                if (lmh_ok) {
                    for (int32_t ni = 0; ni < n_cands; ni++) {
                        const float * lg = draft_logits.data() + (size_t)ni * n_vocab;
                        candidates[ni] = (llama_token)(std::max_element(lg, lg + n_vocab) - lg);
                    }
                }
            } else {
                lmh_ok = llama_model_apply_lm_head(model_tgt, draft_embd.data(), draft_logits.data(), n_cands, false);
                if (lmh_ok) {
                    for (int32_t ni = 0; ni < n_cands; ni++) {
                        const float * lg = draft_logits.data() + (size_t)ni * n_vocab;
                        candidates[ni] = (llama_token)(std::max_element(lg, lg + n_vocab) - lg);
                    }
                }
            }
            t_lmhead_us += ggml_time_us() - t0_lmh;
            if (!lmh_ok) {
                fprintf(stderr, "\n%s: lm_head failed\n", __func__);
                break;
            }
        }

        // Dump first-iteration draft logits for debugging (if requested)
        if (n_gen == 1) {
            const char * dump_path = getenv("DFLASH_DUMP_DRAFT_LOGITS");
            if (dump_path) {
                FILE * fp = fopen(dump_path, "w");
                if (fp) {
                    fprintf(fp, "{\n  \"tok\": %d,\n  \"cands\": [", (int)tok);
                    for (int32_t ni = 0; ni < n_cands; ni++)
                        fprintf(fp, "%d%s", (int)candidates[ni], ni+1<n_cands?",":"");
                    fprintf(fp, "],\n  \"cand_logits_top100\": [\n");
                    for (int32_t ni = 0; ni < n_cands; ni++) {
                        // Collect top-100
                        std::vector<std::pair<float,int>> scored(n_vocab);
                        const float * lg = draft_logits.data() + (size_t)ni * n_vocab;
                        for (int v = 0; v < n_vocab; v++) scored[v] = {lg[v], v};
                        std::partial_sort(scored.begin(), scored.begin()+100, scored.end(),
                            [](const auto&a,const auto&b){return a.first>b.first;});
                        fprintf(fp, "    {\"pos\":%d,\"top1\":%d,\"top1_logit\":%.6f,\"top100\":[",
                                ni, (int)candidates[ni], scored[0].first);
                        for (int k = 0; k < 100; k++)
                            fprintf(fp, "[%d,%.6f]%s", scored[k].second, scored[k].first, k<99?",":"");
                        fprintf(fp, "]}%s\n", ni+1<n_cands?",":"");
                    }
                    fprintf(fp, "  ]\n}\n");
                    fclose(fp);
                    fprintf(stderr, "\nDraft logits dumped to %s\n", dump_path);
                }
            }
        }

        // === Verify with target model ==========================================
        // tok NOT in KV yet. Decode vb = [tok | c0..c_{n_cands-1}].
        // On rejection at ni: crop KV cache (pure-attention: seq_rm; hybrid: snapshot/replay).

        // Save recurrent-only snapshot before verification (hybrid models only).
        if (use_recr_snap) {
            if (gpu_snap) {
                llama_gpu_snapshot_save(ctx_tgt, gpu_snap, 0,
                        LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            } else {
                recr_snap_written = llama_state_seq_get_data_ext(
                        ctx_tgt, recr_snap.data(), recr_snap_size, 0,
                        LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            tok_vb = tok;
        }
        llama_set_hidden_capture_layers(ctx_tgt, tgt_layer_ids.data(), n_tgt_lyrs);
        {
            llama_batch vb = llama_batch_init(n_noise, 0, 1);
            vb.n_tokens = 0;
            common_batch_add(vb, tok, (llama_pos)n_past, {0}, true);
            for (int32_t ni = 0; ni < n_cands; ni++) {
                common_batch_add(vb, candidates[ni], (llama_pos)(n_past + 1 + ni), {0}, true);
            }
            const int64_t t0_verify = ggml_time_us();
            if (llama_decode(ctx_tgt, vb) != 0) {
                fprintf(stderr, "\n%s: verification decode failed\n", __func__);
                llama_batch_free(vb);
                ok = false;
            }
            t_verify_us += ggml_time_us() - t0_verify;
            llama_batch_free(vb);
        }
        if (!ok) break;

        // Dump first-iteration verification logits for debugging
        if (n_gen == 1) {
            const char * dump_path = getenv("DFLASH_DUMP_VERIFY_LOGITS");
            if (dump_path) {
                FILE * fp = fopen(dump_path, "w");
                if (fp) {
                    // vb layout: [tok | c0..c_{n_cands-1}], logit[ni] = prediction at vb[ni]
                    fprintf(fp, "{\n  \"tok\": %d,\n  \"cands\": [", (int)tok);
                    for (int32_t ni = 0; ni < n_cands; ni++)
                        fprintf(fp, "%d%s", (int)candidates[ni], ni+1<n_cands?",":"");
                    fprintf(fp, "],\n  \"n_past\": %d,\n  \"verify_logits_top100\": [\n", n_past);
                    // n_noise logit positions: [0..n_noise-1]
                    for (int32_t ni = 0; ni < n_noise; ni++) {
                        const float * lg = llama_get_logits_ith(ctx_tgt, ni);
                        if (!lg) { fprintf(fp, "    null%s\n", ni+1<n_noise?",":""); continue; }
                        std::vector<std::pair<float,int>> scored(n_vocab);
                        for (int v = 0; v < n_vocab; v++) scored[v] = {lg[v], v};
                        std::partial_sort(scored.begin(), scored.begin()+100, scored.end(),
                            [](const auto&a,const auto&b){return a.first>b.first;});
                        int input_tok = (ni == 0) ? (int)tok : (int)candidates[ni-1];
                        fprintf(fp, "    {\"pos\":%d,\"input_tok\":%d,\"top1\":%d,\"top1_logit\":%.6f,\"top100\":[",
                                ni, input_tok, scored[0].second, scored[0].first);
                        for (int k = 0; k < 100; k++)
                            fprintf(fp, "[%d,%.6f]%s", scored[k].second, scored[k].first, k<99?",":"");
                        fprintf(fp, "]}%s\n", ni+1<n_noise?",":"");
                    }
                    fprintf(fp, "  ]\n}\n");
                    fclose(fp);
                    fprintf(stderr, "\nVerification logits dumped to %s\n", dump_path);
                }
            }
        }

        // === Accept/reject =====================================================
        // vb layout: [tok@n_past, c0@n_past+1, ..., c_{n_cands-1}@n_past+n_cands]
        // logit[ni] = prediction after vb[ni]:
        //   ni=0: after tok → compare to candidates[0]
        //   ni=k: after c_{k-1} → compare to candidates[k]

        const int32_t n_past_before_vb = n_past;
        int32_t n_accepted = 0;
        bool rejected = false;

        for (int32_t ni = 0; ni < n_cands && n_gen < n_predict; ni++) {
            const float * tlogits = llama_get_logits_ith(ctx_tgt, ni);
            llama_token tgt_tok = (llama_token)std::distance(
                tlogits, std::max_element(tlogits, tlogits + n_vocab));

            if (tgt_tok == candidates[ni]) {
                tok = tgt_tok;
                printf("%s", common_token_to_piece(ctx_tgt, tok).c_str());
                fflush(stdout);
                n_accepted++;
                n_gen++;
                if (llama_vocab_is_eog(vocab, tok)) goto done;
            } else {
                // Rejected at ni.
                // Extract hidden states from vb positions 0..ni (tok + accepted before rejection).
                const int32_t n_ctx_new = 1 + ni;

                // Use pre-allocated scratch (sized for n_noise tokens, always enough).
                for (int32_t li = 0; li < n_tgt_lyrs && ok; li++) {
                    const float * h_full = llama_get_layer_hidden(ctx_tgt, tgt_layer_ids[li]);
                    if (!h_full) { ok = false; break; }
                    for (int32_t t = 0; t < n_ctx_new; t++) {
                        memcpy(hidden_tmp_scratch.data() + (size_t)(t * n_tgt_lyrs + li) * n_embd,
                               h_full + (size_t)t * n_embd, n_embd * sizeof(float));
                    }
                }

                if (ok) {
                    // Project and append to ctx_proj. Use pre-allocated proj_scratch.
                    const int64_t t0_proj = ggml_time_us();
                    ok = proj_gpu
                        ? llama_dflash_proj_gpu_apply(proj_gpu, hidden_tmp_scratch.data(), proj_scratch.data(), n_ctx_new)
                        : llama_model_dflash_project(model_dft, hidden_tmp_scratch.data(), proj_scratch.data(), n_ctx_new);
                    t_project_us += ggml_time_us() - t0_proj;
                    if (ok) {
                        ctx_proj.resize((size_t)(n_ctx + n_ctx_new) * n_embd);
                        memcpy(ctx_proj.data() + (size_t)n_ctx * n_embd,
                               proj_scratch.data(), (size_t)n_ctx_new * n_embd * sizeof(float));
                        n_ctx += n_ctx_new;
                        // Sliding window: cap ctx_proj to avoid O(n²) draft cost.
                        if (max_ctx_window > 0 && n_ctx > max_ctx_window) {
                            const int32_t drop = n_ctx - max_ctx_window;
                            memmove(ctx_proj.data(),
                                    ctx_proj.data() + (size_t)drop * n_embd,
                                    (size_t)max_ctx_window * n_embd * sizeof(float));
                            ctx_proj.resize((size_t)max_ctx_window * n_embd);
                            n_ctx = max_ctx_window;
                        }
                    }
                }

                tok = tgt_tok;
                printf("%s", common_token_to_piece(ctx_tgt, tok).c_str());
                fflush(stdout);
                n_gen++;

                // Crop KV cache: keep positions 0..n_past_before_vb+ni (tok + ni accepted).
                // For pure-attention models, seq_rm works directly (no replay needed).
                // For hybrid/recurrent models (e.g. Qwen3.5), seq_rm on a partial range fails
                // because llama_memory_recurrent::seq_rm returns false for partial tail crops.
                // Fix: restore recurrent state (PARTIAL_ONLY, O(1)) → seq_rm attention KV → replay.
                if (!llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past_before_vb + 1 + ni, -1)) {
                    GGML_ASSERT(use_recr_snap && (gpu_snap || recr_snap_written > 0) &&
                                "seq_rm failed but no snapshot was saved — unexpected state");
                    llama_set_hidden_capture_layers(ctx_tgt, nullptr, 0);
                    if (gpu_snap) {
                        llama_gpu_snapshot_restore(ctx_tgt, gpu_snap, 0,
                                                   LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    } else {
                        llama_state_seq_set_data_ext(ctx_tgt, recr_snap.data(), recr_snap_written, 0,
                                                     LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    }
                    // 2. Trim attention KV: remove all verification batch tokens.
                    //    Now safe: recurrent tail < n_past_before_vb, so seq_rm succeeds.
                    llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past_before_vb, -1);
                    // 3. Replay accepted prefix [tok_vb, c0, ..., c_{ni-1}] to rebuild state.
                    //    (no logits needed — use false for want_logits)
                    const int n_replay = 1 + ni;
                    llama_batch rb = llama_batch_init(n_replay, 0, 1);
                    rb.n_tokens = 0;
                    common_batch_add(rb, tok_vb, (llama_pos)n_past_before_vb, {0}, false);
                    for (int32_t ri = 0; ri < ni; ri++) {
                        common_batch_add(rb, candidates[ri], (llama_pos)(n_past_before_vb + 1 + ri), {0}, false);
                    }
                    if (llama_decode(ctx_tgt, rb) != 0) {
                        fprintf(stderr, "\n%s: hybrid replay decode failed\n", __func__);
                        llama_batch_free(rb);
                        ok = false;
                    }
                    llama_batch_free(rb);
                }

                // n_past = orig_tok + accepted (correction tok not yet decoded)
                n_past = n_past_before_vb + 1 + n_accepted;

                rejected = true;
                if (llama_vocab_is_eog(vocab, tok)) goto done;
                break;
            }
        }
        if (!ok) break;

        llama_set_hidden_capture_layers(ctx_tgt, nullptr, 0);

        if (!rejected) {
            // All n_cands accepted.
            // Extract hidden states from vb positions 0..n_cands (tok + all candidates).
            const int32_t n_ctx_new = 1 + n_cands;

            // Use pre-allocated scratch (sized for n_noise = 1+n_cands tokens).
            for (int32_t li = 0; li < n_tgt_lyrs && ok; li++) {
                const float * h_full = llama_get_layer_hidden(ctx_tgt, tgt_layer_ids[li]);
                if (!h_full) { ok = false; break; }
                for (int32_t t = 0; t < n_ctx_new; t++) {
                    memcpy(hidden_tmp_scratch.data() + (size_t)(t * n_tgt_lyrs + li) * n_embd,
                           h_full + (size_t)t * n_embd, n_embd * sizeof(float));
                }
            }

            if (ok) {
                const int64_t t0_proj = ggml_time_us();
                ok = proj_gpu
                    ? llama_dflash_proj_gpu_apply(proj_gpu, hidden_tmp_scratch.data(), proj_scratch.data(), n_ctx_new)
                    : llama_model_dflash_project(model_dft, hidden_tmp_scratch.data(), proj_scratch.data(), n_ctx_new);
                t_project_us += ggml_time_us() - t0_proj;
                if (ok) {
                    ctx_proj.resize((size_t)(n_ctx + n_ctx_new) * n_embd);
                    memcpy(ctx_proj.data() + (size_t)n_ctx * n_embd,
                           proj_scratch.data(), (size_t)n_ctx_new * n_embd * sizeof(float));
                    n_ctx += n_ctx_new;
                    // Sliding window: cap ctx_proj to avoid O(n²) draft cost.
                    if (max_ctx_window > 0 && n_ctx > max_ctx_window) {
                        const int32_t drop = n_ctx - max_ctx_window;
                        memmove(ctx_proj.data(),
                                ctx_proj.data() + (size_t)drop * n_embd,
                                (size_t)max_ctx_window * n_embd * sizeof(float));
                        ctx_proj.resize((size_t)max_ctx_window * n_embd);
                        n_ctx = max_ctx_window;
                    }
                }
            }
            if (!ok) break;

            // Sample bonus token from logit[n_cands]
            if (n_gen < n_predict) {
                tok = common_sampler_sample(smpl_tgt, ctx_tgt, n_cands);
                common_sampler_accept(smpl_tgt, tok, true);
                printf("%s", common_token_to_piece(ctx_tgt, tok).c_str());
                fflush(stdout);
                n_gen++;
                n_past = n_past_before_vb + 1 + n_cands;

                if (llama_vocab_is_eog(vocab, tok)) goto done;
            }
        }
        if (!ok) break;

        n_accept += n_accepted;
    }

done:
    printf("\n");
    {
        const double t_sec = (ggml_time_us() - t_start) * 1e-6;
        fprintf(stderr, "\n%s: DFlash  generated=%d  accepted_draft=%d  acceptance_rate=%.1f%%  %.2f tok/s\n",
                __func__, n_gen, n_accept,
                n_gen > 1 ? 100.0f * n_accept / (float)(n_gen - 1) : 0.0f,
                n_gen / t_sec);

        if (n_iters > 0) {
            const double iters = (double)n_iters;
            fprintf(stderr, "\n%s: timing breakdown (per iteration, %lld iters):\n",
                    __func__, (long long)n_iters);
            fprintf(stderr, "  draft  fwd   : %7.2f ms/iter  (total %.1f ms)\n",
                    t_draft_us   * 1e-3 / iters, t_draft_us   * 1e-3);
            fprintf(stderr, "  target verify: %7.2f ms/iter  (total %.1f ms)\n",
                    t_verify_us  * 1e-3 / iters, t_verify_us  * 1e-3);
            fprintf(stderr, "  lm_head      : %7.2f ms/iter  (total %.1f ms)\n",
                    t_lmhead_us  * 1e-3 / iters, t_lmhead_us  * 1e-3);
            fprintf(stderr, "  hidden proj  : %7.2f ms/iter  (total %.1f ms)\n",
                    t_project_us * 1e-3 / iters, t_project_us * 1e-3);
            fprintf(stderr, "  draft vs tgt : draft=%.2f ms  verify=%.2f ms  ratio=%.2fx\n",
                    t_draft_us  * 1e-3 / iters,
                    t_verify_us * 1e-3 / iters,
                    t_verify_us > 0 ? (double)t_draft_us / t_verify_us : 0.0);
        }
    }

    if (gpu_snap) llama_gpu_snapshot_free(ctx_tgt, gpu_snap);
    common_sampler_free(smpl_tgt);
    llama_lm_head_gpu_free(lmh_gpu);
    llama_dflash_proj_gpu_free(proj_gpu);
    llama_backend_free();
    return 0;
}
