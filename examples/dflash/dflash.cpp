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
        // Draft context must fit n_predict + n_prompt + n_noise tokens in one batch
        // (no KV cache; full ctx is recomputed each iteration).
        params_dft.n_ctx      = params.n_ctx;
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

    // Scratch buffer for 1-token hidden state extraction (n_tgt_lyrs * n_embd).
    // ctx_hidden_tmp was sized for n_prompt tokens during prefill; resize to 1 token now.
    ctx_hidden_tmp.resize((size_t)n_tgt_lyrs * n_embd);

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

        // Apply target lm_head to draft embeddings
        {
            bool lmh_ok;
            const int64_t t0_lmh = ggml_time_us();
            if (lmh_gpu) {
                lmh_ok = llama_lm_head_gpu_apply(lmh_gpu, draft_embd.data(), draft_logits.data(), n_cands);
            } else {
                lmh_ok = llama_model_apply_lm_head(model_tgt, draft_embd.data(), draft_logits.data(), n_cands, false);
            }
            t_lmhead_us += ggml_time_us() - t0_lmh;
            if (!lmh_ok) {
                fprintf(stderr, "\n%s: lm_head failed\n", __func__);
                break;
            }
        }

        // Sample candidates (greedy)
        for (int32_t ni = 0; ni < n_cands; ni++) {
            const float * logits_ni = draft_logits.data() + (size_t)ni * n_vocab;
            candidates[ni] = std::max_element(logits_ni, logits_ni + n_vocab) - logits_ni;
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
        // On rejection at ni: crop KV cache with llama_kv_cache_seq_rm (no replay needed).
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
                // Extract hidden states from vb positions 0..ni (tok + accepted tokens before rejection).
                const int32_t n_ctx_new = 1 + ni;  // tok + accepted tokens before rejection (ni total)

                ctx_hidden_tmp.resize((size_t)n_ctx_new * n_tgt_lyrs * n_embd);
                for (int32_t li = 0; li < n_tgt_lyrs && ok; li++) {
                    const float * h_full = llama_get_layer_hidden(ctx_tgt, tgt_layer_ids[li]);
                    if (!h_full) { ok = false; break; }
                    for (int32_t t = 0; t < n_ctx_new; t++) {
                        memcpy(ctx_hidden_tmp.data() + (size_t)(t * n_tgt_lyrs + li) * n_embd,
                               h_full + (size_t)t * n_embd, n_embd * sizeof(float));
                    }
                }

                if (ok) {
                    // Append new projected tokens to ctx_proj (accumulate across iterations)
                    const int32_t n_ctx_old = n_ctx;
                    std::vector<float> new_proj((size_t)n_ctx_new * n_embd);
                    const int64_t t0_proj = ggml_time_us();
                    ok = proj_gpu
                        ? llama_dflash_proj_gpu_apply(proj_gpu, ctx_hidden_tmp.data(), new_proj.data(), n_ctx_new)
                        : llama_model_dflash_project(model_dft, ctx_hidden_tmp.data(), new_proj.data(), n_ctx_new);
                    t_project_us += ggml_time_us() - t0_proj;
                    if (ok) {
                        ctx_proj.resize((size_t)(n_ctx_old + n_ctx_new) * n_embd);
                        memcpy(ctx_proj.data() + (size_t)n_ctx_old * n_embd,
                               new_proj.data(), (size_t)n_ctx_new * n_embd * sizeof(float));
                        n_ctx = n_ctx_old + n_ctx_new;
                    }
                }

                tok = tgt_tok;
                printf("%s", common_token_to_piece(ctx_tgt, tok).c_str());
                fflush(stdout);
                n_gen++;

                // Crop KV cache: keep positions 0..n_past_before_vb+ni (tok + ni accepted)
                // Equivalent to Python's past_key_values_target.crop(start) — no replay needed.
                llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past_before_vb + 1 + ni, -1);

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
            // These become the context for the next draft run.
            const int32_t n_ctx_new = 1 + n_cands;  // original token + all accepted candidates

            // Resize to hold hidden states for n_ctx_new tokens across n_tgt_lyrs layers
            ctx_hidden_tmp.resize((size_t)n_ctx_new * n_tgt_lyrs * n_embd);

            // For each layer we want to capture, get its hidden representation for each token position
            for (int32_t li = 0; li < n_tgt_lyrs && ok; li++) {
                const float * h_full = llama_get_layer_hidden(ctx_tgt, tgt_layer_ids[li]);
                if (!h_full) {
                    ok = false;
                    break;
                }

                // Extract hidden representations for all token positions (0 to n_cands inclusive)
                for (int32_t t = 0; t < n_ctx_new && ok; t++) {
                    // Copy the hidden state for token position t, layer li
                    // h_full is organized as [pos_0_layer_li, pos_1_layer_li, ..., pos_n_layer_li]
                    memcpy(ctx_hidden_tmp.data() + (size_t)(t * n_tgt_lyrs + li) * n_embd,
                           h_full + (size_t)t * n_embd, n_embd * sizeof(float));
                }
            }

            if (ok) {
                // Append new projected tokens to ctx_proj (accumulate across iterations)
                const int32_t n_ctx_old = n_ctx;
                std::vector<float> new_proj((size_t)n_ctx_new * n_embd);
                const int64_t t0_proj = ggml_time_us();
                ok = proj_gpu
                    ? llama_dflash_proj_gpu_apply(proj_gpu, ctx_hidden_tmp.data(), new_proj.data(), n_ctx_new)
                    : llama_model_dflash_project(model_dft, ctx_hidden_tmp.data(), new_proj.data(), n_ctx_new);
                t_project_us += ggml_time_us() - t0_proj;
                if (ok) {
                    ctx_proj.resize((size_t)(n_ctx_old + n_ctx_new) * n_embd);
                    memcpy(ctx_proj.data() + (size_t)n_ctx_old * n_embd,
                           new_proj.data(), (size_t)n_ctx_new * n_embd * sizeof(float));
                    n_ctx = n_ctx_old + n_ctx_new;
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

    common_sampler_free(smpl_tgt);
    llama_lm_head_gpu_free(lmh_gpu);
    llama_dflash_proj_gpu_free(proj_gpu);
    llama_backend_free();
    return 0;
}
