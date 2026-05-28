// DFlash HTTP inference server
// ============================
// Wraps the DFlash speculative decoding engine in a minimal HTTP server.
//
// Endpoints:
//   GET  /health
//   POST /completion                  (llama.cpp-style, "prompt" string)
//   POST /v1/chat/completions         (OpenAI-style, "messages" array)
//
// Usage:
//   llama-dflash-server -m target.gguf --model-draft draft.gguf \
//                       --host 0.0.0.0 --port 8080 -ngl 99 -n 512

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "sampling.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <future>
#include <vector>

using json = nlohmann::ordered_json;

// ── helpers (copied from dflash.cpp) ────────────────────────────────────────

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

static std::string gen_completion_id() {
    static std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    return "chatcmpl-" + std::to_string(rng());
}

static int64_t now_unix() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// ── GenerateResult ────────────────────────────────────────────────────────────

struct GenerateResult {
    int32_t n_generated  = 0;
    int32_t n_accepted   = 0;   // draft tokens accepted
    bool    error        = false;
    bool    stopped_word = false; // stopped by a stop sequence (vs EOS/max_tokens)
    bool    stopped_eos  = false; // stopped by EOS/EOG token
    bool    stopped_limit = false; // stopped by max_tokens

    float acceptance_rate() const {
        return n_generated > 1 ? (float)n_accepted / (float)(n_generated - 1) : 0.0f;
    }
};

// ── TokenStream — thread-safe token queue for SSE streaming ──────────────────

struct TokenStream {
    std::mutex              mtx;
    std::condition_variable cv;

    struct StreamChunk {
        std::string text;
        std::string type; // "content", "reasoning_content", or "tool_calls"
    };

    std::deque<StreamChunk> buf;
    bool                    done = false;
    GenerateResult          result;
    std::string             full_text;
    bool                    stopped_by_word = false;

    void push(const std::string & text, const std::string & type = "content") {
        if (text.empty()) return;
        { std::lock_guard<std::mutex> lock(mtx); buf.push_back({text, type}); }
        cv.notify_one();
    }

    void finish(const GenerateResult & r, const std::string & text = "", bool stopped_word = false) {
        { std::lock_guard<std::mutex> lock(mtx); result = r; full_text = text; stopped_by_word = stopped_word; done = true; }
        cv.notify_all();
    }

    std::pair<StreamChunk, bool> pop() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return !buf.empty() || done; });
        if (!buf.empty()) {
            auto chunk = std::move(buf.front());
            buf.pop_front();
            return {chunk, true};
        }
        return {{}, false};
    }
};

// ── DFlashEngine ──────────────────────────────────────────────────────────────

struct DFlashEngine {
    // ── Models & contexts ──────────────────────────────────────────────────
    std::unique_ptr<common_init_result> tgt_init;
    std::unique_ptr<common_init_result> dft_init;
    llama_model   * model_tgt = nullptr;
    llama_context * ctx_tgt   = nullptr;
    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;
    const llama_vocab * vocab_tgt = nullptr;
    bool no_draft = false;

    // ── DFlash parameters ──────────────────────────────────────────────────
    int32_t n_noise    = 1;
    int32_t n_cands    = 0;   // n_noise - 1
    int32_t n_tgt_lyrs = 0;
    int32_t n_embd     = 0;
    int32_t n_vocab    = 0;
    int32_t max_ctx_window = 0;
    std::vector<int32_t> tgt_layer_ids;
    std::vector<float>   mask_embd;

    // ── GPU accelerators ───────────────────────────────────────────────────
    llama_lm_head_gpu    * lmh_gpu  = nullptr;
    llama_dflash_proj_gpu * proj_gpu = nullptr;

    // ── Sampler ────────────────────────────────────────────────────────────
    common_sampler * smpl_tgt = nullptr;
    common_params_sampling default_sparams; // saved for restoring after per-request grammar

    // ── RNG for probabilistic acceptance ────────────────────────────────────
    std::mt19937 rng{std::random_device{}()};

    // ── Per-request scratch buffers (sized in init) ─────────────────────────
    std::vector<float>       tok_embd_buf;
    std::vector<float>       noise_embd;
    std::vector<float>       draft_embd;
    std::vector<float>       draft_logits;
    std::vector<llama_token> candidates;
    std::vector<float>       hidden_tmp_scratch;
    std::vector<float>       proj_scratch;
    // ctx_proj and ctx_hidden_tmp are resized per-request (vary with n_prompt)
    std::vector<float>       ctx_proj;
    std::vector<float>       ctx_hidden_tmp;

    // ── Hybrid snapshot support ────────────────────────────────────────────
    bool   use_recr_snap       = false;
    bool   recr_snap_init_done = false;
    llama_gpu_snapshot * gpu_snap = nullptr;
    // CPU fallback (used if GPU snapshot unavailable)
    size_t recr_snap_size      = 0;
    std::vector<uint8_t> recr_snap;

    // ── Prefix cache support ───────────────────────────────────────────────
    std::vector<llama_token> cached_prompt;
    std::vector<uint8_t>     cached_recr_snap;   // SSM state at end of cached prompt
    size_t                   cached_recr_snap_written = 0;
    llama_gpu_snapshot *     cached_gpu_snap = nullptr;
    bool                     cached_snap_valid = false;

    // ── Initialise engine (load models, allocate buffers) ──────────────────
    bool init(common_params & params, int32_t arg_draft_ctx_max) {
        no_draft = params.speculative.mparams_dft.path.empty();

        tgt_init  = common_init_from_params(params);
        model_tgt = tgt_init->model();
        ctx_tgt   = tgt_init->context();
        if (!model_tgt || !ctx_tgt) {
            fprintf(stderr, "%s: failed to load target model\n", __func__);
            return false;
        }
        vocab_tgt = llama_model_get_vocab(model_tgt);

        if (!no_draft) {
            common_params params_dft = params;
            params_dft.model.path = params.speculative.mparams_dft.path;
            // Draft uses incremental KV cache: Q from noise (small) attends to
            // full KV cache history. n_ctx matches target for capacity.
            // n_ubatch kept small — initial prefill is split into chunks.
            params_dft.n_ctx    = params.n_ctx;
            params_dft.n_ubatch = 128;
            dft_init  = common_init_from_params(params_dft);
            model_dft = dft_init->model();
            ctx_dft   = dft_init->context();
            if (!model_dft || !ctx_dft) {
                fprintf(stderr, "%s: failed to load draft model\n", __func__);
                return false;
            }

            n_noise    = (int32_t)llama_model_dflash_block_size(model_dft);
            n_tgt_lyrs = (int32_t)llama_model_dflash_num_target_layers(model_dft);
            if (n_noise <= 0 || n_tgt_lyrs <= 0) {
                fprintf(stderr, "%s: draft model is not a DFlash model\n", __func__);
                return false;
            }
            n_cands = n_noise - 1;
            tgt_layer_ids.resize(n_tgt_lyrs);
            llama_model_dflash_target_layer_ids(model_dft, tgt_layer_ids.data(), n_tgt_lyrs);
            fprintf(stderr, "%s: DFlash n_noise=%d n_target_layers=%d\n",
                    __func__, n_noise, n_tgt_lyrs);
        }

        n_embd  = llama_model_n_embd(model_tgt);
        n_vocab = (int32_t)llama_vocab_n_tokens(vocab_tgt);

        if (!no_draft) {
            // Mask token embedding
            const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);
            const llama_token mask_id = llama_vocab_mask(vocab_dft);
            if (mask_id < 0) {
                fprintf(stderr, "%s: draft model has no mask token\n", __func__);
                return false;
            }
            mask_embd.resize(n_embd, 0.0f);
            if (!llama_model_get_token_embd(model_tgt, mask_id, mask_embd.data(), n_embd)) {
                fprintf(stderr, "%s: failed to get mask token embedding\n", __func__);
                return false;
            }

            // GPU lm_head (apply_output_norm=false: draft embeddings already have it)
            lmh_gpu = llama_lm_head_gpu_create(model_tgt, n_cands, false);
            if (!lmh_gpu)
                fprintf(stderr, "%s: GPU lm_head unavailable, using CPU fallback\n", __func__);

            // GPU dflash proj — sized for per-iteration hot path (n_noise tokens max).
            // The GPU graph runs for exactly n_batch tokens every call regardless of the
            // actual n argument (extra rows are zeroed). Using llama_n_ctx here would cause
            // every projection call to run a 2000-row matmul instead of 4 rows — ~40x slower.
            // Prefill projection (n_prompt tokens) uses CPU fallback; it's only 1 call/request.
            const int32_t proj_batch = n_noise;
            ggml_backend_t shared_be = llama_lm_head_gpu_get_backend(lmh_gpu);
            proj_gpu = llama_dflash_proj_gpu_create(model_dft, proj_batch, shared_be);
            if (!proj_gpu)
                fprintf(stderr, "%s: GPU dflash proj unavailable, using CPU fallback\n", __func__);

            // Pre-allocate scratch buffers
            tok_embd_buf    .resize(n_embd);
            noise_embd      .resize((size_t)n_noise * n_embd, 0.0f);
            draft_embd      .resize((size_t)n_cands * n_embd);
            draft_logits    .resize((size_t)n_cands * n_vocab);
            candidates      .resize(n_cands);
            hidden_tmp_scratch.resize((size_t)n_noise * n_tgt_lyrs * n_embd);
            proj_scratch    .resize((size_t)n_noise * n_embd);
        }

        max_ctx_window = arg_draft_ctx_max;
        if (max_ctx_window == 0) {
            const char * e = getenv("DFLASH_MAX_CTX");
            if (e) max_ctx_window = std::max(0, atoi(e));
        }

        smpl_tgt = common_sampler_init(model_tgt, params.sampling);
        default_sparams = params.sampling;
        if (!smpl_tgt) {
            fprintf(stderr, "%s: failed to init sampler\n", __func__);
            return false;
        }

        fprintf(stderr, "%s: engine ready  n_embd=%d  n_vocab=%d  mode=%s\n",
                __func__, n_embd, n_vocab, no_draft ? "AR-baseline" : "DFlash");
        return true;
    }

    // ── Reset state between requests (preserves KV cache for prefix cache) ──
    void reset() {
        // Don't clear KV cache here — generate() handles prefix cache
        common_sampler_reset(smpl_tgt);
        ctx_proj.clear();
        ctx_hidden_tmp.clear();
    }

    // ── Generate tokens ─────────────────────────────────────────────────────
    // on_token is called for each generated token string. Return false to stop.
    GenerateResult generate(
            const std::vector<llama_token> & prompt_tokens,
            int32_t max_tokens,
            const std::function<bool(const std::string &)> & on_token)
    {
        GenerateResult res;
        const int32_t n_prompt = (int32_t)prompt_tokens.size();
        if (n_prompt == 0 || max_tokens <= 0) return res;

        const int64_t t_gen_start = ggml_time_us();
        int64_t t_prefill_us = 0;
        int64_t t_draft_us = 0, t_verify_us = 0, t_lmhead_us = 0, t_project_us = 0;
        int64_t t_snap_save_us = 0, t_snap_restore_us = 0, t_hidden_extract_us = 0;
        int64_t t_ctxproj_us = 0;
        int64_t t_decode_us = 0;
        int64_t t_decode_start = 0;
        int64_t n_iters = 0;
        int64_t t_noise_build_us = 0;
        int64_t t_batch_build_us = 0;
        int64_t t_embd_collect_us = 0;
        int64_t t_accept_us = 0;
        int64_t t_kv_crop_us = 0;
        int64_t t_sampler_us = 0;

        // ── Prefill with prefix cache ─────────────────────────────────────
        const int64_t t_prefill_start = ggml_time_us();
        if (!no_draft)
            llama_set_hidden_capture_layers(ctx_tgt, tgt_layer_ids.data(), n_tgt_lyrs);

        // Find common prefix with cached prompt
        int32_t n_past_cached = 0;
        {
            const int32_t n_check = std::min((int32_t)cached_prompt.size(), n_prompt);
            for (int32_t i = 0; i < n_check; i++) {
                if (cached_prompt[i] != prompt_tokens[i]) break;
                n_past_cached = i + 1;
            }
        }

        // If prompt diverges from cache, we need to handle KV/SSM rollback
        if (n_past_cached < (int32_t)cached_prompt.size()) {
            // Try to remove tokens beyond the common prefix
            if (n_past_cached == 0) {
                llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, -1, -1);
                if (ctx_dft)
                    llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, -1, -1);
                cached_snap_valid = false;
            } else {
                bool rm_ok = llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past_cached, -1);
                if (!rm_ok) {
                    // Hybrid model: partial removal always fails.
                    // Fall back to full clear + re-prefill from scratch.
                    llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, -1, -1);
                    if (ctx_dft)
                        llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, -1, -1);
                    n_past_cached = 0;
                    cached_snap_valid = false;
                }
                if (ctx_dft)
                    llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, n_past_cached, -1);
            }
        }

        if (n_past_cached > 0) {
            fprintf(stderr, "%s: prefix cache hit: %d / %d tokens reused\n",
                    __func__, n_past_cached, n_prompt);
        }

        // Process only the new tokens
        {
            const int32_t tgt_batch = (int32_t)llama_n_batch(ctx_tgt);
            for (int32_t i_start = n_past_cached; i_start < n_prompt; i_start += tgt_batch) {
                const int32_t i_end = std::min(i_start + tgt_batch, n_prompt);
                const int32_t n_cur = i_end - i_start;
                llama_batch batch = llama_batch_init(n_cur, 0, 1);
                for (int32_t i = i_start; i < i_end; i++)
                    common_batch_add(batch, prompt_tokens[i], (llama_pos)i, {0}, i == n_prompt - 1);
                if (llama_decode(ctx_tgt, batch) != 0) {
                    llama_batch_free(batch);
                    res.error = true; return res;
                }
                llama_batch_free(batch);
            }
        }

        // Save prompt for next request's prefix cache
        cached_prompt.assign(prompt_tokens.begin(), prompt_tokens.end());

        if (!no_draft) {
            const int32_t n_new = n_prompt - n_past_cached;

            if (n_new > 0) {
                // Extract hidden states only for NEW tokens
                ctx_hidden_tmp.resize((size_t)n_new * n_tgt_lyrs * n_embd);
                for (int32_t li = 0; li < n_tgt_lyrs; li++) {
                    const float * h = llama_get_layer_hidden(ctx_tgt, tgt_layer_ids[li]);
                    if (!h) { res.error = true; return res; }
                    for (int32_t t = 0; t < n_new; t++)
                        memcpy(ctx_hidden_tmp.data() + (size_t)(t * n_tgt_lyrs + li) * n_embd,
                               h + (size_t)t * n_embd, n_embd * sizeof(float));
                }

                // Preserve cached projections, compute new ones
                std::vector<float> old_proj;
                if (n_past_cached > 0 && ctx_proj.size() >= (size_t)n_past_cached * n_embd) {
                    old_proj.assign(ctx_proj.begin(), ctx_proj.begin() + (size_t)n_past_cached * n_embd);
                }

                ctx_proj.resize((size_t)n_prompt * n_embd);

                // Copy cached projections
                if (!old_proj.empty()) {
                    memcpy(ctx_proj.data(), old_proj.data(), old_proj.size() * sizeof(float));
                }

                // Project only new tokens
                {
                    const int32_t proj_batch = n_noise;
                    bool proj_ok = true;
                    for (int32_t t = 0; t < n_new && proj_ok; t += proj_batch) {
                        const int32_t n_cur = std::min(proj_batch, n_new - t);
                        float * dst = ctx_proj.data() + (size_t)(n_past_cached + t) * n_embd;
                        proj_ok = proj_gpu
                            ? llama_dflash_proj_gpu_apply(proj_gpu,
                                  ctx_hidden_tmp.data() + (size_t)t * n_tgt_lyrs * n_embd,
                                  dst, n_cur)
                            : llama_model_dflash_project(model_dft,
                                  ctx_hidden_tmp.data() + (size_t)t * n_tgt_lyrs * n_embd,
                                  dst, n_cur);
                    }
                    if (!proj_ok) { res.error = true; return res; }
                }
            } else {
                // Full cache hit — reuse all projections, just resize
                ctx_proj.resize((size_t)n_prompt * n_embd);
            }

            llama_set_hidden_capture_layers(ctx_tgt, nullptr, 0);

            // Detect hybrid model on first call
            if (!recr_snap_init_done) {
                const size_t sz_full = llama_state_seq_get_size(ctx_tgt, 0);
                const size_t sz_recr = llama_state_seq_get_size_ext(
                        ctx_tgt, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                use_recr_snap = (sz_full > 0 && sz_recr < sz_full);
                if (use_recr_snap) {
                    gpu_snap = llama_gpu_snapshot_create(ctx_tgt);
                    if (gpu_snap) {
                        fprintf(stderr, "%s: hybrid model — GPU snapshot enabled\n", __func__);
                    } else {
                        recr_snap_size = sz_recr;
                        recr_snap.resize(sz_recr);
                        fprintf(stderr, "%s: hybrid model — CPU snapshot fallback\n", __func__);
                    }
                }
                recr_snap_init_done = true;
            }

            // Save SSM snapshot for prefix cache (for next request)
            if (use_recr_snap) {
                cached_snap_valid = false;
                if (gpu_snap) {
                    if (!cached_gpu_snap) {
                        cached_gpu_snap = llama_gpu_snapshot_create(ctx_tgt);
                    }
                    if (cached_gpu_snap) {
                        llama_gpu_snapshot_save(ctx_tgt, cached_gpu_snap, 0,
                                LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        cached_snap_valid = true;
                    }
                } else {
                    const size_t sz = llama_state_seq_get_size_ext(ctx_tgt, 0,
                            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    cached_recr_snap.resize(sz);
                    cached_recr_snap_written = llama_state_seq_get_data_ext(ctx_tgt,
                            cached_recr_snap.data(), sz, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    cached_snap_valid = (cached_recr_snap_written == sz);
                }
            }
        }

        t_prefill_us = ggml_time_us() - t_prefill_start;

        // ── Sample first token ────────────────────────────────────────────
        llama_token tok = common_sampler_sample(smpl_tgt, ctx_tgt, -1);
        common_sampler_accept(smpl_tgt, tok, true);
        if (!llama_vocab_is_eog(vocab_tgt, tok)) {
            if (!on_token(common_token_to_piece(ctx_tgt, tok, false))) goto done;
        } else {
            goto done;
        }

        {
            int32_t n_past   = n_prompt;
            // draft KV cache position tracking:
            // ctx_proj holds the PENDING projections to send in the next draft decode.
            // After the first draft decode, ctx_proj is replaced by just the new projections.
            int32_t draft_kv_pos = 0;
            int32_t n_pending    = no_draft ? 0 : (int32_t)(ctx_proj.size() / n_embd);
            res.n_generated  = 1;  // first token already sampled above
            int32_t & n_gen    = res.n_generated;
            int32_t & n_accept = res.n_accepted;

            // ── AR-baseline mode ──────────────────────────────────────────
            if (no_draft) {
                t_decode_start = ggml_time_us();
                while (n_gen < max_tokens) {
                    llama_batch b = llama_batch_init(1, 0, 1);
                    common_batch_add(b, tok, (llama_pos)n_past, {0}, true);
                    bool dec_ok = llama_decode(ctx_tgt, b) == 0;
                    llama_batch_free(b);
                    if (!dec_ok) break;
                    n_past++;
                    tok = common_sampler_sample(smpl_tgt, ctx_tgt, -1);
                    common_sampler_accept(smpl_tgt, tok, true);
                    n_gen++;
                    if (llama_vocab_is_eog(vocab_tgt, tok)) break;
                    if (!on_token(common_token_to_piece(ctx_tgt, tok, false))) break;
                }
                res.n_generated = n_gen;
                t_decode_us = ggml_time_us() - t_decode_start;
                {
                    const int64_t t_total_us = ggml_time_us() - t_gen_start;
                    const double t_prefill_s = t_prefill_us * 1e-6;
                    const double t_decode_s  = t_decode_us  * 1e-6;
                    const double t_total_s   = t_total_us   * 1e-6;
                    fprintf(stderr, "\n");
                    fprintf(stderr, "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                            t_prefill_s * 1e3, n_prompt,
                            n_prompt > 0 ? t_prefill_s * 1e3 / n_prompt : 0.0,
                            t_prefill_s > 0.0 ? n_prompt / t_prefill_s : 0.0);
                    fprintf(stderr, "  eval time     = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                            t_decode_s * 1e3, res.n_generated,
                            res.n_generated > 0 ? t_decode_s * 1e3 / res.n_generated : 0.0,
                            t_decode_s > 0.0 ? res.n_generated / t_decode_s : 0.0);
                    fprintf(stderr, "  total time     = %10.2f ms / %5d tokens\n",
                            t_total_s * 1e3, n_prompt + res.n_generated);
                }
                return res;
            }

            // ── DFlash decode loop ────────────────────────────────────────
            t_decode_start = ggml_time_us();
            llama_token tok_vb = LLAMA_TOKEN_NULL;
            size_t recr_snap_written = 0;
            bool draft_prefill_done = false;

            while (n_gen < max_tokens) {
                bool ok = true;
                n_iters++;

                // Build noise_embd: [embed(tok), mask, ..., mask]
                {
                    int64_t t0_nb = ggml_time_us();
                    if (!llama_model_get_token_embd(model_tgt, tok, tok_embd_buf.data(), n_embd)) break;
                    memcpy(noise_embd.data(), tok_embd_buf.data(), n_embd * sizeof(float));
                    for (int32_t ni = 1; ni < n_noise; ni++)
                        memcpy(noise_embd.data() + (size_t)ni * n_embd, mask_embd.data(), n_embd * sizeof(float));
                    t_noise_build_us += ggml_time_us() - t0_nb;
                }

                // Draft decode: [pending_proj | noise] — incremental KV cache
                // If n_pending + n_noise > n_ubatch, split: fill KV cache with
                // projection chunks first, then decode the final chunk with noise.
                {
                    int64_t t0_bb = ggml_time_us();
                    llama_set_embeddings(ctx_dft, true);
                    const int32_t dft_ubatch = (int32_t)llama_n_ubatch(ctx_dft);
                    const int32_t max_proj_per_chunk = dft_ubatch - n_noise;
                    int32_t proj_sent = 0;

                    // Phase 1: fill KV cache with projection-only chunks (output discarded)
                    while (n_pending - proj_sent > max_proj_per_chunk) {
                        const int32_t n_chunk = max_proj_per_chunk;
                        llama_batch dft_b = llama_batch_init(n_chunk + n_noise, n_embd, 1);
                        dft_b.n_tokens = 0;
                        for (int32_t t = 0; t < n_chunk; t++)
                            batch_embd_add(dft_b, ctx_proj.data() + (size_t)(proj_sent + t) * n_embd, n_embd,
                                           (llama_pos)(draft_kv_pos + proj_sent + t), 0, true);
                        for (int32_t ni = 0; ni < n_noise; ni++)
                            batch_embd_add(dft_b, noise_embd.data() + (size_t)ni * n_embd, n_embd,
                                           (llama_pos)(draft_kv_pos + proj_sent + n_chunk + ni), 0, true);
                        int64_t t0 = ggml_time_us();
                        bool dft_ok = llama_decode(ctx_dft, dft_b) == 0;
                        t_draft_us += ggml_time_us() - t0;
                        llama_batch_free(dft_b);
                        if (!dft_ok) { ok = false; break; }
                        // Remove noise KV, keep projection KV
                        llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, draft_kv_pos + proj_sent + n_chunk, -1);
                        proj_sent += n_chunk;
                    }
                    if (!ok) break;

                    // Phase 2: final chunk with real noise (collect embeddings from this)
                    const int32_t n_remaining = n_pending - proj_sent;
                    const int32_t n_dft = n_remaining + n_noise;
                    llama_batch dft_b = llama_batch_init(n_dft, n_embd, 1);
                    dft_b.n_tokens = 0;
                    for (int32_t t = 0; t < n_remaining; t++)
                        batch_embd_add(dft_b, ctx_proj.data() + (size_t)(proj_sent + t) * n_embd, n_embd,
                                       (llama_pos)(draft_kv_pos + proj_sent + t), 0, true);
                    for (int32_t ni = 0; ni < n_noise; ni++)
                        batch_embd_add(dft_b, noise_embd.data() + (size_t)ni * n_embd, n_embd,
                                       (llama_pos)(draft_kv_pos + proj_sent + n_remaining + ni), 0, true);
                    t_batch_build_us += ggml_time_us() - t0_bb;
                    int64_t t0 = ggml_time_us();
                    bool dft_ok = llama_decode(ctx_dft, dft_b) == 0;
                    t_draft_us += ggml_time_us() - t0;
                    int64_t t0_bf = ggml_time_us();
                    llama_batch_free(dft_b);
                    t_batch_build_us += ggml_time_us() - t0_bf;
                    if (!dft_ok) { ok = false; }
                }
                if (!ok) break;

                // Remove noise K/V from draft cache (keep context projections)
                llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, draft_kv_pos + n_pending, -1);

                // First draft decode fills the entire KV cache (expensive).
                // Reset timing so this cost is counted as prefill, not decode.
                if (!draft_prefill_done) {
                    draft_prefill_done = true;
                    t_prefill_us += ggml_time_us() - t_decode_start;
                    t_decode_start = ggml_time_us();
                    t_draft_us = 0; t_batch_build_us = 0;
                    n_iters = 0;
                }

                // Collect draft embeddings (last n_cands noise outputs)
                // In the final chunk, noise starts at index (n_pending - proj_sent_final)
                // where proj_sent_final = how many proj tokens were in the last batch.
                {
                    const int32_t dft_ubatch = (int32_t)llama_n_ubatch(ctx_dft);
                    const int32_t max_proj = dft_ubatch - n_noise;
                    const int32_t n_last_proj = (n_pending <= max_proj) ? n_pending
                                              : n_pending - (n_pending - max_proj + max_proj - 1) / max_proj * max_proj;
                    // Simpler: just compute directly
                    int32_t proj_already_sent = 0;
                    while (n_pending - proj_already_sent > max_proj) proj_already_sent += max_proj;
                    const int32_t n_final_proj = n_pending - proj_already_sent;

                    int64_t t0_ec = ggml_time_us();
                    for (int32_t ni = 0; ni < n_cands && ok; ni++) {
                        const float * e = llama_get_embeddings_ith(ctx_dft, n_final_proj + 1 + ni);
                        if (!e) { ok = false; break; }
                        memcpy(draft_embd.data() + (size_t)ni * n_embd, e, n_embd * sizeof(float));
                    }
                    llama_set_embeddings(ctx_dft, false);
                    t_embd_collect_us += ggml_time_us() - t0_ec;
                }
                if (!ok) break;

                // Apply target lm_head to draft embeddings → greedy argmax
                {
                    int64_t t0 = ggml_time_us();
                    bool lmh_ok;
                    if (lmh_gpu) {
                        lmh_ok = llama_lm_head_gpu_apply_argmax(lmh_gpu, draft_embd.data(),
                                candidates.data(), n_cands);
                    } else {
                        lmh_ok = llama_model_apply_lm_head(model_tgt, draft_embd.data(), draft_logits.data(), n_cands, false);
                        if (lmh_ok) {
                            for (int32_t ni = 0; ni < n_cands; ni++) {
                                const float * lg = draft_logits.data() + (size_t)ni * n_vocab;
                                candidates[ni] = (llama_token)(std::max_element(lg, lg + n_vocab) - lg);
                            }
                        }
                    }
                    t_lmhead_us += ggml_time_us() - t0;
                    if (!lmh_ok) break;
                }

                // Save snapshot (hybrid models) before verification
                if (use_recr_snap) {
                    int64_t t0_snap = ggml_time_us();
                    if (gpu_snap) {
                        llama_gpu_snapshot_save(ctx_tgt, gpu_snap, 0,
                                LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    } else {
                        recr_snap_written = llama_state_seq_get_data_ext(
                                ctx_tgt, recr_snap.data(), recr_snap_size, 0,
                                LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    }
                    t_snap_save_us += ggml_time_us() - t0_snap;
                    tok_vb = tok;
                }

                // Verification batch: [tok @ n_past | c0..c_{n_cands-1}]
                llama_set_hidden_capture_layers(ctx_tgt, tgt_layer_ids.data(), n_tgt_lyrs);
                {
                    int64_t t0_bb2 = ggml_time_us();
                    llama_batch vb = llama_batch_init(n_noise, 0, 1);
                    vb.n_tokens = 0;
                    common_batch_add(vb, tok, (llama_pos)n_past, {0}, true);
                    for (int32_t ni = 0; ni < n_cands; ni++)
                        common_batch_add(vb, candidates[ni], (llama_pos)(n_past + 1 + ni), {0}, true);
                    t_batch_build_us += ggml_time_us() - t0_bb2;
                    int64_t t0 = ggml_time_us();
                    bool vb_ok = llama_decode(ctx_tgt, vb) == 0;
                    t_verify_us += ggml_time_us() - t0;
                    int64_t t0_bf2 = ggml_time_us();
                    llama_batch_free(vb);
                    t_batch_build_us += ggml_time_us() - t0_bf2;
                    if (!vb_ok) { ok = false; }
                }
                if (!ok) break;

                // Accept / reject (greedy acceptance)
                // Accept draft token if argmax(target_logits) == draft token.
                // On rejection, use target argmax as corrected token.
                const int32_t n_past_before_vb = n_past;
                int32_t n_accepted = 0;
                bool    rejected   = false;

                for (int32_t ni = 0; ni < n_cands && n_gen < max_tokens; ni++) {
                    int64_t t0_acc = ggml_time_us();
                    const float * tlogits = llama_get_logits_ith(ctx_tgt, ni);
                    const llama_token x   = candidates[ni]; // draft token

                    // Greedy acceptance: accept if argmax(target) == draft token
                    llama_token tgt_tok = (llama_token)(std::max_element(tlogits, tlogits + n_vocab) - tlogits);
                    t_accept_us += ggml_time_us() - t0_acc;

                    if (tgt_tok == x) {
                        // Accepted
                        tok = x;
                        n_accepted++;
                        n_gen++;
                        if (llama_vocab_is_eog(vocab_tgt, tok)) goto done;
                        if (!on_token(common_token_to_piece(ctx_tgt, tok, false))) goto done;
                    } else {
                        // Rejected: use target argmax as corrected token
                        tok = tgt_tok;

                        // Extract hidden states → project → append to ctx_proj
                        const int32_t n_ctx_new = 1 + ni;
                        {
                            int64_t t0_hex = ggml_time_us();
                            for (int32_t li = 0; li < n_tgt_lyrs && ok; li++) {
                                const float * h = llama_get_layer_hidden(ctx_tgt, tgt_layer_ids[li]);
                                if (!h) { ok = false; break; }
                                for (int32_t t = 0; t < n_ctx_new; t++)
                                    memcpy(hidden_tmp_scratch.data() + (size_t)(t * n_tgt_lyrs + li) * n_embd,
                                           h + (size_t)t * n_embd, n_embd * sizeof(float));
                            }
                            t_hidden_extract_us += ggml_time_us() - t0_hex;
                        }
                        if (ok) {
                            int64_t t0 = ggml_time_us();
                            bool proj_ok2 = proj_gpu
                                ? llama_dflash_proj_gpu_apply(proj_gpu, hidden_tmp_scratch.data(),
                                                              proj_scratch.data(), n_ctx_new)
                                : llama_model_dflash_project(model_dft, hidden_tmp_scratch.data(),
                                                             proj_scratch.data(), n_ctx_new);
                            t_project_us += ggml_time_us() - t0;
                            if (proj_ok2) {
                                int64_t t0_cp = ggml_time_us();
                                // Replace ctx_proj with new projections (for next draft decode)
                                ctx_proj.resize((size_t)n_ctx_new * n_embd);
                                memcpy(ctx_proj.data(), proj_scratch.data(), (size_t)n_ctx_new * n_embd * sizeof(float));
                                draft_kv_pos += n_pending;  // advance past context we just cached
                                n_pending = n_ctx_new;
                                t_ctxproj_us += ggml_time_us() - t0_cp;
                            }
                        }

                        n_gen++;
                        if (llama_vocab_is_eog(vocab_tgt, tok)) goto done;
                        if (!on_token(common_token_to_piece(ctx_tgt, tok, false))) goto done;

                        // Crop KV cache
                        {
                            int64_t t0_kvc = ggml_time_us();
                            bool crop_ok = llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0,
                                                     n_past_before_vb + 1 + ni, -1);
                            t_kv_crop_us += ggml_time_us() - t0_kvc;
                            if (!crop_ok) {
                                GGML_ASSERT(use_recr_snap && (gpu_snap || recr_snap_written > 0));
                                int64_t t0_rsnap = ggml_time_us();
                                llama_set_hidden_capture_layers(ctx_tgt, nullptr, 0);
                                if (gpu_snap) {
                                    llama_gpu_snapshot_restore(ctx_tgt, gpu_snap, 0,
                                            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                } else {
                                    llama_state_seq_set_data_ext(ctx_tgt, recr_snap.data(),
                                            recr_snap_written, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                }
                                llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past_before_vb, -1);
                                const int n_replay = 1 + ni;
                                llama_batch rb = llama_batch_init(n_replay, 0, 1);
                                rb.n_tokens = 0;
                                common_batch_add(rb, tok_vb, (llama_pos)n_past_before_vb, {0}, false);
                                for (int32_t ri = 0; ri < ni; ri++)
                                    common_batch_add(rb, candidates[ri],
                                                     (llama_pos)(n_past_before_vb + 1 + ri), {0}, false);
                                if (llama_decode(ctx_tgt, rb) != 0) {
                                    llama_batch_free(rb); ok = false;
                                }
                                llama_batch_free(rb);
                                t_snap_restore_us += ggml_time_us() - t0_rsnap;
                            }
                        }

                        n_past   = n_past_before_vb + 1 + n_accepted;
                        rejected = true;
                        break;
                    }
                }
                if (!ok) break;

                llama_set_hidden_capture_layers(ctx_tgt, nullptr, 0);

                if (!rejected) {
                    // All n_cands accepted: extract hidden states + bonus token
                    const int32_t n_ctx_new = 1 + n_cands;
                    {
                        int64_t t0_hex2 = ggml_time_us();
                        for (int32_t li = 0; li < n_tgt_lyrs && ok; li++) {
                            const float * h = llama_get_layer_hidden(ctx_tgt, tgt_layer_ids[li]);
                            if (!h) { ok = false; break; }
                            for (int32_t t = 0; t < n_ctx_new; t++)
                                memcpy(hidden_tmp_scratch.data() + (size_t)(t * n_tgt_lyrs + li) * n_embd,
                                       h + (size_t)t * n_embd, n_embd * sizeof(float));
                        }
                        t_hidden_extract_us += ggml_time_us() - t0_hex2;
                    }
                    if (ok) {
                        int64_t t0 = ggml_time_us();
                        bool proj_ok3 = proj_gpu
                            ? llama_dflash_proj_gpu_apply(proj_gpu, hidden_tmp_scratch.data(),
                                                          proj_scratch.data(), n_ctx_new)
                            : llama_model_dflash_project(model_dft, hidden_tmp_scratch.data(),
                                                         proj_scratch.data(), n_ctx_new);
                        t_project_us += ggml_time_us() - t0;
                        if (proj_ok3) {
                            int64_t t0_cp2 = ggml_time_us();
                            // Replace ctx_proj with new projections (for next draft decode)
                            ctx_proj.resize((size_t)n_ctx_new * n_embd);
                            memcpy(ctx_proj.data(), proj_scratch.data(), (size_t)n_ctx_new * n_embd * sizeof(float));
                            draft_kv_pos += n_pending;
                            n_pending = n_ctx_new;
                            t_ctxproj_us += ggml_time_us() - t0_cp2;
                        }
                    }
                    if (!ok) break;

                    if (n_gen < max_tokens) {
                        int64_t t0_smp = ggml_time_us();
                        tok = common_sampler_sample(smpl_tgt, ctx_tgt, n_cands);
                        common_sampler_accept(smpl_tgt, tok, true);
                        t_sampler_us += ggml_time_us() - t0_smp;
                        n_gen++;
                        n_past = n_past_before_vb + 1 + n_cands;
                        if (llama_vocab_is_eog(vocab_tgt, tok)) goto done;
                        if (!on_token(common_token_to_piece(ctx_tgt, tok, false))) goto done;
                    }
                }
                if (!ok) break;

                n_accept += n_accepted;
            }
        }

done:
        {
            if (t_decode_start > 0) t_decode_us = ggml_time_us() - t_decode_start;

            const int64_t t_total_us = ggml_time_us() - t_gen_start;
            const double t_prefill_s = t_prefill_us * 1e-6;
            const double t_decode_s  = t_decode_us  * 1e-6;
            const double t_total_s   = t_total_us   * 1e-6;

            // ── Prefill timing ──
            fprintf(stderr, "\n");
            fprintf(stderr, "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                    t_prefill_s * 1e3, n_prompt,
                    n_prompt > 0 ? t_prefill_s * 1e3 / n_prompt : 0.0,
                    t_prefill_s > 0.0 ? n_prompt / t_prefill_s : 0.0);

            // ── Decode timing breakdown ──
            if (no_draft) {
                fprintf(stderr, "  eval time     = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                        t_decode_s * 1e3, res.n_generated,
                        res.n_generated > 0 ? t_decode_s * 1e3 / res.n_generated : 0.0,
                        t_decode_s > 0.0 ? res.n_generated / t_decode_s : 0.0);
            } else if (n_iters > 0) {
                const double iters = (double)n_iters;
                const int64_t t_timed_us = t_draft_us + t_verify_us + t_lmhead_us
                                         + t_project_us + t_snap_save_us + t_snap_restore_us
                                         + t_hidden_extract_us
                                         + t_noise_build_us + t_batch_build_us + t_embd_collect_us
                                         + t_accept_us + t_ctxproj_us + t_kv_crop_us + t_sampler_us;
                const int64_t t_other_us = t_decode_us - t_timed_us;
                const int32_t n_gen = res.n_generated;

                fprintf(stderr, "  eval time     = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                        t_decode_s * 1e3, n_gen,
                        n_gen > 0 ? t_decode_s * 1e3 / n_gen : 0.0,
                        t_decode_s > 0.0 ? n_gen / t_decode_s : 0.0);

                fprintf(stderr, "  \n");
                fprintf(stderr, "  decode breakdown per iteration (%d iters, %.2f ms/iter):\n",
                        (int)n_iters, t_decode_us * 1e-3 / iters);
                fprintf(stderr, "    draft  fwd    = %8.2f ms/iter  (%5.1f%%)\n",
                        t_draft_us   * 1e-3 / iters, 100.0 * t_draft_us   / t_decode_us);
                fprintf(stderr, "    target verify = %8.2f ms/iter  (%5.1f%%)\n",
                        t_verify_us  * 1e-3 / iters, 100.0 * t_verify_us  / t_decode_us);
                fprintf(stderr, "    lm_head+LSE   = %8.2f ms/iter  (%5.1f%%)\n",
                        t_lmhead_us  * 1e-3 / iters, 100.0 * t_lmhead_us  / t_decode_us);
                fprintf(stderr, "    hidden proj   = %8.2f ms/iter  (%5.1f%%)\n",
                        t_project_us * 1e-3 / iters, 100.0 * t_project_us / t_decode_us);
                if (t_snap_save_us > 0) {
                    fprintf(stderr, "    snap save     = %8.2f ms/iter  (%5.1f%%)\n",
                            t_snap_save_us    * 1e-3 / iters, 100.0 * t_snap_save_us    / t_decode_us);
                }
                if (t_snap_restore_us > 0) {
                    fprintf(stderr, "    snap restore  = %8.2f ms/iter  (%5.1f%%)\n",
                            t_snap_restore_us * 1e-3 / iters, 100.0 * t_snap_restore_us / t_decode_us);
                }
                fprintf(stderr, "    hidden extr   = %8.2f ms/iter  (%5.1f%%)\n",
                        t_hidden_extract_us * 1e-3 / iters, 100.0 * t_hidden_extract_us / t_decode_us);
                fprintf(stderr, "    --- other breakdown ---\n");
                fprintf(stderr, "    noise build   = %8.2f ms/iter  (%5.1f%%)\n",
                        t_noise_build_us * 1e-3 / iters, 100.0 * t_noise_build_us / t_decode_us);
                fprintf(stderr, "    batch bld/free= %8.2f ms/iter  (%5.1f%%)\n",
                        t_batch_build_us * 1e-3 / iters, 100.0 * t_batch_build_us / t_decode_us);
                fprintf(stderr, "    embd collect  = %8.2f ms/iter  (%5.1f%%)\n",
                        t_embd_collect_us * 1e-3 / iters, 100.0 * t_embd_collect_us / t_decode_us);
                fprintf(stderr, "    accept argmax = %8.2f ms/iter  (%5.1f%%)\n",
                        t_accept_us * 1e-3 / iters, 100.0 * t_accept_us / t_decode_us);
                fprintf(stderr, "    ctx_proj mgmt = %8.2f ms/iter  (%5.1f%%)\n",
                        t_ctxproj_us * 1e-3 / iters, 100.0 * t_ctxproj_us / t_decode_us);
                fprintf(stderr, "    kv crop       = %8.2f ms/iter  (%5.1f%%)\n",
                        t_kv_crop_us * 1e-3 / iters, 100.0 * t_kv_crop_us / t_decode_us);
                fprintf(stderr, "    sampler       = %8.2f ms/iter  (%5.1f%%)\n",
                        t_sampler_us * 1e-3 / iters, 100.0 * t_sampler_us / t_decode_us);
                fprintf(stderr, "    residual      = %8.2f ms/iter  (%5.1f%%)\n",
                        t_other_us * 1e-3 / iters, 100.0 * t_other_us / t_decode_us);

                fprintf(stderr, "  \n");
                fprintf(stderr, "  draft acceptance rate = %.5f (%5d accepted / %5d generated)\n",
                        n_gen > 1 ? (float)res.n_accepted / (float)(n_gen - 1) : 0.0f,
                        res.n_accepted, n_gen - 1);
            }

            // ── Total timing ──
            fprintf(stderr, "  total time     = %10.2f ms / %5d tokens\n",
                    t_total_s * 1e3, n_prompt + res.n_generated);
        }

        // ── Rollback to prompt-only state for prefix cache ──
        // For hybrid models, partial seq_rm doesn't work. Clear all memory,
        // then restore the SSM snapshot. KV cache is lost but SSM state preserved.
        // The next request will re-prefill KV for matching prefix tokens.
        {
            auto * mem = llama_get_memory(ctx_tgt);
            bool rm_ok = llama_memory_seq_rm(mem, 0, n_prompt, -1);
            if (!rm_ok) {
                // Hybrid model: partial removal failed. Clear everything.
                llama_memory_seq_rm(mem, 0, -1, -1);
                if (ctx_dft) llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, -1, -1);
                // Restore SSM snapshot so we don't lose recurrent state
                if (use_recr_snap && cached_snap_valid) {
                    if (cached_gpu_snap) {
                        llama_gpu_snapshot_restore(ctx_tgt, cached_gpu_snap, 0,
                                LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    } else if (cached_recr_snap_written > 0) {
                        llama_state_seq_set_data_ext(ctx_tgt, cached_recr_snap.data(),
                                cached_recr_snap_written, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    }
                }
                // Mark that KV cache needs full re-prefill
                cached_prompt.clear();
            }
        }

        return res;
    }

    ~DFlashEngine() {
        if (gpu_snap) llama_gpu_snapshot_free(ctx_tgt, gpu_snap);
        if (smpl_tgt) common_sampler_free(smpl_tgt);
        llama_lm_head_gpu_free(lmh_gpu);
        llama_dflash_proj_gpu_free(proj_gpu);
    }
};

// ── Global state ─────────────────────────────────────────────────────────────

static DFlashEngine  g_engine;
static std::mutex    g_inference_mtx;   // serialise concurrent requests
static int32_t       g_default_n_predict = 512;

// ── SSE / JSON helpers ────────────────────────────────────────────────────────

static std::string sse(const json & data) {
    return "data: " + data.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
}

static void send_json(httplib::Response & res, int status, const json & body) {
    res.status = status;
    res.set_content(body.dump(-1, ' ', false, json::error_handler_t::replace), "application/json; charset=utf-8");
}

static void send_err(httplib::Response & res, int status, const std::string & msg) {
    send_json(res, status, {{"error", msg}});
}

// ── /health ───────────────────────────────────────────────────────────────────

static void handle_health(const httplib::Request &, httplib::Response & res) {
    send_json(res, 200, {{"status", "ok"}});
}

// ── /tokenize ─────────────────────────────────────────────────────────────────

static void handle_tokenize(const httplib::Request & req, httplib::Response & res) {
    json body;
    try { body = json::parse(req.body); } catch (...) { send_err(res, 400, "invalid JSON"); return; }

    if (!body.contains("content") || !body["content"].is_string()) {
        send_err(res, 400, "missing or invalid \"content\" field"); return;
    }

    const std::string content    = body["content"].get<std::string>();
    const bool add_special       = body.value("add_special", false);
    const bool parse_special     = body.value("parse_special", true);
    const bool with_pieces       = body.value("with_pieces", false);

    std::vector<llama_token> tokens =
        common_tokenize(g_engine.ctx_tgt, content, add_special, parse_special);

    json tokens_response = json::array();
    if (with_pieces) {
        for (const auto & tok : tokens) {
            std::string piece = common_token_to_piece(g_engine.ctx_tgt, tok);
            tokens_response.push_back({{"id", tok}, {"piece", piece}});
        }
    } else {
        tokens_response = tokens;
    }

    send_json(res, 200, {{"tokens", std::move(tokens_response)}});
}

// ── /detokenize ───────────────────────────────────────────────────────────────

static void handle_detokenize(const httplib::Request & req, httplib::Response & res) {
    json body;
    try { body = json::parse(req.body); } catch (...) { send_err(res, 400, "invalid JSON"); return; }

    if (!body.contains("tokens") || !body["tokens"].is_array()) {
        send_err(res, 400, "missing or invalid \"tokens\" field"); return;
    }

    std::string content;
    for (const auto & tok_j : body["tokens"]) {
        content += common_token_to_piece(g_engine.ctx_tgt, tok_j.get<llama_token>());
    }

    send_json(res, 200, {{"content", std::move(content)}});
}

// ── /completion ───────────────────────────────────────────────────────────────

static void handle_completion(const httplib::Request & req, httplib::Response & res) {
    json body;
    try { body = json::parse(req.body); }
    catch (...) { send_err(res, 400, "invalid JSON"); return; }

    const std::string prompt    = body.value("prompt", std::string{});
    int32_t           n_predict = body.value("n_predict", -1);
    if (n_predict < 0)  n_predict = body.value("max_tokens", g_default_n_predict);
    const bool        stream    = body.value("stream", false);

    if (prompt.empty()) { send_err(res, 400, "missing 'prompt'"); return; }

    std::vector<llama_token> tokens =
        common_tokenize(g_engine.ctx_tgt, prompt, /*add_special=*/true, /*parse_special=*/true);
    if (tokens.empty()) { send_err(res, 400, "tokenization produced empty result"); return; }

    const int32_t n_ctx = (int32_t)llama_n_ctx(g_engine.ctx_tgt);
    if ((int32_t)tokens.size() >= n_ctx) {
        send_err(res, 400, "prompt too long: " + std::to_string(tokens.size()) +
                 " tokens >= n_ctx " + std::to_string(n_ctx));
        return;
    }

    if (!stream) {
        std::string text;
        GenerateResult gres;
        {
            std::lock_guard<std::mutex> lock(g_inference_mtx);
            g_engine.reset();
            gres = g_engine.generate(tokens, n_predict, [&text](const std::string & s) {
                text += s; return true;
            });
        }
        if (gres.error) { send_err(res, 500, "generation error"); return; }

        send_json(res, 200, {
            {"content",          text},
            {"tokens_predicted", gres.n_generated},
            {"stop",             true},
            {"draft_n_accepted", gres.n_accepted},
            {"acceptance_rate",  gres.acceptance_rate()},
        });
        return;
    }

    // streaming
    auto session = std::make_shared<TokenStream>();

    std::thread([session, tokens, n_predict]() {
        std::lock_guard<std::mutex> lock(g_inference_mtx);
        g_engine.reset();
        auto gres = g_engine.generate(tokens, n_predict, [&session](const std::string & s) {
            session->push(s); return true;
        });
        session->finish(gres);
    }).detach();

    res.set_chunked_content_provider("text/event-stream",
        [session](size_t, httplib::DataSink & sink) -> bool {
            auto [chunk_data, has_more] = session->pop();
            if (has_more) {
                std::string chunk = sse({
                    {"choices", json::array({{
                        {"text", chunk_data.text},
                        {"finish_reason", nullptr},
                    }})},
                });
                sink.write(chunk.data(), chunk.size());
                return true;
            }
            const auto & r = session->result;
            std::string last = sse({
                {"choices", json::array({{
                    {"text", ""},
                    {"finish_reason", r.n_generated >= 0 ? "stop" : "length"},
                }})},
            });
            last += "data: [DONE]\n\n";
            sink.write(last.data(), last.size());
            sink.done();
            return false;
        }
    );
}

// ── /v1/chat/completions ──────────────────────────────────────────────────────

// Fallback parser for Qwen-style XML tool calls:
//   <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>\n</tool_call>
// Separates thinking (<think>...</think>) from content and extracts tool calls.
static common_chat_msg parse_qwen_xml_tool_calls(const std::string & text) {
    common_chat_msg msg;
    msg.role = "assistant";

    // Extract thinking
    std::string remaining = text;
    {
        auto think_start = remaining.find("<think>");
        auto think_end   = remaining.find("</think>");
        if (think_start != std::string::npos && think_end != std::string::npos) {
            size_t content_start = think_start + 7; // strlen("<think>")
            msg.reasoning_content = remaining.substr(content_start, think_end - content_start);
            // Remove think block from remaining
            remaining = remaining.substr(0, think_start) + remaining.substr(think_end + 8); // strlen("</think>")
        }
    }

    // Extract tool calls
    size_t search_pos = 0;
    while (true) {
        auto tc_start = remaining.find("<tool_call>", search_pos);
        if (tc_start == std::string::npos) break;
        auto tc_end = remaining.find("</tool_call>", tc_start);
        if (tc_end == std::string::npos) break;

        std::string tc_body = remaining.substr(tc_start + 11, tc_end - tc_start - 11); // strlen("<tool_call>")

        // Extract function name: <function=NAME>
        common_chat_tool_call tool_call;
        auto fn_start = tc_body.find("<function=");
        if (fn_start != std::string::npos) {
            auto fn_name_start = fn_start + 10; // strlen("<function=")
            auto fn_name_end = tc_body.find(">", fn_name_start);
            if (fn_name_end != std::string::npos) {
                tool_call.name = tc_body.substr(fn_name_start, fn_name_end - fn_name_start);
            }

            // Extract parameters: <parameter=KEY>\nVALUE\n</parameter>
            json args = json::object();
            size_t param_pos = fn_name_end + 1;
            while (true) {
                auto p_start = tc_body.find("<parameter=", param_pos);
                if (p_start == std::string::npos) break;
                auto key_start = p_start + 11; // strlen("<parameter=")
                auto key_end = tc_body.find(">", key_start);
                if (key_end == std::string::npos) break;
                std::string key = tc_body.substr(key_start, key_end - key_start);

                auto val_start = key_end + 1;
                auto p_end = tc_body.find("</parameter>", val_start);
                if (p_end == std::string::npos) break;
                std::string val = tc_body.substr(val_start, p_end - val_start);

                // Trim whitespace
                while (!val.empty() && (val.front() == '\n' || val.front() == ' ')) val.erase(val.begin());
                while (!val.empty() && (val.back() == '\n' || val.back() == ' '))  val.pop_back();

                args[key] = val;
                param_pos = p_end + 12; // strlen("</parameter>")
            }
            tool_call.arguments = args.dump();
        }

        if (!tool_call.name.empty()) {
            msg.tool_calls.push_back(std::move(tool_call));
        }

        // Remove tool_call block from remaining content
        remaining = remaining.substr(0, tc_start) + remaining.substr(tc_end + 12); // strlen("</tool_call>")
        search_pos = tc_start;
    }

    // Remaining text is content (trim whitespace)
    while (!remaining.empty() && (remaining.front() == '\n' || remaining.front() == ' ')) remaining.erase(remaining.begin());
    while (!remaining.empty() && (remaining.back() == '\n' || remaining.back() == ' '))  remaining.pop_back();
    msg.content = remaining;

    return msg;
}

static void handle_chat_completions(const httplib::Request & req, httplib::Response & res) {
    json body;
    try { body = json::parse(req.body); }
    catch (...) { send_err(res, 400, "invalid JSON"); return; }

    const auto messages_j = body.find("messages");
    if (messages_j == body.end() || !messages_j->is_array()) {
        send_err(res, 400, "missing 'messages' array"); return;
    }

    const int32_t     max_tokens  = body.value("max_tokens", g_default_n_predict);
    const bool        stream      = body.value("stream", false);
    const std::string model_name  = body.value("model", "dflash");

    // Parse tools
    auto tools_j     = body.value("tools", json());
    auto has_tools   = tools_j.is_array() && !tools_j.empty();
    auto tool_choice = body.value("tool_choice", std::string("auto"));

    // Parse stop sequences
    std::vector<std::string> stop_words;
    if (body.contains("stop")) {
        const auto & stop_j = body["stop"];
        if (stop_j.is_string()) {
            stop_words.push_back(stop_j.get<std::string>());
        } else if (stop_j.is_array()) {
            for (const auto & s : stop_j) {
                if (s.is_string()) stop_words.push_back(s.get<std::string>());
            }
        }
    }

    // JSON messages → common_chat_msg
    std::vector<common_chat_msg> msgs;
    try {
        msgs = common_chat_msgs_parse_oaicompat(*messages_j);
    } catch (const std::exception & e) {
        send_err(res, 400, e.what()); return;
    }

    // Apply chat template with tools + thinking support
    std::string prompt;
    common_chat_parser_params parser_params;
    std::string grammar_str;
    bool grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_triggers;
    std::vector<std::string> preserved_tokens_str;
    std::string generation_prompt_str;
    {
        auto tmpls = common_chat_templates_init(g_engine.model_tgt, "");
        common_chat_templates_inputs inputs;
        inputs.messages              = msgs;
        inputs.add_generation_prompt = true;
        inputs.use_jinja             = true;

        if (has_tools) {
            inputs.tools       = common_chat_tools_parse_oaicompat(tools_j);
            inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(tool_choice);
        }

        // Parse reasoning_format
        if (body.contains("reasoning_format")) {
            inputs.reasoning_format = common_reasoning_format_from_name(body["reasoning_format"].get<std::string>());
        }

        // Parse enable_thinking from chat_template_kwargs
        if (body.contains("chat_template_kwargs")) {
            auto kwargs = body["chat_template_kwargs"];
            for (const auto & item : kwargs.items()) {
                inputs.chat_template_kwargs[item.key()] = item.value().dump();
            }
            auto et = kwargs.value("enable_thinking", json());
            if (et.is_boolean()) {
                inputs.enable_thinking = et.get<bool>();
            }
        }

        auto chat_params = common_chat_templates_apply(tmpls.get(), inputs);
        prompt = chat_params.prompt;
        parser_params = common_chat_parser_params(chat_params);
        parser_params.reasoning_format = inputs.reasoning_format;
        parser_params.parse_tool_calls = has_tools;
        if (!chat_params.parser.empty()) {
            parser_params.parser.load(chat_params.parser);
        }

        // Save grammar info for sampler
        grammar_str = chat_params.grammar;
        grammar_lazy = chat_params.grammar_lazy;
        grammar_triggers = chat_params.grammar_triggers;
        preserved_tokens_str = chat_params.preserved_tokens;
        generation_prompt_str = chat_params.generation_prompt;

        // Merge additional stop sequences from template
        for (const auto & s : chat_params.additional_stops) {
            stop_words.push_back(s);
        }
    }

    std::vector<llama_token> tokens =
        common_tokenize(g_engine.ctx_tgt, prompt, /*add_special=*/true, /*parse_special=*/true);
    if (tokens.empty()) { send_err(res, 400, "tokenization produced empty result"); return; }

    const int32_t n_ctx = (int32_t)llama_n_ctx(g_engine.ctx_tgt);
    if ((int32_t)tokens.size() >= n_ctx) {
        send_err(res, 400, "prompt too long: " + std::to_string(tokens.size()) +
                 " tokens >= n_ctx " + std::to_string(n_ctx));
        return;
    }

    const std::string cmpl_id     = gen_completion_id();
    const int64_t     created     = now_unix();
    const int32_t     n_prompt_toks = (int32_t)tokens.size();

    // Per-request sampling parameters
    auto req_sparams = g_engine.default_sparams;
    if (body.contains("temperature"))       req_sparams.temp            = body["temperature"].get<float>();
    if (body.contains("top_p"))             req_sparams.top_p           = body["top_p"].get<float>();
    if (body.contains("top_k"))             req_sparams.top_k           = body["top_k"].get<int32_t>();
    if (body.contains("min_p"))             req_sparams.min_p           = body["min_p"].get<float>();
    if (body.contains("seed"))              req_sparams.seed            = body["seed"].get<uint32_t>();
    if (body.contains("repeat_penalty"))    req_sparams.penalty_repeat  = body["repeat_penalty"].get<float>();
    if (body.contains("frequency_penalty")) req_sparams.penalty_freq    = body["frequency_penalty"].get<float>();
    if (body.contains("presence_penalty"))  req_sparams.penalty_present = body["presence_penalty"].get<float>();

    // Setup per-request sampler (sampling params + optional grammar)
    auto setup_request_sampler = [&]() {
        auto sparams = req_sparams;

        if (!grammar_str.empty()) {
            sparams.grammar = common_grammar(COMMON_GRAMMAR_TYPE_TOOL_CALLS, grammar_str);
            sparams.grammar_lazy = grammar_lazy;
            sparams.generation_prompt = generation_prompt_str;

            const auto * vocab = llama_model_get_vocab(g_engine.model_tgt);
            for (const auto & s : preserved_tokens_str) {
                auto toks = common_tokenize(vocab, s, false, true);
                for (auto tok : toks) sparams.preserved_tokens.insert(tok);
            }
            for (const auto & t : grammar_triggers) {
                if (t.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                    auto ids = common_tokenize(vocab, t.value, false, true);
                    if (ids.size() == 1) {
                        common_grammar_trigger trigger;
                        trigger.type  = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                        trigger.value = t.value;
                        trigger.token = ids[0];
                        sparams.grammar_triggers.push_back(std::move(trigger));
                    } else {
                        sparams.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, t.value});
                    }
                } else {
                    sparams.grammar_triggers.push_back(t);
                }
            }
        }

        if (g_engine.smpl_tgt) common_sampler_free(g_engine.smpl_tgt);
        g_engine.smpl_tgt = common_sampler_init(g_engine.model_tgt, sparams);
    };

    auto restore_default_sampler = [&]() {
        if (g_engine.smpl_tgt) common_sampler_free(g_engine.smpl_tgt);
        g_engine.smpl_tgt = common_sampler_init(g_engine.model_tgt, g_engine.default_sparams);
    };

    if (!stream) {
        std::string text;
        bool stopped_by_word = false;
        GenerateResult gres;
        {
            std::lock_guard<std::mutex> lock(g_inference_mtx);
            g_engine.reset();
            setup_request_sampler();
            gres = g_engine.generate(tokens, max_tokens, [&text, &stop_words, &stopped_by_word](const std::string & s) {
                text += s;
                for (const auto & sw : stop_words) {
                    if (!sw.empty() && text.size() >= sw.size() &&
                        text.compare(text.size() - sw.size(), sw.size(), sw) == 0) {
                        text.resize(text.size() - sw.size());
                        stopped_by_word = true;
                        return false;
                    }
                }
                return true;
            });
            restore_default_sampler();
        }
        if (gres.error) { send_err(res, 500, "generation error"); return; }

        // Parse output for tools + thinking
        common_chat_msg msg;
        try {
            msg = common_chat_parse(text, false, parser_params);
        } catch (const std::exception & e) {
            LOG_WRN("chat parse failed, trying fallback: %s\n", e.what());
            msg = parse_qwen_xml_tool_calls(text);
        }
        if (msg.empty()) {
            msg = parse_qwen_xml_tool_calls(text);
        }
        if (msg.empty()) {
            msg.role    = "assistant";
            msg.content = text;
        }

        std::string finish_reason = "length";
        bool stopped_eos = !stopped_by_word && gres.n_generated < max_tokens;
        if (stopped_by_word || stopped_eos) {
            finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
        }

        send_json(res, 200, {
            {"id",      cmpl_id},
            {"object",  "chat.completion"},
            {"created", created},
            {"model",   model_name},
            {"choices", json::array({{
                {"index",         0},
                {"message",       msg.to_json_oaicompat()},
                {"finish_reason", finish_reason},
            }})},
            {"usage", {
                {"prompt_tokens",     n_prompt_toks},
                {"completion_tokens", gres.n_generated},
                {"total_tokens",      n_prompt_toks + gres.n_generated},
                {"draft_n_accepted",  gres.n_accepted},
                {"acceptance_rate",   gres.acceptance_rate()},
            }},
        });
        return;
    }

    // ── streaming ──
    auto session = std::make_shared<TokenStream>();

    // Capture stop_words and parser_params for the streaming thread
    auto stop_words_shared = std::make_shared<std::vector<std::string>>(std::move(stop_words));
    auto parser_shared     = std::make_shared<common_chat_parser_params>(parser_params);

    // Capture per-request sampler params by value for the streaming thread
    auto req_sparams_copy          = req_sparams;
    auto grammar_str_copy          = grammar_str;
    auto grammar_lazy_copy         = grammar_lazy;
    auto grammar_triggers_copy     = grammar_triggers;
    auto preserved_tokens_str_copy = preserved_tokens_str;
    auto generation_prompt_copy    = generation_prompt_str;

    std::thread([session, tokens, max_tokens, stop_words_shared,
                  req_sparams_copy, grammar_str_copy, grammar_lazy_copy,
                  grammar_triggers_copy, preserved_tokens_str_copy,
                  generation_prompt_copy]() {
        std::lock_guard<std::mutex> lock(g_inference_mtx);
        g_engine.reset();

        // Setup per-request sampler
        {
            auto sparams = req_sparams_copy;
            if (!grammar_str_copy.empty()) {
                sparams.grammar = common_grammar(COMMON_GRAMMAR_TYPE_TOOL_CALLS, grammar_str_copy);
                sparams.grammar_lazy = grammar_lazy_copy;
                sparams.generation_prompt = generation_prompt_copy;
                const auto * vocab = llama_model_get_vocab(g_engine.model_tgt);
                for (const auto & s : preserved_tokens_str_copy) {
                    auto toks = common_tokenize(vocab, s, false, true);
                    for (auto tok : toks) sparams.preserved_tokens.insert(tok);
                }
                for (const auto & t : grammar_triggers_copy) {
                    if (t.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                        auto ids = common_tokenize(vocab, t.value, false, true);
                        if (ids.size() == 1) {
                            common_grammar_trigger trigger;
                            trigger.type  = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                            trigger.value = t.value;
                            trigger.token = ids[0];
                            sparams.grammar_triggers.push_back(std::move(trigger));
                        } else {
                            sparams.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, t.value});
                        }
                    } else {
                        sparams.grammar_triggers.push_back(t);
                    }
                }
            }
            if (g_engine.smpl_tgt) common_sampler_free(g_engine.smpl_tgt);
            g_engine.smpl_tgt = common_sampler_init(g_engine.model_tgt, sparams);
        }

        // State machine: NORMAL → IN_THINK → AFTER_THINK → IN_TOOL_CALL
        enum { ST_NORMAL, ST_IN_THINK, ST_AFTER_THINK, ST_IN_TOOL_CALL } state = ST_NORMAL;
        std::string accum;
        size_t sent_pos = 0;
        bool stopped_word = false;
        size_t max_holdback = 11; // <tool_call> length
        for (const auto & sw : *stop_words_shared) {
            max_holdback = std::max(max_holdback, sw.size());
        }

        auto flush_region = [&](size_t from, size_t to, const std::string & type) {
            if (to <= from) return;
            std::string text = accum.substr(from, to - from);
            if (!text.empty()) session->push(text, type);
        };

        auto gres = g_engine.generate(tokens, max_tokens,
            [&](const std::string & s) {
                accum += s;

                // Full stop word match
                for (const auto & sw : *stop_words_shared) {
                    if (!sw.empty() && accum.size() >= sw.size() &&
                        accum.compare(accum.size() - sw.size(), sw.size(), sw) == 0) {
                        accum.resize(accum.size() - sw.size());
                        stopped_word = true;
                        return false;
                    }
                }

                // Partial stop word match - expand holdback
                size_t holdback = max_holdback;
                for (const auto & sw : *stop_words_shared) {
                    size_t partial = string_find_partial_stop(accum, sw);
                    if (partial != std::string::npos) {
                        holdback = std::max(holdback, accum.size() - partial);
                    }
                }

                if (state == ST_IN_TOOL_CALL) return true;

                size_t search_from = sent_pos > holdback ? sent_pos - holdback : 0;

                if (state == ST_NORMAL) {
                    auto think_pos = accum.find("<think>", search_from);
                    if (think_pos != std::string::npos && think_pos >= sent_pos) {
                        flush_region(sent_pos, think_pos, "content");
                        sent_pos = think_pos + 7;
                        state = ST_IN_THINK;
                    }
                }

                if (state == ST_IN_THINK) {
                    auto end_pos = accum.find("</think>", search_from);
                    if (end_pos != std::string::npos && end_pos >= sent_pos) {
                        flush_region(sent_pos, end_pos, "reasoning_content");
                        sent_pos = end_pos + 8;
                        state = ST_AFTER_THINK;
                    }
                }

                if (state == ST_AFTER_THINK || state == ST_NORMAL) {
                    auto tc_pos = accum.find("<tool_call>", search_from);
                    if (tc_pos != std::string::npos && tc_pos >= sent_pos) {
                        flush_region(sent_pos, tc_pos, "content");
                        sent_pos = accum.size();
                        state = ST_IN_TOOL_CALL;
                        return true;
                    }
                }

                // Flush safe region (holding back for tag detection / partial stop)
                size_t safe_pos = accum.size() > holdback ? accum.size() - holdback : 0;
                while (safe_pos > sent_pos && (accum[safe_pos] & 0xC0) == 0x80) safe_pos--;
                std::string type = (state == ST_IN_THINK) ? "reasoning_content" : "content";
                flush_region(sent_pos, safe_pos, type);
                if (safe_pos > sent_pos) sent_pos = safe_pos;

                return true;
            });
        session->finish(gres, accum, stopped_word);
        // Restore default sampler
        if (g_engine.smpl_tgt) common_sampler_free(g_engine.smpl_tgt);
        g_engine.smpl_tgt = common_sampler_init(g_engine.model_tgt, g_engine.default_sparams);
    }).detach();

    auto make_chunk = [cmpl_id, created, model_name](const json & delta, const char * finish_reason) {
        return json{
            {"id",      cmpl_id},
            {"object",  "chat.completion.chunk"},
            {"created", created},
            {"model",   model_name},
            {"choices", json::array({{
                {"index",         0},
                {"delta",         delta},
                {"finish_reason", finish_reason ? json(finish_reason) : json(nullptr)},
            }})},
        };
    };

    // Role chunk
    std::string role_chunk = sse(make_chunk({{"role", "assistant"}, {"content", ""}}, nullptr));

    res.set_chunked_content_provider("text/event-stream",
        [session, make_chunk, n_prompt_toks, parser_shared, max_tokens,
         role_chunk = std::move(role_chunk), role_sent = false](
             size_t, httplib::DataSink & sink) mutable -> bool {
            if (!role_sent) {
                sink.write(role_chunk.data(), role_chunk.size());
                role_sent = true;
                return true;
            }
            auto [chunk_data, has_more] = session->pop();
            if (has_more) {
                json delta;
                delta[chunk_data.type] = chunk_data.text;
                std::string chunk = sse(make_chunk(delta, nullptr));
                sink.write(chunk.data(), chunk.size());
                return true;
            }
            // Final: parse the full text for tool calls
            const auto & r = session->result;
            const auto & full_text = session->full_text;
            common_chat_msg msg;
            try {
                msg = common_chat_parse(full_text, false, *parser_shared);
            } catch (const std::exception & e) {
                LOG_WRN("chat parse failed, trying fallback: %s\n", e.what());
                msg = parse_qwen_xml_tool_calls(full_text);
            }
            if (msg.empty()) {
                msg = parse_qwen_xml_tool_calls(full_text);
            }

            std::string finish_reason = "length";
            if (session->stopped_by_word || r.n_generated < max_tokens) {
                finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
            }

            // Send tool_calls in a delta chunk if present
            if (!msg.tool_calls.empty()) {
                json tc_array = json::array();
                for (size_t i = 0; i < msg.tool_calls.size(); i++) {
                    tc_array.push_back({
                        {"index", (int)i},
                        {"id", msg.tool_calls[i].id.empty() ? "call_" + std::to_string(i) : msg.tool_calls[i].id},
                        {"type", "function"},
                        {"function", {
                            {"name", msg.tool_calls[i].name},
                            {"arguments", msg.tool_calls[i].arguments},
                        }},
                    });
                }
                std::string tc_chunk = sse(make_chunk({{"tool_calls", tc_array}}, nullptr));
                sink.write(tc_chunk.data(), tc_chunk.size());
            }

            json stop_chunk = make_chunk(json::object(), finish_reason.c_str());
            stop_chunk["usage"] = {
                {"prompt_tokens",     n_prompt_toks},
                {"completion_tokens", r.n_generated},
                {"total_tokens",      n_prompt_toks + r.n_generated},
                {"draft_n_accepted",  r.n_accepted},
                {"acceptance_rate",   r.acceptance_rate()},
            };
            std::string last = sse(stop_chunk);
            last += "data: [DONE]\n\n";
            sink.write(last.data(), last.size());
            sink.done();
            return false;
        }
    );
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    // Pre-extract server-specific args before common_params_parse
    int32_t     arg_draft_ctx_max = 0;
    std::string arg_host          = "127.0.0.1";
    int         arg_port          = 8080;

    for (int i = 1; i < argc; ) {
        auto match = [&](const char * flag, bool has_val) -> bool {
            if (strcmp(argv[i], flag) != 0) return false;
            if (has_val && i + 1 >= argc) {
                fprintf(stderr, "error: %s requires a value\n", flag);
                exit(1);
            }
            return true;
        };
        if (match("--host", true)) {
            arg_host = argv[i + 1];
            for (int j = i; j < argc - 2; j++) argv[j] = argv[j + 2];
            argc -= 2;
        } else if (match("--port", true)) {
            arg_port = atoi(argv[i + 1]);
            for (int j = i; j < argc - 2; j++) argv[j] = argv[j + 2];
            argc -= 2;
        } else if (match("--draft-ctx-max", true)) {
            arg_draft_ctx_max = std::max(0, atoi(argv[i + 1]));
            for (int j = i; j < argc - 2; j++) argv[j] = argv[j + 2];
            argc -= 2;
        } else {
            i++;
        }
    }

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    if (params.n_predict > 0)
        g_default_n_predict = params.n_predict;

    if (!g_engine.init(params, arg_draft_ctx_max)) {
        llama_backend_free();
        return 1;
    }

    httplib::Server srv;
    srv.set_default_headers({{"Access-Control-Allow-Origin", "*"}});

    srv.Get ("/health",              handle_health);
    srv.Post("/tokenize",            handle_tokenize);
    srv.Post("/detokenize",          handle_detokenize);
    srv.Post("/completion",          handle_completion);
    srv.Post("/v1/completions",      handle_completion);
    srv.Post("/v1/chat/completions", handle_chat_completions);

    fprintf(stderr, "\nllama-dflash-server listening on http://%s:%d\n\n",
            arg_host.c_str(), arg_port);

    if (!srv.listen(arg_host.c_str(), arg_port)) {
        fprintf(stderr, "Failed to start HTTP server on %s:%d\n",
                arg_host.c_str(), arg_port);
        llama_backend_free();
        return 1;
    }

    llama_backend_free();
    return 0;
}
