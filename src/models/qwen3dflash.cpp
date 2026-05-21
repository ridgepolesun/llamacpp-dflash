#include "models.h"
#include "llama-impl.h"
#include "../llama-kv-cache.h"

// DFlash Draft Model Graph Builder — with incremental KV cache
// =============================================================
// Batch layout: [new_ctx_0..new_ctx_{n_new-1} | noise_0..noise_{n_noise-1}]
//   n_new  = newly accepted token projections (variable, 0 on first non-prefill call)
//   n_noise = block_size (typically 16)
//
// Per-layer attention (matches Python DFlashDraftModel):
//   Q from noise only (after layernorm + q_norm + RoPE with noise positions)
//   K/V from new_ctx (raw, no layernorm) + noise (normed)
//   K: concat → k_norm → RoPE → store to KV cache
//   V: concat → store to KV cache
//   Attention: Q[n_noise] × K[n_kv_cached] / V[n_kv_cached]  (bidirectional, no mask)
//
// KV cache stores all historical context K/V + current noise K/V.
// After each iteration, the server crops noise entries via seq_rm.

llm_build_qwen3dflash::llm_build_qwen3dflash(
    const llama_model & model,
    const llm_graph_params & params) : llm_graph_context(params)
{
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    const int64_t n_noise = (n_tokens >= (int64_t)hparams.n_dflash_block_size)
        ? (int64_t)hparams.n_dflash_block_size
        : n_tokens;
    const int64_t n_new = n_tokens - n_noise;
    GGML_ASSERT(n_new >= 0);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // ── embedding input ────────────────────────────────────────────────────
    {
        auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd);
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        cb(inp->tokens, "inp_tokens", -1);
        ggml_set_input(inp->tokens);
        ggml_build_forward_expand(gf, inp->tokens);

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
        cb(inp->embd, "inp_embd", -1);
        ggml_set_input(inp->embd);

        inpL = inp->embd;
        res->t_inp_tokens = inp->tokens;
        res->add_input(std::move(inp));
    }

    // ── split new_ctx / noise ─────────────────────────────────────────────
    ggml_tensor * new_ctx = (n_new > 0)
        ? ggml_view_2d(ctx0, inpL, hparams.n_embd, n_new, inpL->nb[1], 0)
        : nullptr;
    ggml_tensor * noise_embd = ggml_view_2d(ctx0, inpL, hparams.n_embd, n_noise, inpL->nb[1],
                                             (size_t)n_new * inpL->nb[1]);
    if (new_ctx) { cb(new_ctx, "new_ctx", -1); }
    cb(noise_embd, "noise_embd", -1);

    ggml_tensor * inpNoise = noise_embd;

    // ── positional inputs ──────────────────────────────────────────────────
    ggml_tensor * inp_pos   = build_inp_pos();
    ggml_tensor * pos_noise = ggml_view_1d(ctx0, inp_pos, n_noise,
                                            (size_t)n_new * sizeof(int32_t));
    ggml_tensor * pos_all   = inp_pos;  // length n_tokens = n_new + n_noise

    // ── KV cache input ─────────────────────────────────────────────────────
    // Manually build k_idxs/v_idxs for KV cache storage without creating the
    // large kq_mask tensor (bidirectional attention uses mask=nullptr).
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);
    GGML_ASSERT(mctx_cur != nullptr);

    ggml_tensor * k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
    ggml_tensor * v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);
    ggml_set_input(k_idxs);
    ggml_set_input(v_idxs);

    // Register as graph input so set_input() is called during execution
    struct kv_idxs_input : public llm_graph_input_i {
        const llama_kv_cache_context * mctx;
        ggml_tensor * ki;
        ggml_tensor * vi;
        kv_idxs_input(const llama_kv_cache_context * m, ggml_tensor * k, ggml_tensor * v)
            : mctx(m), ki(k), vi(v) {}
        void set_input(const llama_ubatch * ubatch) override {
            mctx->set_input_k_idxs(ki, ubatch);
            mctx->set_input_v_idxs(vi, ubatch);
        }
        bool can_reuse(const llm_graph_params &) override { return false; }
    };
    res->add_input(std::make_unique<kv_idxs_input>(mctx_cur, k_idxs, v_idxs));

    // ── decoder layers ────────────────────────────────────────────────────
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpNoise;

        // Pre-attention RMSNorm on noise tokens only
        cur = build_norm(inpNoise, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // ── Attention ─────────────────────────────────────────────────────
        {
            // Q from noise → [head_dim, n_head, n_noise]
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_noise);
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            Qcur = ggml_rope_ext(ctx0, Qcur, pos_noise, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur_rope", il);

            // K from noise: [head_dim, n_head_kv, n_noise]
            ggml_tensor * Kn = build_lora_mm(model.layers[il].wk, cur);
            Kn = ggml_reshape_3d(ctx0, Kn, n_embd_head, n_head_kv, n_noise);

            // K/V from new_ctx (no layernorm) + noise
            ggml_tensor * K_new, * V_new;
            if (new_ctx && n_new > 0) {
                ggml_tensor * Kc = build_lora_mm(model.layers[il].wk, new_ctx);
                ggml_tensor * Vc = build_lora_mm(model.layers[il].wv, new_ctx);
                Kc = ggml_reshape_3d(ctx0, Kc, n_embd_head, n_head_kv, n_new);
                Vc = ggml_reshape_3d(ctx0, Vc, n_embd_head, n_head_kv, n_new);

                ggml_tensor * Vn = build_lora_mm(model.layers[il].wv, cur);
                Vn = ggml_reshape_3d(ctx0, Vn, n_embd_head, n_head_kv, n_noise);

                // K = [Kc | Kn] → k_norm → RoPE (all positions)
                K_new = ggml_cont(ctx0, ggml_concat(ctx0, Kc, Kn, 2));
                K_new = ggml_reshape_3d(ctx0, K_new, n_embd_head, n_head_kv, n_new + n_noise);
                K_new = build_norm(K_new, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
                K_new = ggml_rope_ext(ctx0, K_new, pos_all, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(K_new, "K_new_rope", il);

                // V = [Vc | Vn]
                V_new = ggml_cont(ctx0, ggml_concat(ctx0, Vc, Vn, 2));
            } else {
                // No new ctx: K/V from noise only
                Kn = build_norm(Kn, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
                Kn = ggml_rope_ext(ctx0, Kn, pos_noise, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);

                ggml_tensor * Vn = build_lora_mm(model.layers[il].wv, cur);
                Vn = ggml_reshape_3d(ctx0, Vn, n_embd_head, n_head_kv, n_noise);
                K_new = Kn;
                V_new = Vn;
            }

            // ── KV cache: store new K/V, read full history ────────────────
            ggml_build_forward_expand(gf, Qcur);
            ggml_build_forward_expand(gf, V_new);
            ggml_build_forward_expand(gf, K_new);

            ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, K_new, k_idxs, il));
            ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, V_new, v_idxs, il));

            // Read full K/V from cache (all historical + current)
            // Layout: [head_dim, n_head_kv, n_kv, n_stream]
            ggml_tensor * K_full = mctx_cur->get_k(ctx0, il);
            ggml_tensor * V_full = mctx_cur->get_v(ctx0, il);

            // Bidirectional attention via build_attn_mha (handles GQA + permutes correctly)
            // Q: [head_dim, n_head, n_noise], K/V: [head_dim, n_head_kv, n_kv]
            // Pass nullptr mask for bidirectional (no causal masking)
            ggml_tensor * kqv = build_attn_mha(Qcur, K_full, V_full,
                    /*kq_b=*/nullptr, /*kq_mask=*/nullptr, /*sinks=*/nullptr, /*v_mla=*/nullptr,
                    /*scale=*/1.0f / sqrtf((float)n_embd_head), il);
            cb(kqv, "kqv_out", il);

            // build_attn_mha returns [n_embd, n_noise]
            cur = build_lora_mm(model.layers[il].wo, kqv);
            cb(cur, "attn_out", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        inpNoise = ggml_add(ctx0, cur, ffn_inp);
        cb(inpNoise, "l_out", il);
    }
    cur = inpNoise;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    // Extend t_embd to cover all n_tokens (new_ctx rows pass-through, noise rows = output)
    if (n_new > 0) {
        ggml_tensor * ctx_pass = ggml_view_2d(ctx0, inpL, n_embd, n_new, inpL->nb[1], 0);
        cur = ggml_cont(ctx0, ggml_concat(ctx0, ctx_pass, cur, 1));
    }
    res->t_embd   = cur;
    res->t_logits = nullptr;

    ggml_build_forward_expand(gf, cur);
}
