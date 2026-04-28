#include "models.h"
#include "llama-impl.h"
#include "../llama-kv-cache.h"

// DFlash Draft Model Graph Builder
// =================================
// Batch layout: [ctx_0..ctx_{n_ctx-1} | noise_0..noise_{n_noise-1}]
//
// Per-layer attention (matches Python DFlashDraftModel exactly):
//   - Q/K/V from noise (after input_layernorm)
//   - K/V from ctx (no input_layernorm, matches Python where target_hidden bypasses layernorm)
//   - K: concat([Kc, Kn]) → k_norm → RoPE (entire K uses full position_ids)
//   - Q: q_norm → RoPE (Q uses noise positions only, matching apply_rotary_pos_emb q_len slice)
//   - V: concat([Vc, Vn]) (no norm)
//   - Bidirectional flash-attn (no mask): Q[n_noise] × K[n_ctx+n_noise] / V[n_ctx+n_noise]
//
// No KV cache: each call recomputes full attention from scratch.
// (Matching Python use_cache=False behavior for correctness verification)

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
    const int64_t n_ctx = n_tokens - n_noise;
    GGML_ASSERT(n_ctx >= 0);

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

    // ── split ctx / noise ─────────────────────────────────────────────────
    ggml_tensor * target_ctx = (n_ctx > 0)
        ? ggml_view_2d(ctx0, inpL, hparams.n_embd, n_ctx, inpL->nb[1], 0)
        : nullptr;
    ggml_tensor * noise_embd = ggml_view_2d(ctx0, inpL, hparams.n_embd, n_noise, inpL->nb[1],
                                             (size_t)n_ctx * inpL->nb[1]);
    if (target_ctx) { cb(target_ctx, "target_ctx", -1); }
    cb(noise_embd, "noise_embd", -1);

    ggml_tensor * inpNoise = noise_embd;

    // ── positional inputs ──────────────────────────────────────────────────
    // inp_pos[0..n_ctx-1]        = ctx token positions
    // inp_pos[n_ctx..n_tokens-1] = noise token positions
    ggml_tensor * inp_pos   = build_inp_pos();
    ggml_tensor * pos_ctx   = (n_ctx > 0)
        ? ggml_view_1d(ctx0, inp_pos, n_ctx,   0)
        : nullptr;
    ggml_tensor * pos_noise = ggml_view_1d(ctx0, inp_pos, n_noise,
                                            (size_t)n_ctx * sizeof(int32_t));
    // full position_ids for K (ctx + noise combined), matching Python's
    // apply_rotary_pos_emb which applies RoPE to the full K vector.
    ggml_tensor * pos_all   = inp_pos;  // length n_tokens = n_ctx + n_noise

    // ── decoder layers ────────────────────────────────────────────────────
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpNoise;

        // Pre-attention RMSNorm on noise tokens only (matches input_layernorm in Python)
        cur = build_norm(inpNoise, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // ── Attention ─────────────────────────────────────────────────────
        {
            // Q from noise → [head_dim, n_head, n_noise]
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_noise);
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            // Q RoPE: use noise positions only (Python: apply_rotary_pos_emb uses q_len slice)
            Qcur = ggml_rope_ext(ctx0, Qcur, pos_noise, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur_rope", il);

            // K from noise: [head_dim, n_head_kv, n_noise]
            ggml_tensor * Kn = build_lora_mm(model.layers[il].wk, cur);
            Kn = ggml_reshape_3d(ctx0, Kn, n_embd_head, n_head_kv, n_noise);

            // K/V from ctx: [head_dim, n_head_kv, n_ctx] — no input_layernorm on target_hidden
            ggml_tensor * K_all, * V_all;
            if (target_ctx && n_ctx > 0) {
                ggml_tensor * Kc = build_lora_mm(model.layers[il].wk, target_ctx);
                ggml_tensor * Vc = build_lora_mm(model.layers[il].wv, target_ctx);
                Kc = ggml_reshape_3d(ctx0, Kc, n_embd_head, n_head_kv, n_ctx);
                Vc = ggml_reshape_3d(ctx0, Vc, n_embd_head, n_head_kv, n_ctx);

                // V from noise: [head_dim, n_head_kv, n_noise]
                ggml_tensor * Vn = build_lora_mm(model.layers[il].wv, cur);
                Vn = ggml_reshape_3d(ctx0, Vn, n_embd_head, n_head_kv, n_noise);

                // Concat K: [Kc | Kn] → [head_dim, n_head_kv, n_ctx+n_noise]
                // Apply k_norm to full K (matching Python: k = k_norm(cat([k_ctx, k_noise])))
                K_all = ggml_cont(ctx0, ggml_concat(ctx0, Kc, Kn, 2));
                K_all = ggml_reshape_3d(ctx0, K_all, n_embd_head, n_head_kv, n_ctx + n_noise);
                K_all = build_norm(K_all, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);

                // Apply RoPE to full K using all positions [ctx_pos | noise_pos]
                K_all = ggml_rope_ext(ctx0, K_all, pos_all, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(K_all, "K_all_rope", il);

                // V: concat [Vc | Vn], no norm
                V_all = ggml_cont(ctx0, ggml_concat(ctx0, Vc, Vn, 2));
            } else {
                // No ctx tokens: K/V from noise only
                Kn = build_norm(Kn, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
                Kn = ggml_rope_ext(ctx0, Kn, pos_noise, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);

                ggml_tensor * Vn = build_lora_mm(model.layers[il].wv, cur);
                Vn = ggml_reshape_3d(ctx0, Vn, n_embd_head, n_head_kv, n_noise);
                K_all = Kn;
                V_all = Vn;
            }

            // Transpose for flash_attn:
            // Q: [head_dim, n_head, n_noise] → [head_dim, n_noise, n_head, 1]
            // K/V: [head_dim, n_head_kv, n_kv] → [head_dim, n_kv, n_head_kv, 1]
            Qcur  = ggml_cont(ctx0, ggml_permute(ctx0, Qcur,  0, 2, 1, 3));
            K_all = ggml_cont(ctx0, ggml_permute(ctx0, K_all, 0, 2, 1, 3));
            V_all = ggml_cont(ctx0, ggml_permute(ctx0, V_all, 0, 2, 1, 3));

            // Bidirectional flash-attn (no causal mask)
            ggml_tensor * kqv = ggml_flash_attn_ext(ctx0, Qcur, K_all, V_all,
                    /*mask=*/nullptr,
                    /*scale=*/1.0f / sqrtf((float)n_embd_head),
                    /*max_bias=*/0.0f,
                    /*logit_softcap=*/0.0f);
            cb(kqv, LLAMA_TENSOR_NAME_FATTN, il);

            // [head_dim, n_head, n_noise, 1] → [n_embd, n_noise]
            kqv = ggml_reshape_2d(ctx0, kqv, kqv->ne[0] * kqv->ne[1], kqv->ne[2] * kqv->ne[3]);
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

    // Extend t_embd to cover all n_tokens (ctx rows pass-through, noise rows = output)
    if (n_ctx > 0) {
        ggml_tensor * ctx_pass = ggml_view_2d(ctx0, inpL, n_embd, n_ctx, inpL->nb[1], 0);
        cur = ggml_cont(ctx0, ggml_concat(ctx0, ctx_pass, cur, 1));
    }
    res->t_embd   = cur;
    res->t_logits = nullptr;

    ggml_build_forward_expand(gf, cur);
}
