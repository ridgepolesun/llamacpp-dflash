"""
Dump target model verification logits for the first DFlash iteration.
Replicates Python DFlash's verification step:
  - prefill with prompt tokens
  - build block = [tok | C0..C14]
  - decode block with target model (KV cache from prefill)
  - extract logits at each position
"""
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, DynamicCache

TARGET_MODEL_PATH = "/home/nvidia/user/sundongdong/model/Qwen3-4B"
DRAFT_MODEL_PATH  = "/home/nvidia/user/sundongdong/model/Qwen3-4B-DFlash-b16"
TOKENS = [151644, 872, 198, 4340, 1657, 6785, 4361, 25854, 3429, 41214,
          1558, 220, 16, 24, 21, 614, 30, 151645, 198, 151644,
          77091, 198, 151667, 271, 151668, 271]  # chat template, 26 tokens
TOP_K = 100

def main():
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_PATH)

    print("Loading target model (BF16)...")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_PATH, dtype="auto", device_map="cuda:0"
    ).eval()

    print("Loading draft model...")
    draft = AutoModel.from_pretrained(
        DRAFT_MODEL_PATH, trust_remote_code=True, dtype="auto", device_map="cuda:0"
    ).eval()

    target_layer_ids = draft.target_layer_ids
    mask_token_id = draft.mask_token_id
    block_size = draft.block_size
    n_cands = block_size - 1
    print(f"  block_size={block_size}  mask_token_id={mask_token_id}")

    input_ids = torch.tensor([TOKENS], dtype=torch.long).to(target.device)
    n_tokens = len(TOKENS)

    with torch.inference_mode():
        # === Prefill ===
        past_kv = DynamicCache()
        pos_all = torch.arange(n_tokens + block_size, device=target.device).unsqueeze(0)

        prefill_out = target(
            input_ids,
            position_ids=pos_all[:, :n_tokens],
            past_key_values=past_kv,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        tok = int(torch.argmax(prefill_out.logits[0, 0]).item())
        print(f"  tok (from prefill) = {tok}")

        # === Draft: get candidates ===
        hidden_list = []
        for lid in target_layer_ids:
            hidden_list.append(prefill_out.hidden_states[lid + 1])
        target_hidden = torch.cat(hidden_list, dim=-1).to(dtype=next(draft.parameters()).dtype)

        noise_ids = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=target.device)
        noise_ids[0, 0] = tok
        noise_embedding = target.model.embed_tokens(noise_ids).to(dtype=next(draft.parameters()).dtype)

        draft_hidden = draft(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=pos_all[:, :n_tokens + block_size],
            use_cache=False,
        )
        draft_logits = target.lm_head(draft_hidden.to(dtype=next(target.parameters()).dtype))
        candidates = [int(torch.argmax(draft_logits[0, i+1]).item()) for i in range(n_cands)]
        print(f"  candidates[:5] = {candidates[:5]}")

        # === Verification batch: [tok | C0..C_{n_cands-1}] ===
        block_ids = torch.tensor([[tok] + candidates], dtype=torch.long).to(target.device)
        block_pos = pos_all[:, n_tokens: n_tokens + block_size]

        verify_out = target(
            block_ids,
            position_ids=block_pos,
            past_key_values=past_kv,
            use_cache=True,
        )
        # verify_out.logits shape: [1, block_size, vocab]
        verify_logits = verify_out.logits[0].float().cpu().numpy()  # [block_size, vocab]

    # Dump
    out = {
        "tokens": TOKENS,
        "tok": tok,
        "cands": candidates,
        "n_past": n_tokens,
        "verify_logits_top100": []
    }
    print(f"\nVerification logits (top-1 per position):")
    for ni in range(block_size):
        lg = verify_logits[ni]
        top_idx = np.argsort(lg)[::-1][:TOP_K]
        top1 = int(top_idx[0])
        top1_logit = float(lg[top1])
        input_tok = tok if ni == 0 else candidates[ni-1]
        out["verify_logits_top100"].append({
            "pos": ni,
            "input_tok": input_tok,
            "top1": top1,
            "top1_logit": top1_logit,
            "top100": [[int(t), float(lg[t])] for t in top_idx]
        })
        print(f"  pos={ni} input={input_tok} top1={top1} logit={top1_logit:.4f}")

    with open("/tmp/verify_logits_py_chat.json", "w") as f:
        json.dump(out, f)
    print("\nSaved to /tmp/verify_logits_py.json")

if __name__ == "__main__":
    main()
