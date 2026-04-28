"""
Compare draft model outputs between C++ and Python DFlash implementations.
The draft model takes target hidden states as context + noise embeddings.
We compare the draft's predicted logits for position 1 (first candidate after tok).
"""
import json
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

TARGET_MODEL_PATH = "/home/nvidia/user/sundongdong/model/Qwen3-4B"
DRAFT_MODEL_PATH  = "/home/nvidia/user/sundongdong/model/Qwen3-4B-DFlash-b16"

# Fixed token IDs (same as C++ test): raw prompt without chat template
TOKENS = [4340, 1657, 6785, 4361, 25854, 3429, 41214, 1558, 220, 16, 24, 21, 614, 30]

TOP_K = 100

def get_target_hidden_and_logits(target, token_ids, target_layer_ids):
    """Run target model, extract hidden states at specified layers + logits."""
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(target.device)
    with torch.inference_mode():
        out = target(input_ids, use_cache=False, output_hidden_states=True)
    # hidden_states: tuple of (n_layers+1) tensors, each [1, seq, hidden]
    # target_layer_ids index into [layer+1] (offset by 1 for embedding layer)
    hidden_list = []
    for lid in target_layer_ids:
        h = out.hidden_states[lid + 1]  # +1 because index 0 = embedding
        hidden_list.append(h)  # [1, seq, hidden]
    hidden_concat = torch.cat(hidden_list, dim=-1)  # [1, seq, n_layers*hidden]
    logits = out.logits[0, -1].float().cpu().numpy()
    return hidden_concat, logits

def get_draft_prediction(target, draft, token_ids, target_layer_ids, mask_token_id, block_size=16):
    """Run one DFlash draft step: get target hidden, then run draft for block_size tokens."""
    from transformers import DynamicCache
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(target.device)
    n_tokens = len(token_ids)

    with torch.inference_mode():
        # Target prefill — get all hidden states and sample tok
        tgt_out = target(input_ids, use_cache=False, output_hidden_states=True)
        tok = torch.argmax(tgt_out.logits[0, -1]).unsqueeze(0).unsqueeze(0)  # [1,1]

        # target_hidden: all n_tokens positions, concat across layers
        # hidden_states[0] = embedding, hidden_states[lid+1] = layer lid output
        hidden_list = []
        for lid in target_layer_ids:
            hidden_list.append(tgt_out.hidden_states[lid + 1])  # [1, n_tokens, hidden]
        # target_hidden shape: [1, n_tokens, n_layers * hidden]
        target_hidden = torch.cat(hidden_list, dim=-1).to(dtype=next(draft.parameters()).dtype)

        # Build noise tokens: [tok | MASK | MASK | ...] — block_size tokens
        noise_ids = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=target.device)
        noise_ids[:, 0] = tok[0, 0]
        noise_embedding = target.model.embed_tokens(noise_ids).to(dtype=next(draft.parameters()).dtype)

        # position_ids: cover [0, n_tokens + block_size)
        # (same as dflash_generate: position_ids[:, 0: start + block_size] on first call)
        pos_ids = torch.arange(n_tokens + block_size, device=target.device).unsqueeze(0)

        # Run draft model
        draft_hidden = draft(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=pos_ids,
            use_cache=False,
        )  # [1, block_size, hidden]

        draft_logits = target.lm_head(
            draft_hidden.to(dtype=next(target.parameters()).dtype)
        )  # [1, block_size, vocab]

    # candidate positions: indices 1..block_size-1 in draft output
    # (index 0 is prediction for C0 using tok as "noise", indices 1+ predict C1, C2, ...)
    cand_logits = draft_logits[0, 1:, :].float().cpu().numpy()  # [block_size-1, vocab]
    tok_id = int(tok[0, 0].item())
    return tok_id, cand_logits

def topk_dict(logits_1d, k=TOP_K):
    idx = np.argsort(logits_1d)[::-1][:k]
    return {int(i): float(logits_1d[i]) for i in idx}

def compare_dicts(name_a, logits_a_1d, name_b, logits_b_1d, pos_label):
    da = topk_dict(logits_a_1d)
    db = topk_dict(logits_b_1d)
    ta = int(np.argmax(logits_a_1d))
    tb = int(np.argmax(logits_b_1d))
    common = set(da.keys()) & set(db.keys())
    diffs = [logits_a_1d[t] - logits_b_1d[t] for t in common]
    print(f"  [{pos_label}] top-1: {name_a}={ta}  {name_b}={tb}  match={ta==tb}")
    print(f"           top-{TOP_K} overlap={len(common)}  "
          f"diff mean={np.mean(diffs):.4f} std={np.std(diffs):.4f} max_abs={np.max(np.abs(diffs)):.4f}")

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
    print(f"  target_layer_ids={target_layer_ids}  mask_token_id={mask_token_id}  block_size={block_size}")

    print(f"\nTokens: {TOKENS}")
    print("\nRunning draft model...")
    tok_id, cand_logits = get_draft_prediction(
        target, draft, TOKENS, target_layer_ids, mask_token_id, block_size
    )
    print(f"  tok (greedy from target prefill) = {tok_id}")
    print(f"  draft candidate logits shape: {cand_logits.shape}")

    # Save candidate logits for C++ comparison
    out = {
        "tokens": TOKENS,
        "tok_id": tok_id,
        "cand_logits_top100": []
    }
    for i in range(min(5, cand_logits.shape[0])):
        d = topk_dict(cand_logits[i])
        top1 = int(np.argmax(cand_logits[i]))
        out["cand_logits_top100"].append({
            "pos": i,
            "top1": top1,
            "top1_logit": float(cand_logits[i, top1]),
            "top100": [[k, v] for k, v in sorted(d.items(), key=lambda x: -x[1])[:TOP_K]]
        })

    with open("/tmp/draft_logits_py.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n  Top-5 draft candidate predictions:")
    for i in range(min(5, cand_logits.shape[0])):
        top1 = int(np.argmax(cand_logits[i]))
        top1_logit = float(cand_logits[i, top1])
        print(f"    C[{i}]: top1_token={top1}  logit={top1_logit:.4f}")

    print("\nPython draft logits saved to /tmp/draft_logits_py.json")

if __name__ == "__main__":
    main()
