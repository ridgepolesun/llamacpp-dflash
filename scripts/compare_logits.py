"""
Compare prefill logits between C++ (GGUF) and Python (transformers) implementations.
Runs target BF16 model and draft F16 model.
"""
import json
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

PROMPT = "How many positive whole-number divisors does 196 have?"
TARGET_MODEL_PATH = "/home/nvidia/user/sundongdong/model/Qwen3-4B"
DRAFT_MODEL_PATH  = "/home/nvidia/user/sundongdong/model/Qwen3-4B-DFlash-b16"
CPP_BF16_JSON     = "/tmp/logits_cpp_bf16.json"
CPP_Q8_JSON       = "/tmp/logits_cpp_q8.json"
TOP_K = 200

def get_hf_logits(model, tokenizer, token_ids=None, prompt=None):
    """Run model on the given token_ids (or tokenize prompt if token_ids is None)."""
    if token_ids is not None:
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(model.device)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    print(f"  n_tokens={input_ids.shape[1]}", flush=True)
    print(f"  tokens={input_ids[0].tolist()}", flush=True)
    with torch.inference_mode():
        out = model(input_ids, use_cache=False)
    logits = out.logits[0, -1].float().cpu().numpy()  # [vocab]
    return logits, input_ids[0].tolist()

def load_cpp_json(path):
    with open(path) as f:
        d = json.load(f)
    top = d["top_logits"]  # list of [token_id, logit]
    return {t: v for t, v in top}, d["tokens"]

def compare(name_a, logits_a, name_b, logits_b, tokens_a, tokens_b):
    print(f"\n{'='*60}")
    print(f"  {name_a} vs {name_b}")
    print(f"{'='*60}")
    if tokens_a != tokens_b:
        print(f"  WARNING: token mismatch!")
        print(f"    {name_a}: {tokens_a}")
        print(f"    {name_b}: {tokens_b}")
    else:
        print(f"  Tokens match: {tokens_a}")

    # Get top-K from logits_a (numpy array)
    topk_a = np.argsort(logits_a)[::-1][:TOP_K]

    print(f"\n  Top-10 from {name_a}:")
    print(f"  {'rank':>4} {'tok_id':>7} {'logit_a':>10} {'logit_b':>10} {'diff':>10}")
    for i, tid in enumerate(topk_a[:10]):
        la = logits_a[tid]
        lb = logits_b.get(int(tid), float('nan'))
        diff = la - lb if not np.isnan(lb) else float('nan')
        print(f"  {i+1:>4} {tid:>7} {la:>10.4f} {lb:>10.4f} {diff:>10.4f}")

    # Compute rank correlation for top-200
    ranks_a = {int(tid): i for i, tid in enumerate(topk_a[:TOP_K])}
    # Get sorted list from logits_b
    logits_b_sorted = sorted(logits_b.items(), key=lambda x: -x[1])
    ranks_b = {tid: i for i, (tid, _) in enumerate(logits_b_sorted[:TOP_K])}

    common = set(ranks_a.keys()) & set(ranks_b.keys())
    print(f"\n  Top-{TOP_K} overlap: {len(common)}/{TOP_K} tokens in common")

    # For tokens that appear in BOTH top-200, compute logit differences
    if common:
        diffs = [logits_a[tid] - logits_b[tid] for tid in common]
        print(f"  Logit diff (a-b) for overlapping tokens:")
        print(f"    mean={np.mean(diffs):.4f}  std={np.std(diffs):.4f}  max_abs={np.max(np.abs(diffs)):.4f}")

    # Top-1 match
    top1_a = int(topk_a[0])
    top1_b = int(logits_b_sorted[0][0])
    print(f"\n  Top-1: {name_a}={top1_a}  {name_b}={top1_b}  match={top1_a == top1_b}")

def main():
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_PATH)

    print("Loading target model (BF16)...")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_PATH, dtype="auto", device_map="cuda:0"
    ).eval()
    print(f"  dtype: {next(target.parameters()).dtype}")

    print("Getting target BF16 logits (Python)...")
    # Load C++ token IDs first to ensure same input
    try:
        with open(CPP_BF16_JSON) as f:
            cpp_token_ids = json.load(f)["tokens"]
        print(f"  Using C++ token IDs: {cpp_token_ids}")
        py_bf16_logits, py_tokens = get_hf_logits(target, tokenizer, token_ids=cpp_token_ids)
    except FileNotFoundError:
        print("  C++ JSON not found, using raw tokenization")
        py_bf16_logits, py_tokens = get_hf_logits(target, tokenizer, prompt=PROMPT)

    # Save Python BF16 logits as dict for comparison
    topk_py = np.argsort(py_bf16_logits)[::-1][:TOP_K]
    py_bf16_dict = {int(tid): float(py_bf16_logits[tid]) for tid in topk_py}

    del target
    torch.cuda.empty_cache()

    # Load C++ logit files
    try:
        cpp_bf16_dict, cpp_bf16_tokens = load_cpp_json(CPP_BF16_JSON)
        print(f"\nLoaded C++ BF16 logits from {CPP_BF16_JSON}")
        compare("Python-BF16", py_bf16_logits, "C++-BF16", cpp_bf16_dict, py_tokens, cpp_bf16_tokens)
    except FileNotFoundError:
        print(f"C++ BF16 JSON not found: {CPP_BF16_JSON}")

    try:
        cpp_q8_dict, cpp_q8_tokens = load_cpp_json(CPP_Q8_JSON)
        print(f"\nLoaded C++ Q8_0 logits from {CPP_Q8_JSON}")
        compare("Python-BF16", py_bf16_logits, "C++-Q8_0", cpp_q8_dict, py_tokens, cpp_q8_tokens)
    except FileNotFoundError:
        print(f"C++ Q8_0 JSON not found: {CPP_Q8_JSON} (run Q8_0 dump separately)")

    print("\nDone.")

if __name__ == "__main__":
    main()
