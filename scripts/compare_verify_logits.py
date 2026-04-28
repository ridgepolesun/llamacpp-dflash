"""
Compare C++ vs Python DFlash verification logits (per position).
Both runs use the same 26-token chat template input.
"""
import json
import numpy as np

CPP_FILE = "/tmp/verify_logits_cpp_chat.json"
PY_FILE  = "/tmp/verify_logits_py_chat.json"
TOP_K = 100

def load(path):
    with open(path) as f:
        return json.load(f)

def topk_dict(top100):
    """Convert [[tok, logit], ...] list to {tok: logit} dict."""
    return {t: v for t, v in top100}

def main():
    cpp = load(CPP_FILE)
    py  = load(PY_FILE)

    print(f"C++ tok={cpp['tok']}  cands[:5]={cpp['cands'][:5]}")
    print(f"Py  tok={py['tok']}   cands[:5]={py['cands'][:5]}")
    print()

    n_pos = len(py["verify_logits_top100"])
    cpp_pos = {e["pos"]: e for e in cpp["verify_logits_top100"]}
    py_pos  = {e["pos"]: e for e in py["verify_logits_top100"]}

    print(f"{'pos':>4} {'input':>7} {'cpp_top1':>9} {'py_top1':>9} {'match':>6} "
          f"{'overlap':>8} {'mean_diff':>10} {'max_abs':>10} {'cand':>6} {'cpp_acc':>8} {'py_acc':>8}")
    print("-" * 100)

    cands = py["cands"]  # 15 candidates (C0..C14)
    tok   = py["tok"]

    for ni in range(n_pos):
        cp = cpp_pos.get(ni)
        pp = py_pos.get(ni)
        if cp is None or pp is None:
            print(f"  pos={ni} missing in one of the files")
            continue

        cpp_d = topk_dict(cp["top100"])
        py_d  = topk_dict(pp["top100"])

        cpp_top1 = cp["top1"]
        py_top1  = pp["top1"]
        match = (cpp_top1 == py_top1)

        common = set(cpp_d.keys()) & set(py_d.keys())
        if common:
            diffs = [cpp_d[t] - py_d[t] for t in common]
            mean_d = np.mean(diffs)
            max_d  = np.max(np.abs(diffs))
        else:
            mean_d = float('nan')
            max_d  = float('nan')

        # At pos ni, input=block[ni], logit predicts block[ni+1] = cands[ni]
        # (pos 15 = last candidate, predicts bonus — no acceptance constraint)
        if ni < len(cands):
            cand = cands[ni]
            cpp_acc = "YES" if cpp_top1 == cand else "NO "
            py_acc  = "YES" if py_top1  == cand else "NO "
        else:
            cand = -1
            cpp_acc = "---"
            py_acc  = "---"

        print(f"{ni:>4} {pp['input_tok']:>7} {cpp_top1:>9} {py_top1:>9} {str(match):>6} "
              f"{len(common):>8} {mean_d:>10.4f} {max_d:>10.4f} {cand:>6} {cpp_acc:>8} {py_acc:>8}")

    print()
    print("Legend: 'cand' = the candidate token at this position (accepted if top1 == cand)")
    print("        cpp_acc / py_acc = whether top-1 matches the candidate (drives accept/reject)")

if __name__ == "__main__":
    main()
