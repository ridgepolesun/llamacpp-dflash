#!/usr/bin/env python3
"""
DFlash GGUF Validation Script
==============================
Validates:
1. GGUF weight correctness — compares GGUF tensors to HF safetensors.
2. DFlash forward pass — runs the HF reference implementation to confirm
   the model produces plausible outputs.

Usage:
    python3 scripts/validate_dflash.py
"""

import sys
import os
import struct
import json
import torch
import numpy as np

HF_DRAFT_PATH  = "/home/nvidia/user/sundongdong/model/Qwen3.5-4B-DFlash"
GGUF_PATH      = "/home/nvidia/user/sundongdong/model/Qwen3.5-4B-DFlash/Qwen3.5-4B-DFlash-F16.gguf"
DFLASH_SRC     = "/home/nvidia/user/sundongdong/project/algorithm/dflash"

# ── 0. Minimal GGUF reader ──────────────────────────────────────────────────

GGUF_MAGIC = 0x46554747
GGUFValueType = {
    0: "uint8", 1: "int8", 2: "uint16", 3: "int16",
    4: "uint32", 5: "int32", 6: "float32", 7: "bool",
    8: "string", 9: "array", 10: "uint64", 11: "int64", 12: "float64",
}
GGUF_TYPE_SIZE = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

def read_gguf_string(f):
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")

def read_gguf_value(f, vtype):
    if vtype == 8:  # string
        return read_gguf_string(f)
    elif vtype == 9:  # array
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        return [read_gguf_value(f, elem_type) for _ in range(count)]
    elif vtype in GGUF_TYPE_SIZE:
        fmt = {"uint8":"B","int8":"b","uint16":"H","int16":"h",
               "uint32":"I","int32":"i","float32":"f","bool":"?",
               "uint64":"Q","int64":"q","float64":"d"}[GGUFValueType[vtype]]
        return struct.unpack(f"<{fmt}", f.read(GGUF_TYPE_SIZE[vtype]))[0]
    else:
        raise ValueError(f"Unknown GGUF value type {vtype}")

GGML_DTYPE_TO_NP = {0: np.float32, 1: np.float16}  # F32, F16

def load_gguf_tensors(path):
    """Return {name: np.ndarray} for all F16/F32 tensors in the GGUF file."""
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == GGUF_MAGIC, "Not a GGUF file"
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        # Skip KV metadata
        for _ in range(n_kv):
            key = read_gguf_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            val = read_gguf_value(f, vtype)

        # Read tensor metadata
        tensor_meta = []
        for _ in range(n_tensors):
            name = read_gguf_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims))
            dtype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensor_meta.append((name, dims, dtype, offset))

        # Data starts at alignment boundary
        alignment = 32
        data_start = f.tell()
        remainder = data_start % alignment
        if remainder:
            data_start += alignment - remainder
        # Actually we need to seek to the alignment boundary
        pos = f.tell()
        if pos % alignment:
            f.seek(alignment - (pos % alignment), 1)
        data_start = f.tell()

        tensors = {}
        for name, dims, dtype, rel_offset in tensor_meta:
            if dtype not in GGML_DTYPE_TO_NP:
                continue  # skip quantized
            np_dtype = GGML_DTYPE_TO_NP[dtype]
            abs_offset = data_start + rel_offset
            f.seek(abs_offset)
            n_elems = 1
            for d in dims:
                n_elems *= d
            data = np.frombuffer(f.read(n_elems * np.dtype(np_dtype).itemsize), dtype=np_dtype)
            # GGUF dims: innermost first — same as C row-major if we reverse
            tensors[name] = data.reshape(dims[::-1])
        return tensors


# ── 1. Load GGUF tensors ────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Loading GGUF tensors from", GGUF_PATH)
gguf_tensors = load_gguf_tensors(GGUF_PATH)
print(f"  Loaded {len(gguf_tensors)} F16 tensors")


# ── 2. Load HF safetensors ──────────────────────────────────────────────────

print("\nSTEP 2: Loading HF safetensors from", HF_DRAFT_PATH)
try:
    from safetensors.torch import load_file
    hf_path = os.path.join(HF_DRAFT_PATH, "model.safetensors")
    hf_tensors = load_file(hf_path, device="cpu")
    print(f"  Loaded {len(hf_tensors)} HF tensors")
except ImportError:
    print("  safetensors not available, skipping weight comparison")
    hf_tensors = None


# ── 3. Weight comparison ────────────────────────────────────────────────────

if hf_tensors is not None:
    print("\nSTEP 3: Comparing GGUF vs HF weights")

    # Mapping: HF name -> GGUF name
    # Global
    NAME_MAP = {
        "fc.weight":          "dflash.fc.weight",
        "hidden_norm.weight": "dflash.hidden_norm.weight",
        "norm.weight":        "output_norm.weight",
    }
    # Per-layer
    LAYER_MAP = {
        "layers.{i}.input_layernorm.weight":            "blk.{i}.attn_norm.weight",
        "layers.{i}.self_attn.q_proj.weight":           "blk.{i}.attn_q.weight",
        "layers.{i}.self_attn.k_proj.weight":           "blk.{i}.attn_k.weight",
        "layers.{i}.self_attn.v_proj.weight":           "blk.{i}.attn_v.weight",
        "layers.{i}.self_attn.o_proj.weight":           "blk.{i}.attn_output.weight",
        "layers.{i}.self_attn.q_norm.weight":           "blk.{i}.attn_q_norm.weight",
        "layers.{i}.self_attn.k_norm.weight":           "blk.{i}.attn_k_norm.weight",
        "layers.{i}.post_attention_layernorm.weight":   "blk.{i}.ffn_norm.weight",
        "layers.{i}.mlp.gate_proj.weight":              "blk.{i}.ffn_gate.weight",
        "layers.{i}.mlp.down_proj.weight":              "blk.{i}.ffn_down.weight",
        "layers.{i}.mlp.up_proj.weight":                "blk.{i}.ffn_up.weight",
    }

    n_layers = 5
    all_maps = dict(NAME_MAP)
    for i in range(n_layers):
        for hf_pat, gguf_pat in LAYER_MAP.items():
            all_maps[hf_pat.format(i=i)] = gguf_pat.format(i=i)

    errors = 0
    checked = 0
    for hf_name, gguf_name in all_maps.items():
        if hf_name not in hf_tensors:
            print(f"  SKIP  {hf_name} (not in HF)")
            continue
        if gguf_name not in gguf_tensors:
            print(f"  MISS  {gguf_name} (not in GGUF)")
            errors += 1
            continue

        hf_t  = hf_tensors[hf_name].float().numpy()
        gg_t  = gguf_tensors[gguf_name].astype(np.float32)

        # GGUF stores tensors transposed relative to HF for matmul weights
        # (HF: [out, in], GGUF: [in, out] after reshape)
        if hf_t.shape != gg_t.shape:
            if hf_t.shape == gg_t.T.shape:
                gg_t = gg_t.T
            else:
                print(f"  SHAPE MISMATCH {hf_name}: HF {hf_t.shape} vs GGUF {gg_t.shape}")
                errors += 1
                continue

        # Compare in F16 precision (GGUF is F16, round-trip introduces small error)
        hf_f16 = hf_t.astype(np.float16).astype(np.float32)
        gg_f16 = gg_t.astype(np.float16).astype(np.float32)
        max_diff = np.abs(hf_f16 - gg_f16).max()
        rel_diff = max_diff / (np.abs(hf_f16).max() + 1e-8)

        status = "OK  " if rel_diff < 1e-3 else "WARN"
        if rel_diff >= 1e-3:
            errors += 1
        print(f"  {status} {gguf_name:45s} max_diff={max_diff:.2e} rel={rel_diff:.2e}")
        checked += 1

    print(f"\n  Checked {checked} tensors, {errors} mismatches")
    if errors == 0:
        print("  ✓ All weights match between HF and GGUF (within F16 precision)")
    else:
        print("  ✗ Some weights do NOT match — investigate above")


# ── 4. DFlash HF forward pass ───────────────────────────────────────────────

print("\nSTEP 4: DFlash HF model forward pass sanity check")

try:
    # Patch scipy before transformers imports it to avoid numpy.Inf incompatibility
    import sys as _sys, types as _types, importlib.machinery as _imm
    if "scipy" not in _sys.modules:
        _scipy = _types.ModuleType("scipy")
        _scipy.__spec__ = _imm.ModuleSpec("scipy", None)
        _scipy_opt = _types.ModuleType("scipy.optimize")
        _scipy_opt.__spec__ = _imm.ModuleSpec("scipy.optimize", None)
        _scipy_opt.linear_sum_assignment = None
        _scipy.optimize = _scipy_opt
        _sys.modules["scipy"] = _scipy
        _sys.modules["scipy.optimize"] = _scipy_opt

    from transformers import AutoConfig, AutoModel
    import torch

    cfg = AutoConfig.from_pretrained(HF_DRAFT_PATH, trust_remote_code=True)
    print(f"  Config: {cfg.architectures}, layers={cfg.num_hidden_layers}, "
          f"hidden={cfg.hidden_size}, target_layers={cfg.num_target_layers}")
    print(f"  dflash_config: {cfg.dflash_config}")

    draft = AutoModel.from_pretrained(
        HF_DRAFT_PATH, trust_remote_code=True, torch_dtype=torch.float32
    ).eval()
    n_params = sum(p.numel() for p in draft.parameters())
    print(f"  Draft model loaded: {n_params/1e6:.1f}M parameters")

    n_embd  = cfg.hidden_size
    n_target_layers = len(cfg.dflash_config.get("target_layer_ids", [1, 8, 15, 22, 29]))
    n_ctx   = 10
    n_noise = 4

    torch.manual_seed(42)
    target_hidden   = torch.randn(1, n_ctx, n_embd * n_target_layers)
    noise_embedding = torch.randn(1, n_noise, n_embd)
    # position_ids must cover all positions: ctx tokens + noise tokens
    # (RoPE is applied to concat(K_ctx, K_noise) which has ctx+noise tokens)
    position_ids    = torch.arange(n_ctx + n_noise).unsqueeze(0)

    with torch.inference_mode():
        out = draft(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
        )

    print(f"  Forward pass output shape: {out.shape}  (expected [1, {n_noise}, {n_embd}])")
    assert out.shape == (1, n_noise, n_embd), f"Unexpected shape: {out.shape}"
    print(f"  Output norm: {out.norm().item():.4f}")
    assert out.isfinite().all(), "Output contains NaN/Inf!"
    print("  ✓ DFlash forward pass produces finite outputs of correct shape")

except Exception as e:
    import traceback
    print(f"  ✗ Forward pass failed: {e}")
    traceback.print_exc()


print("\n" + "=" * 60)
print("Validation complete.")
