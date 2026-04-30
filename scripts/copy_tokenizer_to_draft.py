#!/usr/bin/env python3
"""
Binary-level copy of tokenizer KV entries from a source GGUF into a draft GGUF.
Does NOT touch tensor data — just inserts raw KV bytes into the header section.

Usage:
    python3 copy_tokenizer_to_draft.py <source.gguf> <draft.gguf> [--inplace]

Without --inplace: writes to <draft_stem>_with_vocab.gguf
With    --inplace: overwrites draft in place
"""

import sys
import os
import struct
import argparse
import shutil


# ── GGUF binary primitives ────────────────────────────────────────────────────

GGUF_MAGIC   = b"GGUF"
GGUF_VERSION = 3

# GGUFValueType
TYPE_UINT8   = 0
TYPE_INT8    = 1
TYPE_UINT16  = 2
TYPE_INT16   = 3
TYPE_UINT32  = 4
TYPE_INT32   = 5
TYPE_FLOAT32 = 6
TYPE_BOOL    = 7
TYPE_STRING  = 8
TYPE_ARRAY   = 9
TYPE_UINT64  = 10
TYPE_INT64   = 11
TYPE_FLOAT64 = 12

TYPE_SIZES = {
    TYPE_UINT8:   1,
    TYPE_INT8:    1,
    TYPE_UINT16:  2,
    TYPE_INT16:   2,
    TYPE_UINT32:  4,
    TYPE_INT32:   4,
    TYPE_FLOAT32: 4,
    TYPE_BOOL:    1,
    TYPE_UINT64:  8,
    TYPE_INT64:   8,
    TYPE_FLOAT64: 8,
}


def read_u32(data: bytes, off: int) -> tuple[int, int]:
    return struct.unpack_from('<I', data, off)[0], off + 4

def read_u64(data: bytes, off: int) -> tuple[int, int]:
    return struct.unpack_from('<Q', data, off)[0], off + 8

def read_string_raw(data: bytes, off: int) -> tuple[bytes, int]:
    n, off = read_u64(data, off)
    return data[off:off+n], off + n

def skip_value(data: bytes, off: int, vtype: int) -> int:
    """Advance offset past one GGUF value of given type. Returns new offset."""
    if vtype in TYPE_SIZES:
        return off + TYPE_SIZES[vtype]
    if vtype == TYPE_STRING:
        n, off = read_u64(data, off)
        return off + n
    if vtype == TYPE_ARRAY:
        elem_type, off = read_u32(data, off)
        count, off = read_u64(data, off)
        for _ in range(count):
            off = skip_value(data, off, elem_type)
        return off
    raise ValueError(f"Unknown GGUF value type {vtype} at offset {off}")

def skip_kv(data: bytes, off: int) -> int:
    """Advance offset past one KV entry. Returns new offset."""
    _, off = read_string_raw(data, off)   # key
    vtype, off = read_u32(data, off)       # value type
    off = skip_value(data, off, vtype)    # value
    return off

def read_kv_name(data: bytes, off: int) -> tuple[str, int]:
    """Read just the key name, do NOT advance past the value."""
    key_bytes, off = read_string_raw(data, off)
    return key_bytes.decode('utf-8', errors='replace'), off


# ── Main logic ────────────────────────────────────────────────────────────────

TOKENIZER_PREFIXES = ("tokenizer.",)

def is_tokenizer_key(name: str) -> bool:
    return any(name.startswith(p) for p in TOKENIZER_PREFIXES)


def extract_kv_map(data: bytes, n_kv: int, kv_start: int) -> dict[str, tuple[int, int]]:
    """
    Returns {key_name: (start_offset, end_offset)} for each KV in the file.
    """
    off = kv_start
    result = {}
    for _ in range(n_kv):
        start = off
        name, off2 = read_kv_name(data, off)
        vtype, off2 = read_u32(data, off2)
        end = skip_value(data, off2, vtype)
        result[name] = (start, end)
        off = end
    return result


def parse_header(data: bytes) -> tuple[int, int, int, int]:
    """Returns (n_tensors, n_kv, kv_start, version)."""
    assert data[:4] == GGUF_MAGIC, "Not a GGUF file"
    version, off = read_u32(data, 4)
    n_tensors, off = read_u64(data, off)
    n_kv, off = read_u64(data, off)
    return n_tensors, n_kv, off, version   # off = kv_start


def main():
    parser = argparse.ArgumentParser(
        description="Copy tokenizer KV entries from source GGUF into draft GGUF (binary patch)")
    parser.add_argument("source", help="Source GGUF (target model with tokenizer)")
    parser.add_argument("draft",  help="Draft GGUF (missing tokenizer)")
    parser.add_argument("--inplace", action="store_true",
                        help="Overwrite draft file in place (otherwise write <name>_with_vocab.gguf)")
    parser.add_argument("--outfile", default=None,
                        help="Explicit output path (overrides --inplace and default naming)")
    args = parser.parse_args()

    if args.outfile:
        out_path = args.outfile
    elif args.inplace:
        out_path = args.draft
    else:
        base, ext = os.path.splitext(args.draft)
        out_path = base + "_with_vocab" + ext

    print(f"Source : {args.source}")
    print(f"Draft  : {args.draft}")
    print(f"Output : {out_path}")

    # ── Load files ────────────────────────────────────────────────────────────
    print("Reading source GGUF...", flush=True)
    with open(args.source, 'rb') as f:
        src_data = f.read()

    print("Reading draft GGUF...", flush=True)
    with open(args.draft, 'rb') as f:
        dft_data = f.read()

    # ── Parse headers ─────────────────────────────────────────────────────────
    src_n_tensors, src_n_kv, src_kv_start, _ = parse_header(src_data)
    dft_n_tensors, dft_n_kv, dft_kv_start, dft_version = parse_header(dft_data)

    print(f"Source: {src_n_tensors} tensors, {src_n_kv} KV entries")
    print(f"Draft : {dft_n_tensors} tensors, {dft_n_kv} KV entries")

    # ── Map KV entries in both files ──────────────────────────────────────────
    print("Scanning source KV entries...")
    src_kvs = extract_kv_map(src_data, src_n_kv, src_kv_start)

    print("Scanning draft KV entries...")
    dft_kvs = extract_kv_map(dft_data, dft_n_kv, dft_kv_start)

    # ── Decide what to copy ───────────────────────────────────────────────────
    to_copy = {}
    for name, span in src_kvs.items():
        if is_tokenizer_key(name) and name not in dft_kvs:
            to_copy[name] = span

    already = [n for n in src_kvs if is_tokenizer_key(n) and n in dft_kvs]
    print(f"\nTokenizer fields in source  : {sum(1 for k in src_kvs if is_tokenizer_key(k))}")
    print(f"  Already in draft (skip)   : {len(already)}")
    print(f"  To copy                   : {len(to_copy)}")
    for name in sorted(to_copy):
        start, end = to_copy[name]
        print(f"    + {name}  ({end - start} bytes)")

    if not to_copy:
        print("\nNothing to copy — draft already has all tokenizer fields.")
        return

    # ── Build new tokenizer blob (raw bytes from source) ──────────────────────
    tok_blob = b"".join(src_data[s:e] for s, e in to_copy.values())

    # ── Find end of draft KV section ─────────────────────────────────────────
    off = dft_kv_start
    for _ in range(dft_n_kv):
        off = skip_kv(dft_data, off)
    dft_kv_end = off

    # ── Find end of draft tensor-info section ────────────────────────────────
    # Tensor info entries: name(str) + n_dims(u32) + dims(u64×n_dims) + type(u32) + offset(u64)
    off = dft_kv_end
    for _ in range(dft_n_tensors):
        _, off = read_string_raw(dft_data, off)  # name
        n_dims, off = read_u32(dft_data, off)
        for _ in range(n_dims):
            _, off = read_u64(dft_data, off)
        _, off = read_u32(dft_data, off)  # type
        _, off = read_u64(dft_data, off)  # offset
    dft_ti_end = off

    # ── Separate tensor info from tensor data (re-aligning padding) ──────────
    # GGUF alignment: tensor data must start at a file offset that is a multiple
    # of ALIGNMENT.  The padding bytes between tensor-info and tensor-data
    # depend on the size of everything that comes before, so they must be
    # recomputed whenever the KV section grows.
    ALIGNMENT = 32  # default; TODO: honour general.alignment KV if present
    old_tdata_start = dft_ti_end + (-dft_ti_end % ALIGNMENT)

    original_kv_bytes = dft_data[dft_kv_start:dft_kv_end]
    tensor_info_bytes = dft_data[dft_kv_end:dft_ti_end]
    tensor_data       = dft_data[old_tdata_start:]

    # New tensor-info ends at: header(24) + kv + tok_blob + tensor_info
    new_ti_end = 24 + len(original_kv_bytes) + len(tok_blob) + len(tensor_info_bytes)
    new_padding = bytes(-new_ti_end % ALIGNMENT)

    # ── Build new file ─────────────────────────────────────────────────────────
    # Header: magic(4) + version(4) + n_tensors(8) + n_kv(8)  = 24 bytes
    new_n_kv = dft_n_kv + len(to_copy)
    new_header = (
        GGUF_MAGIC
        + struct.pack('<I', dft_version)
        + struct.pack('<Q', dft_n_tensors)
        + struct.pack('<Q', new_n_kv)
    )

    print(f"\nBuilding output file...")
    print(f"  Original KV section : {len(original_kv_bytes):,} bytes")
    print(f"  New tokenizer blob  : {len(tok_blob):,} bytes")
    print(f"  Tensor info section : {len(tensor_info_bytes):,} bytes")
    print(f"  Alignment padding   : {len(new_padding)} bytes (was {old_tdata_start - dft_ti_end})")
    print(f"  Tensor data         : {len(tensor_data):,} bytes")

    tmp_path = out_path + ".tmp"
    with open(tmp_path, 'wb') as f:
        f.write(new_header)
        f.write(original_kv_bytes)
        f.write(tok_blob)
        f.write(tensor_info_bytes)
        f.write(new_padding)
        f.write(tensor_data)

    os.replace(tmp_path, out_path)
    out_size = os.path.getsize(out_path)
    print(f"\nDone → {out_path}  ({out_size / 1024**3:.2f} GB)")


if __name__ == "__main__":
    main()
