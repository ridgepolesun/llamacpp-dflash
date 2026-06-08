# DFlash 草稿模型 — 命令行推理工具

DFlash 是基于 Block Diffusion 的投机解码（Speculative Decoding）方案。它使用一个小型草稿模型（DFlash Draft Model）来并行预测多个候选 token，再由目标模型批量验证，从而加速自回归推理。

## 模型准备

### 1. 转换 DFlash 草稿模型到 GGUF

DFlash 草稿模型的 HuggingFace 架构标识为 `DFlashDraftModel`。使用 `convert_hf_to_gguf.py` 转换：

```bash
# 转换草稿模型（支持 DFlashDraftModel 架构）
python3 convert_hf_to_gguf.py /path/to/Qwen3.5-4B-DFlash \
    --outfile Qwen3.5-4B-DFlash-F16.gguf \
    --outtype f16

# 量化（可选，推荐 Q8_0 保持精度）
./build/bin/llama-quantize Qwen3.5-4B-DFlash-F16.gguf Qwen3.5-4B-DFlash-Q8_0.gguf Q8_0
```

### 2. 准备目标模型（Target Model）

目标模型为标准 Qwen3.5 模型，使用常规转换流程：

```bash
python3 convert_hf_to_gguf.py /path/to/Qwen3.5-9B \
    --outfile Qwen3.5-9B-Q8_0.gguf \
    --outtype q8_0
```

### 模型要求

| 模型 | 说明 |
|------|------|
| Target | 标准 Qwen3.5 模型（`qwen35` 架构），包含 embed_tokens 和 lm_head |
| Draft  | DFlash 草稿模型（`qwen3dflash` 架构），不含 embed_tokens/lm_head，依赖 target 模型的词表 |

## 编译

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc) --target llama-dflash
```

## 运行

```bash
./build/bin/llama-dflash \
    -m /path/to/Qwen3.5-9B-Q8_0.gguf \
    --model-draft /path/to/Qwen3.5-4B-DFlash-Q8_0.gguf \
    -p "你好，请介绍一下自己" \
    -n 256 \
    -ngl 99
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `-m` | 目标模型 GGUF 路径 |
| `--model-draft` / `-md` | DFlash 草稿模型 GGUF 路径 |
| `-p` | 输入 prompt |
| `-n` | 最大生成 token 数 |
| `-ngl` | GPU offload 层数（99 = 全部） |
| `-c` | 上下文长度（默认自动） |
| `--draft-ctx-max` | 草稿模型最大上下文窗口（可选，限制 draft 的 KV cache 大小） |

### 无草稿模型运行（纯自回归基线）

省略 `--model-draft` 即退化为标准自回归解码（用于对比速度）：

```bash
./build/bin/llama-dflash -m /path/to/Qwen3.5-9B-Q8_0.gguf -p "hello" -n 128 -ngl 99
```

## 输出说明

运行结束后会打印性能分析：

```
prompt eval time = ...
eval time        = ... tokens per second

decode breakdown per iteration:
    draft  fwd    = ...    (草稿模型前向)
    target verify = ...    (目标模型验证 batch)
    lm_head+LSE   = ...    (目标 lm_head 应用于草稿 embedding)
    hidden proj   = ...    (隐藏状态投影)
    snap save/restore = .. (SSM 状态快照)

draft acceptance rate = X.XX (N accepted / M generated)
```

`draft acceptance rate` 越高，投机解码加速越明显。
