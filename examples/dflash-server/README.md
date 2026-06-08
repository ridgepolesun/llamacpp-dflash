# DFlash Server — HTTP 推理服务

基于 DFlash 投机解码的 HTTP 推理服务器，提供 OpenAI 兼容 API。支持 tool calling、thinking/reasoning 输出分离、streaming 和 prefix cache。

## 模型准备

### 1. 转换 DFlash 草稿模型到 GGUF

```bash
# 草稿模型（HuggingFace 架构: DFlashDraftModel）
python3 convert_hf_to_gguf.py /path/to/Qwen3.5-4B-DFlash \
    --outfile Qwen3.5-4B-DFlash-F16.gguf \
    --outtype f16

# 量化（推荐 Q8_0）
./build/bin/llama-quantize Qwen3.5-4B-DFlash-F16.gguf Qwen3.5-4B-DFlash-Q8_0.gguf Q8_0
```

### 2. 准备目标模型

```bash
python3 convert_hf_to_gguf.py /path/to/Qwen3.5-9B \
    --outfile Qwen3.5-9B-Q8_0.gguf \
    --outtype q8_0
```

## 编译

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc) --target llama-dflash-server
```

## 启动服务

```bash
./build/bin/llama-dflash-server \
    -m /path/to/Qwen3.5-9B-Q8_0.gguf \
    --model-draft /path/to/Qwen3.5-4B-DFlash-Q8_0.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 99 \
    -c 30000 \
    -n 512
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `-m` | 目标模型 GGUF 路径 |
| `--model-draft` / `-md` | DFlash 草稿模型 GGUF 路径 |
| `--host` | 监听地址（默认 127.0.0.1） |
| `--port` | 监听端口（默认 8080） |
| `-ngl` | GPU offload 层数（99 = 全部） |
| `-c` | 上下文长度 |
| `-n` | 默认最大生成 token 数 |
| `--draft-ctx-max` | 草稿模型最大上下文窗口（可选） |

## API 端点

### `GET /health`

健康检查。

### `POST /v1/chat/completions`

OpenAI 兼容的 Chat Completions API。

**请求示例：**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 200,
    "stream": false
  }'
```

**支持的请求字段：**

| 字段 | 说明 |
|------|------|
| `messages` | 对话消息数组（必填） |
| `max_tokens` | 最大生成 token 数 |
| `stream` | 是否开启流式输出 |
| `temperature` | 采样温度 |
| `top_p` / `top_k` / `min_p` | 采样参数 |
| `seed` | 随机种子 |
| `stop` | 停止词（字符串或数组） |
| `tools` | 工具定义数组（OpenAI 格式） |
| `tool_choice` | 工具选择策略 |
| `reasoning_format` | 推理格式 |
| `chat_template_kwargs` | 模板参数（如 `{"enable_thinking": true}`） |

**响应字段：**

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "回复内容",
      "reasoning_content": "思考过程（如果模型产生了 <think> 块）",
      "tool_calls": [{"type": "function", "function": {"name": "...", "arguments": "..."}}]
    },
    "finish_reason": "stop|tool_calls|length"
  }],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "draft_n_accepted": 30,
    "acceptance_rate": 0.6
  }
}
```

### `POST /completion`

llama.cpp 原生格式。

```bash
curl http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "n_predict": 128}'
```

### `POST /tokenize` / `POST /detokenize`

Tokenize / Detokenize 工具。

## Tool Calling 示例

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "查看天气"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "获取天气信息",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "城市名"}
          },
          "required": ["city"]
        }
      }
    }],
    "max_tokens": 200
  }'
```

## Streaming 示例

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "讲个故事"}], "stream": true, "max_tokens": 200}'
```

Streaming 输出：
- `<think>` 内容作为 `reasoning_content` 字段发送
- 正常回复作为 `content` 字段发送
- Tool calls 在最后一个 chunk 以结构化 `tool_calls` 字段发送
- `<tool_call>` 标签内容不会泄漏到 `content` 中

## Prefix Cache

服务器自动缓存 prompt 的完整模型状态（KV cache + SSM state）。连续请求如果 prompt 相同，跳过 prefill 阶段（从 ~15秒降到 ~0ms）。

日志中会显示：
```
generate: saved full state for prefix cache (XX.X MiB)
generate: prefix cache hit: XXXXX / XXXXX tokens reused (full state restore)
```

## 与 llama-server 的区别

| 特性 | dflash-server | llama-server |
|------|:---:|:---:|
| DFlash 投机解码 | ✅ | ❌ |
| 并发请求 | ❌（单请求串行） | ✅（多 slot） |
| 多模态 | ❌ | ✅ |
| LoRA | ❌ | ✅ |
| Grammar 约束 | 有限（工具调用） | 完整 |
| Embedding/Rerank | ❌ | ✅ |
