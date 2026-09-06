import json
import os
import re
import time
import threading
import itertools
from typing import Dict, Any, List, Optional, Tuple
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

# 本机固定只用 1 号 P40（单卡 24GB 装得下：16GB 权重 + 16K KV ~3GB + 缓冲 ≈ 20GB）。
# GPU0 常驻 OCR 服务，按 C-Land 分卡并行惯例 gemma 独占 GPU1（免 PCIe 卡间同步，decode 更快）。
# 如需临时改卡/临时双卡：启动前显式 export CUDA_VISIBLE_DEVICES=... 即可覆盖 setdefault。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from llama_cpp import Llama

# --- 1. 初始化模型 ---
# 模板适配：llama-cpp-python >= 0.3.20 会自动使用 GGUF 内嵌的官方 chat_template
# （Jinja2ChatFormatter），不再需要手写 GEMMA_4_JINJA。
# n_ctx 注意：131072 的 KV cache 约 32GB（f16），P40 24GB 装不下，实测 16384 稳妥。
llm = Llama(
    model_path="/mnt/data/ai_workspace/models/gemma-4-26B-A4B-it-UD-Q4_K_M/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=16384,
    flash_attn=True,
    verbose=False,
)

# --- 2. 协议转换层（引擎不做 Gemma DSL -> OpenAI JSON 的解析，这里补上） ---

def normalize_tool_args(args_raw: str) -> str:
    """将 DSL 格式 (key:value) 转化为标准 JSON"""
    clean = args_raw.replace('<|"|>', '"').replace('<|""|>', '"')
    # 匹配 key:value
    clean = re.sub(r'(?<!")(\b\w+\b)(?=\s*:)', r'"\1"', clean)
    clean = clean.strip()
    if not clean.startswith('{'): clean = "{" + clean + "}"
    clean = re.sub(r',\s*}', '}', clean)
    return clean


def parse_gemma_content(content: str) -> Tuple[str, List[Dict], str]:
    """把 Gemma 原生 DSL 文本拆成 (thought, tool_calls, clean_text)"""
    # 提取思考内容 <|channel>thought...<channel|>
    thought = ""
    thought_match = re.search(r"<\|channel>thought\n?(.*?)(?:\n?<channel\|>|$)", content, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    # 提取工具调用 <|tool_call>call:name{args}<tool_call|>
    tool_calls = []
    tool_pattern = r"<\|tool_call>call:([\w\-]+)\{(.*?)\}<tool_call\|>"
    for i, match in enumerate(re.finditer(tool_pattern, content, re.DOTALL)):
        method_name = match.group(1)
        args_raw = match.group(2)
        tool_calls.append({
            "id": f"call_{i}_{method_name}",
            "type": "function",
            "function": {
                "name": method_name,
                "arguments": normalize_tool_args(args_raw),
            },
        })

    # 清洗正文：去掉 thought / tool_call / 边界标签
    clean = re.sub(r"<\|channel>thought.*?(?:<channel\|>|$)", "", content, flags=re.DOTALL)
    clean = re.sub(r"<\|tool_call>.*?<tool_call\|>", "", clean, flags=re.DOTALL)
    clean = re.sub(r"<\|tool_response>.*?(?:<tool_response\|>|$)", "", clean, flags=re.DOTALL)
    for tag in ["<|turn>", "<turn|>", "<bos>", "<eos>", "<|channel>", "<channel|>", "<|tool_response>", "<tool_response|>"]:
        clean = clean.replace(tag, "")
    clean = clean.strip()

    # OpenAI 语义：有工具调用时正文应为空（tool_calls 与 content 互斥）
    if tool_calls:
        clean = ""

    return thought, tool_calls, clean


def to_openai_response(r: Dict[str, Any], thought: str, tool_calls: List[Dict], clean: str) -> Dict[str, Any]:
    """组装 OpenAI 格式响应"""
    message = {"role": "assistant", "content": clean if clean else None}
    if thought:
        message["reasoning_content"] = thought
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": r.get("id", "chatcmpl-gemma"),
        "object": "chat.completion",
        "created": r.get("created"),
        "model": r.get("model", "gemma-4-26b"),
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }],
        "usage": r.get("usage", {}),
    }


# --- 3. API 服务 ---

app = FastAPI(title="Gemma-4 OpenAI Converter")

_counter = itertools.count(1)

TOOL_PATTERN = re.compile(r"<\|tool_call>call:([\w\-]+)\{(.*?)\}<tool_call\|>", re.DOTALL)
THOUGHT_CLOSED = re.compile(r"<\|channel>thought\n?(.*?)\n?<channel\|>", re.DOTALL)


def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _make_tool_call(match: re.Match, idx: int) -> dict:
    method_name = match.group(1)
    return {
        "id": f"call_{idx}_{method_name}",
        "type": "function",
        "function": {
            "name": method_name,
            "arguments": normalize_tool_args(match.group(2)),
        },
    }


async def stream_generator(kwargs: Dict[str, Any]):
    """流式输出：缓冲跨 token 的 DSL，完整时转换；thought 转 reasoning_content；
    工具调用 DSL 一旦出现即暂停下发普通文本，防止标签泄漏，匹配后物理截断。"""
    buffer = ""
    tool_idx = 0
    for chunk in llm.create_chat_completion(**kwargs):
        choices = chunk.get("choices") or []
        if not choices:
            # usage 等尾包透传
            yield sse(chunk)
            continue
        token = (choices[0].get("delta") or {}).get("content") or ""
        buffer += token

        # 1) 完整工具调用 -> OpenAI tool_calls delta，物理截断
        m = TOOL_PATTERN.search(buffer)
        if m:
            tc = _make_tool_call(m, tool_idx)
            tool_idx += 1
            yield sse({"choices": [{"delta": {"content": None, "tool_calls": [tc]},
                                   "finish_reason": "tool_calls"}]})
            yield "data: [DONE]\n\n"
            return

        # 2) 完整 thought 块 -> reasoning_content delta
        tm = THOUGHT_CLOSED.search(buffer)
        if tm:
            t = tm.group(1).strip()
            if t:
                yield sse({"choices": [{"delta": {"reasoning_content": t},
                                       "finish_reason": None}]})
            buffer = buffer[tm.end():]

        # 3) 工具调用 DSL 进行中：暂停下发，等闭合匹配
        if "<|tool_call" in buffer:
            continue

        # 4) 下发安全前缀（暂缓尾部未闭合的标签）
        safe_len = len(buffer)
        li = buffer.rfind("<")
        if li != -1 and ">" not in buffer[li:]:
            safe_len = li
        safe = buffer[:safe_len]
        if safe:
            yield sse({"choices": [{"delta": {"content": safe},
                                   "finish_reason": None}]})
        buffer = buffer[safe_len:]

    # 收尾：剩余无标签文本 + stop
    clean = buffer.replace("<|channel>", "").replace("<channel|>", "").strip()
    if clean:
        yield sse({"choices": [{"delta": {"content": clean}, "finish_reason": None}]})
    yield sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    yield "data: [DONE]\n\n"

    # 收尾：剩余无标签文本 + stop
    clean = buffer.replace("<|channel>", "").replace("<channel|>", "").strip()
    if clean:
        yield sse({"choices": [{"delta": {"content": clean}, "finish_reason": None}]})
    yield sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_endpoint(request: Request):
    _req_id = next(_counter)
    _t0 = time.time()
    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools")  # 透传给模板渲染，让模型知道可用工具
    print(f"[req#{_req_id}] ENTER thr={threading.current_thread().name} "
          f"stream={bool(body.get('stream'))} max_tokens={body.get('max_tokens')}", flush=True)

    kwargs: Dict[str, Any] = dict(
        messages=messages,
        max_tokens=body.get("max_tokens", 4096),
        temperature=body.get("temperature", 0.7),
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = body.get("tool_choice", "auto")
        # 物理截断：模型输出 tool_call 后立即停止，防止它自导自演工具结果
        kwargs["stop"] = ["<|tool_response|>", "<tool_response|>"]

    if body.get("stream"):
        kwargs["stream"] = True
        return StreamingResponse(stream_generator(kwargs), media_type="text/event-stream")

    r = llm.create_chat_completion(**kwargs)
    print(f"[req#{_req_id}] DONE  thr={threading.current_thread().name} total={time.time()-_t0:.2f}s "
          f"p={r.get('usage',{}).get('prompt_tokens')} c={r.get('usage',{}).get('completion_tokens')}", flush=True)

    content = r["choices"][0]["message"].get("content") or ""
    thought, tool_calls, clean = parse_gemma_content(content)
    return to_openai_response(r, thought, tool_calls, clean)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10303)
