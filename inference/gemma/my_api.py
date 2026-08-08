import json
import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from llama_cpp import Llama

# 1. 初始化底层（完全复用你跑通的参数）
llm = Llama(
    model_path="/mnt/data/ai_workspace/models/gemma-4-26B-it/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=131072,
    type_k=8,  # Q8_0 KV Cache
    type_v=8,
    flash_attn=True,
    logits_all=False,
    tensor_split=[1, 1],
)

app = FastAPI(title="Gemma-4-26B P40 Dual-Card API")


# 2. 核心逻辑：支持非流式和流式
@app.post("/v1/chat/completions")
async def chat_endpoint(request: Request):
    body = await request.json()
    messages = body.get("messages")
    stream = body.get("stream", False)

    # 如果是流式输出
    if stream:

        async def generator():
            # 调用 llama-cpp-python 的生成器
            output_iter = llm.create_chat_completion(messages=messages, stream=True)
            for chunk in output_iter:
                # 按照 OpenAI 标准格式包装数据
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generator(), media_type="text/event-stream")

    # 如果是非流式输出
    else:
        response = llm.create_chat_completion(messages=messages, stream=False)
        return response


if __name__ == "__main__":
    # 强制监听 IP，绕过主机名解析，解决之前的 DNS 错误
    uvicorn.run(app, host="0.0.0.0", port=10303, log_level="info")
