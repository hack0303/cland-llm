import json
import re
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from llama_cpp import Llama

# 初始化：P40 双卡，80K 上下文
llm = Llama(
    model_path="/mnt/data/ai_workspace/models/gemma-4-26B-it/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=81920,
    chat_format=None, # 二开必须禁用内置模板，手动控制
    verbose=False
)

app = FastAPI()

def parse_gemma_to_openai_tool(full_text):
    """
    协议转换核心：将 <|tool_call|>call:file:write{...} 映射为 OpenAI JSON
    """
    # 匹配原生标签：支持冒号命名空间
    pattern = r"<\|tool_call\|>call:([\w:]+)\{(.*?)\}"
    match = re.search(pattern, full_text, re.S)
    
    if not match:
        return None

    raw_func = match.group(1)
    # 命名空间映射：例如 file:write -> file_write
    func_name = raw_func.replace(":", "_")
    
    args_raw = match.group(2)
    # 鲁棒解析参数对 (key:val)
    args_dict = {}
    pairs = re.findall(r"(\w+)\s*:\s*(\".*?\"|'.*?'|[^,}]+)", args_raw, re.S)
    for k, v in pairs:
        args_dict[k.strip()] = v.strip().strip('"').strip("'").replace('\\n', '\n')

    return {
        "id": f"call_{func_name}",
        "type": "function",
        "function": {
            "name": func_name,
            "arguments": json.dumps(args_dict)
        }
    }

@app.post("/v1/chat/completions")
async def chat_endpoint(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    
    # 按照官方 Practice 拼接 Prompt
    prompt = ""
    for m in messages:
        role = "model" if m["role"] == "assistant" else m["role"]
        prompt += f"<|turn>{role}\n{m['content']}<turn|>\n"
    prompt += "<|turn>model\n"

    # 官方采样参数：1.0 / 0.95 / 64
    gen_config = {
        "prompt": prompt,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "stream": True,
        # 核心：遇到工具结束符立即停止，防止模型“自导自演”
        "stop": ["<|tool_response|>", "<tool_call|>", "model"]
    }

    async def generator():
        full_buffer = ""
        print("\n" + "="*20 + " [TOOL LOGIC ACTIVE] " + "="*20)
        
        output_iter = llm.create_completion(**gen_config)
        
        for chunk in output_iter:
            token = chunk["choices"][0].get("text", "")
            full_buffer += token
            
            # 实时打印原始流 (绿色)
            print(f"\033[92m{token}\033[0m", end="", flush=True)

            # --- 拦截逻辑开始 ---
            if "<|tool_call|>" in full_buffer:
                tool_call = parse_gemma_to_openai_tool(full_buffer)
                
                if tool_call:
                    # 转换协议并下发 finish_reason: tool_calls
                    openai_tool_chunk = {
                        "choices": [{
                            "delta": {
                                "content": None,
                                "tool_calls": [tool_call]
                            },
                            "finish_reason": "tool_calls"
                        }]
                    }
                    yield f"data: {json.dumps(openai_tool_chunk)}\n\n"
                    
                    print(f"\n\033[93m[SUCCESS] 拦截到工具调用: {tool_call['function']['name']}\033[0m")
                    break # 物理截断：交由前端执行物理 IO
            # --- 拦截逻辑结束 ---

            # 正常文本下发
            yield f"data: {json.dumps({'choices': [{'delta': {'content': token}, 'finish_reason': None}]})}\n\n"
        
        yield "data: [DONE]\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10303)
