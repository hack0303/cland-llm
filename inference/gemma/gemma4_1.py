import json
import os
import uvicorn
import re
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from llama_cpp import Llama

# ---------------------------------------------------------
# 1. 初始化配置
# ---------------------------------------------------------

MODEL_PATH = "/mnt/data/ai_workspace/models/gemma-4-31b-it-IQ2_M-GGUF/gemma4-31b-IQ2_M.gguf"

# 初始化 Llama
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=8192,
    type_k=8,
    type_v=8,
    flash_attn=True,
    tensor_split=[1, 1],
    verbose=True
)

app = FastAPI(title="Gemma-4-26B P40 DSL-Parsing API")

# ---------------------------------------------------------
# 2. Gemma-4 专属 DSL 解析器 (增强鲁棒性)
# ---------------------------------------------------------

def parse_gemma4_dsl(text):
    """
    专门处理 Gemma-4 的特殊语法，兼容缺少闭合标签的情况: 
    <|tool_call>call:func{key:<|"|>value<|"|>}(<tool_call|>)?
    """
    if not text:
        return None

    # 1. 匹配起始标签和内容。注意：使末尾的 <tool_call|> 变为可选 (?)
    # 增加对 call: 前缀的匹配
    pattern = r"<\|tool_call>call:(\w+)\{(.*?)\}(?:<tool_call\|>)?"
    matches = list(re.finditer(pattern, text, re.DOTALL))
    
    if not matches:
        return None

    parsed_calls = []
    for match in matches:
        func_name = match.group(1)
        args_raw = match.group(2)

        # 2. 深度清洗引号 Token
        # 兼容多种可能的转义情况: <|"|>, <|\">, <|\\"> 等
        clean_args = args_raw
        clean_args = re.sub(r'<\|\\?"|\\?"\|>', '"', clean_args)
        
        # 3. DSL 转 JSON
        try:
            # 补齐 key 的引号。Gemma-4 格式通常是 key:value
            # 使用正则寻找未被引号包裹的 key (单词后跟冒号)
            json_ready = "{" + re.sub(r'(\w+):', r'"\1":', clean_args) + "}"
            
            # 尝试解析
            arguments = json.loads(json_ready)
            
            parsed_calls.append({
                "id": f"call_{os.urandom(4).hex()}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False)
                }
            })
        except Exception as e:
            # 如果解析失败，尝试最后一次保底：处理简单的 key: "value" 字符串
            print(f"Standard JSON parse failed, trying backup: {e}")
            try:
                # 最后的保底逻辑，手动提取简单的 kv 对
                kv_pairs = re.findall(r'(\w+):"([^"]+)"', clean_args)
                if kv_pairs:
                    backup_args = {k: v for k, v in kv_pairs}
                    parsed_calls.append({
                        "id": f"call_{os.urandom(4).hex()}",
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": json.dumps(backup_args, ensure_ascii=False)
                        }
                    })
            except:
                continue

    return parsed_calls if parsed_calls else None

# ---------------------------------------------------------
# 3. 接口逻辑
# ---------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_endpoint(request: Request):
    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    messages = body.get("messages", [])
    tools = body.get("tools", None)
    tool_choice = body.get("tool_choice", "auto")
    stream = body.get("stream", False)
    
    temperature = body.get("temperature", 0.1)
    max_tokens = body.get("max_tokens", 4096)

    # 停止词列表
    stop = ["<|im_end|>", "<end_of_turn>", "<|tool_response>", "<start_of_turn>"]

    try:
        if stream:
            async def generator():
                output_iter = llm.create_chat_completion(
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop
                )
                for chunk in output_iter:
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(generator(), media_type="text/event-stream")

        else:
            response = llm.create_chat_completion(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop
            )
            
            choice = response["choices"][0]
            content = choice["message"].get("content", "")
            
            # 执行增强版 DSL 解析
            tool_calls = parse_gemma4_dsl(content)
            
            if tool_calls:
                choice["message"]["tool_calls"] = tool_calls
                # 如果只有工具调用，则清空 content 字段（符合 OpenAI 规范）
                # 如果包含文本和工具调用，则只保留文本
                clean_content = re.sub(r"<\|tool_call>.*?(?:<tool_call\|>|$)", "", content, flags=re.DOTALL).strip()
                choice["message"]["content"] = clean_content if clean_content else None
                choice["finish_reason"] = "tool_calls"
            
            return response

    except Exception as e:
        print(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Gemma-4 P40 API is starting...")
    uvicorn.run(app, host="0.0.0.0", port=10303)
