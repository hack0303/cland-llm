import json
import re
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request
from jinja2 import Template
from llama_cpp import Llama

# --- 1. 初始化模型 ---
llm = Llama(
    model_path="/mnt/data/ai_workspace/models/gemma-4-26B-it/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
    n_gpu_layers=-1, 
    n_ctx=131072,
    flash_attn=True,
)

# --- 2. 增强型多轮对话 Jinja2 模板 ---
# 增加了对 role: "tool" 的支持，这是多轮工具调用的核心
GEMMA_4_JINJA = """
{%- set ns = namespace(prev_message_type=None) -%}
{{- bos_token -}}

{# 处理系统提示词 #}
{%- if messages[0]['role'] in ['system', 'developer'] -%}
    {{- '<|turn>system\n' -}}
    {%- if enable_thinking -%}{{- '<|think|>\n' -}}{%- endif -%}
    {{- messages[0]['content'] | trim -}}
    {{- '<turn|>\n' -}}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set loop_messages = messages -%}
{%- endif -%}

{# 循环处理历史消息，实现多轮记忆 #}
{%- for message in loop_messages -%}
    {%- if message['role'] == 'tool' -%}
        {# 核心更新：处理工具执行后的返回结果 #}
        {# 格式: <|tool_response>response:函数名{结果内容}<tool_response|> #}
        {{- '<|tool_response>response:' + (message['name'] if message['name'] else 'unknown') + '{' + message['content'] + '}<tool_response|>\n' -}}
    {%- else -%}
        {%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}
        {{- '<|turn>' + role + '\n' -}}
        
        {# 如果历史消息中有思考过程，也需要还原 #}
        {%- if message['reasoning_content'] -%}
            {{- '<|channel>thought\n' + message['reasoning_content'] + '\n<channel|>' -}}
        {%- endif -%}
        
        {# 消息正文 #}
        {%- if message['content'] -%}
            {{- message['content'] | trim -}}
        {%- endif -%}
        
        {# 如果历史消息中有工具调用请求 #}
        {%- if message['tool_calls'] -%}
            {%- for tc in message['tool_calls'] -%}
                {{- '<|tool_call>call:' + tc['function']['name'] + '{' + tc['function']['arguments'] + '}<tool_call|>' -}}
            {%- endfor -%}
        {%- endif -%}
        
        {{- '<turn|>\n' -}}
    {%- endif -%}
{%- endfor -%}

{# 提示模型开始生成当前轮次的回复 #}
{%- if add_generation_prompt -%}
    {{- '<|turn>model\n' -}}
    {{- '<|channel>thought\n' -}}
{%- endif -%}
"""
chat_template = Template(GEMMA_4_JINJA)

# --- 3. 协议转化函数 ---

def normalize_tool_args(args_raw: str) -> str:
    """将 DSL 格式 (key:value) 转化为标准 JSON"""
    clean = args_raw.replace('<|"|>', '"').replace('<|""|>', '"')
    # 匹配 key:value
    clean = re.sub(r'(?<!")(\b\w+\b)(?=\s*:)', r'"\1"', clean)
    clean = clean.strip()
    if not clean.startswith('{'): clean = "{" + clean + "}"
    clean = re.sub(r',\s*}', '}', clean)
    return clean

def parse_gemma_response_to_openai(llm_response: Dict[str, Any]) -> Dict[str, Any]:
    """解析单次推理结果并映射到 OpenAI 格式"""
    raw_text = llm_response["choices"][0]["text"]
    
    # 提取思考内容
    thought = ""
    thought_match = re.search(r"<\|channel>thought\n?(.*?)(?:\n?<channel\|>|$)", raw_text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    # 提取工具调用
    tool_calls = []
    tool_pattern = r"<\|tool_call>call:([\w\-]+)\{(.*?)\}<tool_call\|>"
    for i, match in enumerate(re.finditer(tool_pattern, raw_text, re.DOTALL)):
        method_name = match.group(1)
        args_raw = match.group(2)
        tool_calls.append({
            "id": f"call_{i}_{method_name}",
            "type": "function",
            "function": {
                "name": method_name, 
                "arguments": normalize_tool_args(args_raw)
            }
        })

    # 清洗正文
    content = re.sub(r"<\|channel>thought.*?(?:<channel\|>|$)", "", raw_text, flags=re.DOTALL)
    content = re.sub(r"<\|tool_call>.*?<tool_call\|>", "", content, flags=re.DOTALL)
    for tag in ["<|turn>", "<turn|>", "<bos>", "<eos>", "<|channel>", "<channel|>"]:
        content = content.replace(tag, "")
    
    content = content.strip()

    message = {"role": "assistant", "content": content if content else None}
    if thought: message["reasoning_content"] = thought
    if tool_calls: message["tool_calls"] = tool_calls

    return {
        "id": llm_response.get("id", "chatcmpl-gemma-multi"),
        "object": "chat.completion",
        "created": llm_response.get("created"),
        "model": llm_response.get("model", "gemma-4-26b"),
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else "stop"
        }],
        "usage": llm_response.get("usage", {})
    }

# --- 4. API 服务 ---

app = FastAPI(title="Gemma-4 Multi-turn Converter")

@app.post("/v1/chat/completions")
async def chat_endpoint(request: Request):
    body = await request.json()
    
    # 这里的 messages 数组包含了多轮历史，Jinja 模板会遍历它
    prompt = chat_template.render(
        messages=body.get("messages", []),
        enable_thinking=body.get("enable_thinking", True),
        add_generation_prompt=True,
        bos_token="<bos>"
    )

    response = llm.create_completion(
        prompt=prompt,
        max_tokens=body.get("max_tokens", 4096),
        temperature=body.get("temperature", 0.7),
        stop=["<turn|>", "<bos>", "<|turn>"],
        stream=False 
    )

    return parse_gemma_response_to_openai(response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10303)
