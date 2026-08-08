import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_path = "/mnt/data/ai_workspace/models/gemma-4-31B-it"

# 使用 bitsandbytes 模拟极低位宽效果
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16, # P40 必备
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,       # 二次量化，显存占用进一步降低
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
