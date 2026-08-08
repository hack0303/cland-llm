#!/bin/bash
set -e

# 核心配置（满足QOS+CPU兜底）
PARTITION="hx2hdtest"
GRES_TYPE="dcu"
GRES_NUM=1
CPU_CORES=15
MEMORY=15G

# 申请节点
echo -e "\n===== 1. 申请${PARTITION}分区节点 ====="
srun --partition=${PARTITION} \
     --gres=${GRES_TYPE}:${GRES_NUM} \
     --cpus-per-task=${CPU_CORES} \
     --mem=${MEMORY} \
     --mpi=none bash << 'EOF'

# 配置环境
echo -e "\n===== 2. 配置微调环境 ====="
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install --user torch==2.3.0 transformers==4.40.0 datasets==2.19.0 peft==0.11.1 accelerate==0.30.1 sentencepiece==0.1.99

# 验证环境
python3 -c "import torch; print(f'Torch版本：{torch.__version__}'); print(f'设备数：{torch.cuda.device_count()}'); print(f'使用CPU训练兜底')"

# 生成CPU兜底微调脚本（修复路径+纯CPU运行）
cat > finetune_cpu_final.py << 'PYTHON_EOF'
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

# 核心配置（CPU兜底+路径修复）
MODEL_PATH = "/public/models/Llama3.1-8B-Instruct"
OUTPUT_DIR = "./llama3.1_cpu_final"
# 强制使用CPU（设备数为0时兜底）
DEVICE = torch.device("cpu")
print(f"当前使用设备：{DEVICE}（DCU卡未挂载，CPU兜底）")

# 训练数据
train_data = [
    {"instruction": "hx2hdtest分区申请DCU卡指令", "response": "srun --partition=hx2hdtest --gres=dcu:1 --cpus-per-task=15 --mem=15G --mpi=none bash"},
    {"instruction": "释放hx2hdtest节点方法", "response": "scancel 作业ID（squeue -u $USER查看）"}
]

# 加载Tokenizer（关键：补全local_files_only=True + trust_remote_code=True）
print("\n===== 加载本地Tokenizer（修复路径问题） =====")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,  # 强制读本地文件，核心修复！
    trust_remote_code=True, # 适配Llama3.1
    padding_side="right"
)
tokenizer.pad_token = tokenizer.eos_token

# 加载模型（CPU模式+低内存配置）
print("\n===== 加载模型到CPU（低内存模式） =====")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,  # CPU用float32更稳定
    device_map="cpu",           # 强制CPU
    local_files_only=True,     # 核心修复！
    trust_remote_code=True,
    low_cpu_mem_usage=True     # 低内存模式，适配15G内存
)
model.config.use_cache = False

# LoRA极致轻量化（CPU专用）
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=1,                # 最低维度，CPU能跑
    lora_alpha=2,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    device=DEVICE
)
model = get_peft_model(model, lora_config)
print("\n===== 可训练参数 =====")
model.print_trainable_parameters()

# 数据预处理（CPU模式）
def format_data(example):
    text = f"User: {example['instruction']}\nAssistant: {example['response']}"
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=128,  # 最短长度，省内存
        padding="max_length",
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

train_ds = Dataset.from_list(train_data).map(format_data, batched=False)

# 训练参数（CPU专用，最低资源）
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=1e-4,
    logging_steps=1,
    save_strategy="no",
    fp16=False,  # CPU关闭fp16
    report_to="none",
    dataloader_pin_memory=False,
    no_cuda=True, # 强制关闭CUDA，用纯CPU
    gradient_checkpointing=False, # CPU关闭梯度检查点
    dataloader_num_workers=0
)

# 开始训练
print("\n===== 开始CPU微调Llama3.1-8B =====")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
trainer.train()

# 保存+验证
model.save_pretrained(f"{OUTPUT_DIR}/cpu_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/cpu_model")
print(f"\n===== 模型保存完成：{OUTPUT_DIR}/cpu_model =====")

# 验证效果
def infer(prompt):
    inputs = tokenizer(f"User: {prompt}\nAssistant:", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.6, pad_token_id=tokenizer.eos_token_id)
    ans = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
    print(f"\n问题：{prompt}\n回答：{ans}")

infer("hx2hdtest分区怎么申请DCU卡？")
infer("怎么释放hx2hdtest节点？")
PYTHON_EOF

# 执行CPU微调（兜底方案）
echo -e "\n===== 3. 执行CPU兜底微调（DCU卡未挂载） ====="
python3 finetune_cpu_final.py

# 释放节点
echo -e "\n===== 4. 释放节点 ====="
JOB_ID=$(squeue -u $USER | grep "${PARTITION}" | awk '{print $1}')
if [ -n "${JOB_ID}" ]; then
    scancel "${JOB_ID}"
    echo "节点已释放，作业ID：${JOB_ID}"
else
    echo "未找到作业ID"
fi

EOF

echo -e "\n===== CPU微调流程完成！====="