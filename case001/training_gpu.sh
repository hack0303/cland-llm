#!/bin/bash
set -e

# ===================== 第一步：申请DCU节点（满足所有集群规则） =====================
echo -e "\n===== 1. 申请hx2hdtest分区DCU节点 ====="
# 申请命令（唯一能成功的配置）
srun --partition=hx2hdtest \
     --gres=dcu:1 \
     --cpus-per-task=15 \
     --mem=45G \
     --mpi=none bash << 'INNER_SCRIPT'

# ===================== 第二步：在节点内配置环境+前置检查 =====================
echo -e "\n===== 2. 配置环境+前置检查 ====="
# 配置pip源+安装依赖
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install --user torch==2.3.0 transformers==4.40.0 datasets==2.19.0 peft==0.11.1 sentencepiece==0.1.99

# 检查模型路径（核心！先确认路径是否存在）
echo -e "\n===== 3. 检查模型路径 ====="
MODEL_DIR="/public/models/Llama3.1-8B-Instruct"
if [ -d "$MODEL_DIR" ]; then
    echo "✅ 模型路径存在：$MODEL_DIR"
    # 列出路径内的关键文件
    ls -l $MODEL_DIR | grep -E "config.json|tokenizer|pytorch_model" | head -5
else
    echo "⚠️  模型路径不存在！尝试常见的正确路径："
    # 尝试集群常见的模型路径格式
    MODEL_DIR="/public/models/Llama-3.1-8B-Instruct"  # 横线替代点
    if [ -d "$MODEL_DIR" ]; then
        echo "✅ 找到正确模型路径：$MODEL_DIR"
    else
        echo "❌ 未找到模型路径！请确认正确路径后修改脚本中的MODEL_DIR变量"
        exit 1
    fi
fi

# 验证DCU/GPU设备
echo -e "\n===== 4. 验证DCU/GPU设备 ====="
python3 -c "
import torch
print('=== DCU/GPU设备验证结果 ===')
print(f'Torch版本：{torch.__version__}')
print(f'CUDA设备数：{torch.cuda.device_count()}')
print(f'CUDA是否可用：{torch.cuda.is_available()}')
if torch.cuda.device_count() > 0:
    print(f'✅ 设备名称：{torch.cuda.get_device_name(0)}')
    print(f'✅ 设备显存：{torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.1f}GB')
else:
    print('⚠️  未检测到DCU/GPU设备，将使用CPU兜底')
"

# ===================== 第三步：编写并执行微调脚本 =====================
echo -e "\n===== 5. 执行DCU/GPU微调测试 ====="
cat > finetune_dcu_full_test.py << PYTHON_SCRIPT
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

# 核心配置：使用检查后的模型路径
MODEL_PATH = "$MODEL_DIR"
OUTPUT_DIR = "./llama3.1_dcu_full_test"

# 设备自动检测（DCU/GPU优先，CPU兜底）
if torch.cuda.device_count() > 0:
    DEVICE = torch.device("cuda:0")
    DEVICE_TYPE = "DCU/GPU"
    DTYPE = torch.float16
    BATCH_SIZE = 4
    MAX_LENGTH = 512
    R_VALUE = 8
    print(f'✅ 使用{DEVICE_TYPE}：{torch.cuda.get_device_name(0)}（显存：{torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.1f}GB）')
else:
    DEVICE = torch.device("cpu")
    DEVICE_TYPE = "CPU"
    DTYPE = torch.float32
    BATCH_SIZE = 1
    MAX_LENGTH = 128
    R_VALUE = 2
    print(f'⚠️  使用{DEVICE_TYPE}兜底（无DCU/GPU）')

# 训练数据（适配集群规则）
train_data = [
    {"instruction": "hx2hdtest分区申请DCU卡的正确命令", "response": "srun --partition=hx2hdtest --gres=dcu:1 --cpus-per-task=15 --mem=45G --mpi=none bash"},
    {"instruction": "DCU/GPU验证命令", "response": "python3 -c 'import torch; print(torch.cuda.device_count())'"},
    {"instruction": f"{DEVICE_TYPE}微调Llama3.1-8B的参数", "response": f"batch_size={BATCH_SIZE}，{DTYPE}精度，LoRA r={R_VALUE}，梯度检查点开启"}
]

# 加载Tokenizer（强制本地读取）
print("\\n===== 加载本地Tokenizer =====")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
    padding_side="right"
)
tokenizer.pad_token = tokenizer.eos_token

# 加载模型（适配设备）
print(f"\\n===== 加载模型到{DEVICE_TYPE} =====")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE_TYPE != "CPU" else "cpu",
    local_files_only=True,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
model.config.use_cache = False
if DEVICE_TYPE != "CPU":
    model = model.to(DEVICE)

# LoRA配置（适配设备）
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=R_VALUE,
    lora_alpha=R_VALUE*2,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    device=DEVICE
)
model = get_peft_model(model)
print("\\n可训练参数占比：")
model.print_trainable_parameters()

# 数据预处理
def format_data(example):
    text = f"User: {example['instruction']}\\nAssistant: {example['response']}"
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )
    if DEVICE_TYPE != "CPU":
        tokenized["input_ids"] = tokenized["input_ids"].to(DEVICE)
        tokenized["attention_mask"] = tokenized["attention_mask"].to(DEVICE)
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

train_ds = Dataset.from_list(train_data).map(format_data, batched=False)

# 训练参数（适配设备）
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    num_train_epochs=2,
    learning_rate=2e-4 if DEVICE_TYPE != "CPU" else 1e-4,
    logging_steps=1,
    save_strategy="no",
    fp16=(DEVICE_TYPE != "CPU"),
    report_to="none",
    no_cuda=(DEVICE_TYPE == "CPU"),
    gradient_checkpointing=(DEVICE_TYPE != "CPU"),
    dataloader_num_workers=0
)

# 开始训练
print(f"\\n===== 开始{DEVICE_TYPE}微调Llama3.1-8B =====")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
trainer.train()

# 保存+验证
model.save_pretrained(f"{OUTPUT_DIR}/final_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
print(f"\\n✅ 模型保存完成：{OUTPUT_DIR}/final_model")

# 推理验证
print(f"\\n===== {DEVICE_TYPE}推理验证 =====")
def infer(prompt):
    inputs = tokenizer(f"User: {prompt}\\nAssistant:", return_tensors="pt")
    if DEVICE_TYPE != "CPU":
        inputs = inputs.to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    ans = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
    print(f"问题：{prompt}\\n回答：{ans}\\n")

# 核心验证
infer("hx2hdtest分区申请DCU卡的正确命令是什么？")
infer(f"{DEVICE_TYPE}微调Llama3.1-8B的最优参数是什么？")
PYTHON_SCRIPT

# 执行微调脚本
python3 finetune_dcu_full_test.py

# ===================== 第四步：释放节点 =====================
echo -e "\n===== 6. 任务完成，释放节点 ====="
# 修复Shell转义错误：用正确的方式获取JOB_ID
JOB_ID=$(squeue -u $USER | grep -w "hx2hdtest" | awk '{print $1}')
if [ -n "$JOB_ID" ]; then
    scancel "$JOB_ID"
    echo "节点已释放，作业ID：$JOB_ID"
else
    echo "无待释放的节点"
fi

INNER_SCRIPT

# 脚本结束提示
echo -e "\n===== 一键式DCU/GPU微调测试完成！====="