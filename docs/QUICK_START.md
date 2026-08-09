# CLand-LLM 快速开始

CLand-LLM 覆盖两条线：**推理服务**（本机常驻，开箱即用）与**模型训练**（LoRA 微调，见文末）。

## 一、推理服务（快速启动）

### 1. 一键启动

```bash
cd /mnt/data/ai_workspace/cland-llm
bash scripts/start_services.sh --wait    # 启动 4 个核心服务并等待健康
bash scripts/start_services.sh --with-sfx # 额外启动音效生成（内存紧张时慎用）
```

> ⚠️ 机器仅 15GB RAM：默认只起 4 个服务（SDXL/TripoSG/TTS/ASR），AudioGen 音效按需 `--with-sfx`。机器重启后 /tmp 会清空，Spark-TTS 代码已内嵌项目，无需重新 clone。

### 2. 服务总览

| 端口 | 服务 | 模型 | 环境 | GPU | 用法 |
|---|---|---|---|---|---|
| 10331 | 文生图 | SDXL base 1.0 | base | 0 | `POST /generate` 提示词→PNG |
| 10332 | 图生 3D | TripoSG 1.5B | triposg_env | 1 | `POST /generate` 图片→GLB |
| 10333 | 语音合成 | Spark-TTS 0.5B | audio_env | 1 | `POST /generate` 文本→WAV |
| 10334 | 语音识别 | SenseVoice 234M | audio_env | 1 | `POST /recognize` 音频→文本 |
| 10336 | 音效生成 | AudioGen 1.5B | audio_env | 1 | `POST /generate` 提示词→WAV（默认停） |

### 3. 调用速查

```bash
# 健康检查（任一）
curl http://127.0.0.1:10331/health

# 文生图（~71s）
curl -X POST http://127.0.0.1:10331/generate -H "Content-Type: application/json" \
  -d '{"prompt":"a cat astronaut on the moon, cinematic","steps":30,"seed":42}'

# 图生 3D（~26min，务必后台跑）
curl -X POST http://127.0.0.1:10332/generate -F "image=@img.png" -F "steps=50" -F "seed=42"

# 语音合成（中文）
curl -X POST http://127.0.0.1:10333/generate -F "text=你好世界" -F "gender=female"

# 语音识别（返回文本+情感标签）
curl -X POST http://127.0.0.1:10334/recognize -F "audio=@voice.wav"
```

### 4. 联动管线（图 → 3D → 语音）

```
SDXL 生图 (10331) → 参考图 → TripoSG 图生 3D (10332) → GLB
Spark-TTS (10333) → 角色配音
SenseVoice (10334) → 语音指令/字幕
```

## 二、环境速查

| 环境 | python | torch | 用途 |
|---|---|---|---|
| base | 3.13 | 2.7.1+cu118 | SDXL |
| triposg_env | 3.10 | 2.6.0+cu118 | TripoSG（diso 已编译） |
| audio_env | 3.10 | 2.6.0+cu118 | TTS/ASR/SFX（numpy 1.26.4） |

⚠️ P40 (sm_61) 铁律：torch 必须 cu118 构建（cu128+ 不支持）；cuDNN 9.x 不支持 Pascal，服务已内置 `cudnn.enabled=False` 降级。

## 三、文档导航

| 文档 | 内容 |
|---|---|
| `docs/sdxl_USAGE.md` | SDXL 完整使用手册 |
| `docs/3d/model_selection.md` | 3D 模型选型 + 部署记录 |
| `docs/speech/model_selection.md` | 语音模型选型（TTS/ASR） |
| `docs/audio/model_selection.md` | 音乐/音效选型 |
| `inference/triposg/README.md` | TripoSG 部署踩坑全记录 |
| `PITFAILLOG.md` | 全部踩坑记录（diso 编译/cuDNN/内存等） |
| `CHANGELOG.md` | 变更日志 |

## 四、训练路线（LoRA 微调）

> 以下为模型蒸馏/微调实操指南（原有内容保留）。

模型蒸馏的核心思路：让**教师模型**（大模型）针对你的问题生成高质量回答（含思维链），用这些数据**微调学生模型**（小模型）。

### 🔧 蒸馏实操流程

1. **明确目标与准备**
   - 确定场景（数学推理/代码生成/客服问答）
   - 选"师徒"：教师选 DeepSeek-R1、Qwen2-72B-Instruct 等；学生选 Llama-3.2-3B、Qwen2-7B
   - 准备种子数据（代表任务场景的问题和指令）

2. **核心步骤：生成"蒸馏数据"**
   - 设计提示词让教师模型展示推理过程（CoT）
   - 调用教师模型批量生成
   - 数据清洗（过滤错误/不完整回答）

3. **训练学生模型**
   - 用"指令-推理-回答"数据集对学生模型微调
   - 参考 `case001/` 的 GPU/DCU/CPU 训练脚本

### 📝 提示词与数据模板

#### 1. 数据生成提示词模板（向教师模型提问）

> **你是一位精通[填入你的领域，如：数学推理]的专家。请解决以下问题，并**务必**在给出最终答案前，用中文详细展示你的推理步骤（即一步步的思考过程）。**
>
> **问题：**
> {在这里插入你的问题}
>
> **请开始你的推理和解答：**

#### 2. 蒸馏数据集格式示例

（学生模型微调数据集 JSON 格式，见原文档及 `case001/` 说明）

## 五、故障速查

| 症状 | 处理 |
|---|---|
| 服务全部离线 | 机器可能重启过 → `bash scripts/start_services.sh --wait` |
| 内存不足（<1GB available） | 停 SFX：`kill $(pgrep -f "port 10336")`；必要时按需重启单个服务 |
| `no kernel image` | torch 版本非 cu118（gemma_env 的 cu128 不能用于这些服务） |
| `libcudnn.so.9` 缺失 | audio_env 装了 cuDNN 9.1 即可（P40 上 LSTM 需 `cudnn.enabled=False`） |
| TripoSG 请求 20min+ 无响应 | 正常，网格提取阶段 ~18min |
