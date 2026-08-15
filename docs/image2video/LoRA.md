---
title: "Lumo 角色 LoRA 训练技术文档"
summary: "SD1.5 角色 LoRA 完整技术文档：数据规范、训练配置（diffusers/kohya）、质量评估、本机执行步骤"
read_when:
  - "训练/复训 Lumo 或任何原创角色 LoRA"
  - "评估 LoRA 数据质量与训练效果"
scope:
  - cland-llm
status: "active"
updated: "2026-08-15"
---

# Lumo 角色 LoRA 训练技术文档

## 一、训练目标

- **模型**：`lumo_v1.safetensors`（SD1.5 系 LoRA，rank 32）
- **底座**：Counterfeit-V3.0_fp16（与出图/AnimateDiff 同底座）
- **触发词**：`1lumo`（加数字前缀减少与真实词冲突，社区惯例）
- **双用途**：出图（定妆/资产）+ 视频（AnimateDiff 注入后仍生效）

## 二、数据规范（关键：多样性 ≥ 数量）

### 2.1 最低合格标准

| 维度 | 要求 | 说明 |
|---|---|---|
| 数量 | **20~30 张** | <15 张过拟合风险高 |
| **角色一致性** | **全部锚定同一张定妆照生成（IPAdapter）** | ⚠️ 第一要求：纯文生图每张脸独立漂移，12 张 = 12 个角色，训练出"平均脸"必崩 |
| 姿态多样性 | ≥3 种姿态（站立/挥手/行走/持物…） | 全站姿 → 只会画站桩 |
| 表情 | ≥3 种（微笑/惊讶/悲伤） | 特写图提供面部细节 |
| 画风一致 | **全部同底座生成**（Counterfeit-V3） | 混底座训练会学出"四不像" |
| 视角 | 以正面为主 + 少量 3/4 侧 | 不必三视图（back 生成不可靠） |
| 构图 | 全身 70% + 特写 30% | 特写给脸部细节，全身给比例 |
| 背景 | 统一白底/纯色 | 背景干扰特征提取 |

> **数据生成必须锚定**：以定妆照（front.png）为 IPAdapter 参考，生成动作/表情图（weight ≥0.8）——保证"同一张脸 × 不同姿态"，而非"每张不同脸"。生成后人工筛选脸型一致的。

### 2.2 Caption 规范

```
每张图配同名 .txt，格式：
1lumo, <简短描述>
```

| 图 | caption 示例 |
|---|---|
| 正面定妆 | `1lumo, a tiny round glowing light fairy, standing, front view, full body` |
| 挥手 | `1lumo, waving one hand, cheerful pose` |
| 微笑特写 | `1lumo, smiling happily, close up, headshot` |

- **不要**写长提示词（防过拟合到 prompt 文本而非形象）
- 触发词 `1lumo` 每张必带（训练"这个词 = 这个角色"的映射）
- 描述用短词（standing/waving/happy），不含底座无关属性（画风词由底座负责）

### 2.3 数据清单（Lumo v1，锚定生成版）

| 组 | 数量 | 生成方式 | 状态 |
|---|---|---|---|
| 正面定妆（精选） | 3~4 | 从 801-812 人工精选脸型一致的（811 为基准） | ⚠️ 需人工筛选 |
| 动作（挥手/行走/持物/坐姿） | 8~10 | **811 定妆照 IPAdapter 锚定 + Counterfeit** | ⏳ 待生成 |
| 表情（微笑/惊讶/悲伤特写） | 6~8 | **同上（特写 prompt）** | ⏳ 待生成 |

合计 **18~22 张锚定一致数据** → 达标。数据目录：`inference/i2v/lora/dataset/`

> ⚠️ 801-812 纯文生图**不可直接入训**（脸型漂移），仅作定妆筛选源；训练主体 = 锚定生成的动作/表情数据。

## 三、训练配置

### 3.1 方案选择：diffusers 官方脚本（轻量）

```bash
# 依赖（base 环境已装 torch/diffusers/transformers；补装）
pip install accelerate peft datasets -i https://pypi.tuna.tsinghua.edu.cn/simple

# 脚本
diffusers/examples/text_to_image/train_text_to_image_lora.py
```

### 3.2 关键超参

| 参数 | 值 | 理由 |
|---|---|---|
| resolution | 512 | 底座训练分辨率 |
| rank (network_dim) | 32 | 角色 LoRA 通用值 |
| alpha (network_alpha) | 16 | alpha=rank/2 防过拟合 |
| learning_rate | 1e-4 | SD1.5 LoRA 标准 |
| batch_size | 2 | P40 显存 8-10GB |
| epochs | 10 | 小数据防过拟合 |
| lr_scheduler | constant | 简单稳定 |
| 触发词 | `1lumo` | caption 内嵌 |

### 3.3 显存/耗时（P40 预估）

- 训练峰值显存：~8-10GB（batch 2，512²）→ GPU1 可跑（与 ComfyUI 错峰）
- 耗时：25 张 × 10 epochs × ~2s/step ≈ **30-50 分钟**
- 内存：RAM ~4-6GB（15GB OK）

## 四、质量评估（训练后必做）

| 测试 | 方法 | 通过标准 |
|---|---|---|
| 触发响应 | 同 prompt：有/无 `1lumo` 各出 2 张 | 有触发词 = Lumo 形象；无 = 泛化形象 |
| 姿态泛化 | 新姿态 prompt（不存在的姿态，如坐姿） | 能画 Lumo 形象的新姿态（非站桩） |
| 稳定性 | 同一 prompt 5 个 seed | 形象一致（脸/斗篷/比例稳定） |
| 视频兼容 | AnimateDiff workflow 加 LoRA 节点跑 1 镜头 | 动画中角色稳定 |
| 无擦边 | 纯真约束负面词下抽查 | 无 NSFW 倾向 |

## 五、本机执行步骤

```bash
# 1. 数据生成（Counterfeit，ComfyUI 抽卡）——补齐动作/表情 ~10 张
python3 inference/i2v/lora/prepare_data.py   # （待写：调 ComfyUI 批量出图 + caption）

# 2. 训练（GPU1，与 ComfyUI 错峰）
accelerate launch train_lora.py \
  --pretrained_model_name_or_path Counterfeit-V3.0_fp16 路径 \
  --dataset_name inference/i2v/lora/dataset \
  --output_dir inference/i2v/lora/output \
  --resolution 512 --train_batch_size 2 --max_train_epochs 10 \
  --learning_rate 1e-4 --rank 32 --seed 42

# 3. 评估（对比抽卡）
# 4. 通过 → 产出 lumo_v1.safetensors → 接入 char_sheet prompt（desc → 1lumo）
```

## 六、风险与规避

| 风险 | 规避 |
|---|---|
| 数据少过拟合 | epochs ≤10 + alpha=16 + 数据多样性补齐 |
| 姿态过拟合（站桩） | 动作/表情数据占比 ≥40% |
| 底座混风格 | 全部 Counterfeit 同源生成 |
| 触发词冲突 | 用 `1lumo`（数字前缀） |
| 训练与 I2V 抢 GPU | 错峰：重跑完成后启动训练 |
| 效果不佳 | 回到锚定链路兜底（IPAdapter + 抽卡仍在） |

## 七、验收标准

- [ ] 触发词响应稳定（5 seed 形象一致）
- [ ] 姿态泛化（新姿态可画）
- [ ] 视频链路兼容（AnimateDiff + LoRA 节点）
- [ ] 出图阶段不再需要长 desc（`1lumo` 即出）
