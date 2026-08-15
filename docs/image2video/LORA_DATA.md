---
title: "Lumo LoRA 训练数据生成文档"
summary: "IPAdapter 锚定生成 LoRA 训练数据：ComfyUI 工作流设计、数据清单、caption 规范、筛选流程"
read_when:
  - "生成/补充 LoRA 训练数据"
  - "运行 prepare_data.py 或手改锚定出图工作流"
  - "筛选数据质量"
scope:
  - cland-llm
status: "active"
updated: "2026-08-15"
---

# Lumo LoRA 训练数据生成文档

> 配套：`LoRA.md`（训练技术文档）· 原则：**数据第一要求是角色一致性**（同一张脸 × 不同姿态），靠 IPAdapter 锚定实现

## 一、目标与原则

- **产出**：20± 张"同一个 Lumo"数据集 → `inference/i2v/lora/dataset/`
- **一致性来源**：811 定妆照（front.png）作为 IPAdapter 锚定参考，所有动作/表情图生成时锚定它
- **底座**：Counterfeit-V3.0_fp16（与训练/出图/AnimateDiff 全链路一致）

```
811 定妆照（脸型唯一基准）
    │ IPAdapter weight 0.85（锚定强）
    ▼
Counterfeit + 动作/表情 prompt → 数据图（脸统一到 811）
```

## 二、生成工作流（ComfyUI API，prepare_data.py 封装）

节点链（与角色锚定 I2V 工作流同源，去掉视频部分）：

```
[CheckpointLoader: Counterfeit-V3.0_fp16] ──model──▶ [IPAdapterUnifiedLoader(SD1.5)]
        │CLIP ──▶ [CLIPTextEncode 动作/表情prompt]         │
        │VAE  ──▶ [VAEEncode? 不用]                        ▼
[LoadImage: 811定妆照] ─────────────────▶ [IPAdapterAdvanced(weight 0.85)]
                                                    │
[EmptyLatentImage 512²] ──────────────────────────▶ [KSampler 25步] ──▶ [VAEDecode] ──▶ [SaveImage]
```

| 节点 | 关键参数 |
|---|---|
| CheckpointLoaderSimple | `Counterfeit-V3.0_fp16.safetensors` |
| LoadImage | `lumo_811.png`（定妆照，锚定参考） |
| IPAdapterAdvanced | `weight=0.85`（锚定强，防脸漂移）、`start_at=0, end_at=1` |
| KSampler | 25 步 / euler / cfg 7 / seed 递增 |
| 负面词 | 全约束（NSFW/底座/侧身/画质，与出图一致） |

## 三、数据清单（每项 2~3 张，多 seed）

### 3.1 动作（全身，8~10 张）

| 姿态 | prompt 片段 | 说明 |
|---|---|---|
| 站立（补充视角） | `standing, three-quarter view` | 少量 3/4 侧增加视角多样性 |
| 挥手 | `waving one hand, cheerful pose` | 手部姿态样本 |
| 行走 | `walking forward, gentle stride` | 动态姿势 |
| 持物 | `holding a lantern in both hands` | 与角色道具互动 |
| 坐姿 | `sitting down, cozy pose` | 大姿态差异 |

### 3.2 表情（特写，6~8 张）

| 表情 | prompt 片段 |
|---|---|
| 微笑 | `smiling happily, close up, headshot` |
| 惊讶 | `surprised, eyes wide open, close up` |
| 悲伤 | `sad, teary eyes, close up` |
| 害羞 | `shy, blushing, close up` |

### 3.3 定妆精选（3~4 张）

从 candidates_cf 801-812 **人工挑选脸型最贴近 811 的**（不可直接全用——纯文生图脸型漂移）

## 四、Caption 规范（训练用）

每张图同名 `.txt`：

```
1lumo, waving one hand, cheerful pose
1lumo, smiling happily, close up, headshot
```

- 触发词 `1lumo` 必带
- 短描述（动作/表情/景别），不写画风（底座负责）

## 五、筛选流程（生成后必做）

| 步骤 | 方法 | 淘汰标准 |
|---|---|---|
| 1. 自动预筛 | 定妆照 vs 数据图脸区相似度（SSIM/特征） | 相似度过低淘汰 |
| 2. **人工终审** | 看图对比定妆照 | 脸型/五官不像、姿态崩、擦边 → 淘汰 |
| 3. 构图检查 | 全身图主体完整 | 截肢/主体缺失淘汰 |
| 4. 计数 | ≥18 张合格 | 不足补抽 |

## 六、工具接口（prepare_data.py）

```bash
python3 inference/i2v/lora/prepare_data.py \
  --character outputs_character/lumo/character.json   # 取 811 front.png + desc
  --out inference/i2v/lora/dataset \
  --actions WAVING WALKING HOLDING SITTING \
  --expressions HAPPY SURPRISED SAD SHY \
  --per-item 2          # 每姿态/表情抽 2 张
  --seed 9000           # 种子起点
```

输出：
```
dataset/
├── lumo_001.png / lumo_001.txt   # 1lumo, waving one hand...
├── lumo_002.png / lumo_002.txt
└── ...
```

## 七、质量验收

- [ ] ≥18 张合格数据（人工终审过）
- [ ] 脸型统一（目测像 811）
- [ ] 姿态/表情多样性 ≥3 类
- [ ] 全部 Counterfeit 同源 + 白底
- [ ] caption 全部带 `1lumo` 触发词

## 八、变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| 1.0 | 2026-08-15 | 初版：锚定生成方案 + 工作流 + 清单 + 筛选流程 |
