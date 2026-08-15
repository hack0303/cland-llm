---
title: "Lumo 角色 LoRA 训练方案"
summary: "用 LoRA 固化 Lumo 角色形象（角色一致性治本方案）：数据准备、训练配置、与现有锚定链路的整合"
read_when:
  - "训练 Lumo 或其他原创角色的 LoRA"
  - "角色一致性需要治本（摆脱抽卡+提示词碰运气）"
  - "评估本机（P40）LoRA 训练可行性"
scope:
  - cland-llm
status: "planning"
updated: "2026-08-15"
---

# Lumo 角色 LoRA 训练方案

## 一、为什么需要 LoRA（治本）

当前角色一致性靠**三层软约束**：desc 提示词 + 锚定图（IPAdapter）+ 首帧 img2img。问题是：

| 现状 | 痛点 |
|---|---|
| desc 提示词 | 每次抽卡碰运气（脸/脚/擦边都要人工筛） |
| IPAdapter 锚定 | 只锁"视频阶段"，SDXL 出图阶段仍漂移 |
| 抽卡 | 5 张挑 1 张，废图率高 |

**LoRA 后**：`1lumo` 一个触发词 → 任何底座/镜头稳定生成 Lumo 形象——抽卡成本消失，出图阶段就锁死。

## 二、可行性（P40 + 15GB RAM）

| 项 | 评估 |
|---|---|
| 底座 | SD1.5（Counterfeit-V3.0_fp16）——LoRA 训练生态最成熟 |
| 显存 | SD1.5 LoRA 训练（512²，batch 2）≈ 8-10GB ✅（GPU1 余量） |
| 数据量 | 10~15 张定妆/动作/表情图（已有候选资产可复用） |
| 训练时长 | 512² × 15 图 × 10 epochs ≈ **30-60 分钟**（P40） |
| 工具 | kohya sd-scripts 或 diffusers LoRA（本机 base 环境可装） |

## 三、训练数据准备

```
outputs_character/lumo/lora_dataset/
├── 801.png  → 提示词: 1lumo, full body, front view, standing, plain background
├── 802.png  → 同上（多 seed 多样本）
├── happy.png  → 1lumo, close up, smiling
├── walking.png → 1lumo, full body, walking
└── ...
```

- 来源：candidates_cf（801-805）+ Counterfeit 重新生成动作/表情资产（统一底座，保证画风一致）
- 关键：**训练集与目标底座同为 Counterfeit-V3**（同源训练质量最高）
- 配文（caption）：`1lumo` 触发词 + 简单描述（不用长提示词，防过拟合到提示词）

## 四、训练配置（kohya sd-scripts 参考）

```bash
# 参数要点（SD1.5 LoRA，512²）
--network_dim 32           # LoRA 秩（32 够用）
--network_alpha 16
--learning_rate 1e-4
--train_batch_size 2
--resolution 512
--max_train_epochs 10
--trigger_word "1lumo"
--output_name lumo_v1
```

## 五、整合到现有管线

```
训练后：
char_sheet.py / 分镜 prompt 里 desc 前加 "1lumo, "（或替换角色短语）
    ↓
Counterfeit-V3 + 1lumo → 出图稳定 Lumo 形象（不再抽卡）
    ↓
锚定图（IPAdapter）+ 首帧（img2img）仍是双保险（LoRA 为主，锚定兜底）
```

- 角色短语从"desc 描述"升级为"1lumo 触发词"——**desc 简化为 LoRA 不覆盖的细节**（斗篷颜色/道具变化）
- 与 IPAdapter 并存：LoRA 锁"长相"，锚定锁"镜头内"，互不冲突

## 六、执行计划

- [ ] 1. Counterfeit 重生成训练集（定妆 5 + 动作 3 + 表情 3 ≈ 11 张，~2 分钟）
- [ ] 2. 配 caption（1lumo 触发词 + 短描述）
- [ ] 3. 安装 kohya sd-scripts（或 diffusers 简易脚本）到独立 venv
- [ ] 4. 训练（30-60 分钟，GPU1 与 ComfyUI 错开）
- [ ] 5. 验证：同 prompt 有/无 1lumo 对比抽卡
- [ ] 6. 通过后：更新 char_sheet 提示词模板（desc → 1lumo 触发）

## 七、风险

- P40 训练慢 → 用 512² + batch 2 + 少 epoch，先出 v1 验证
- 数据量少（11 张）→ 过拟合风险：epoch 控制在 10 以内，加正则化（网络 alpha 16）
- 与 Counterfeit 同源训练 → 换底座（如 SDXL）时 LoRA 不通用（可接受，当前底座已定版）
