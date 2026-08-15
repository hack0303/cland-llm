# 角色设定工作流（Character Sheet Pipeline）

> 版本：1.0（2026-08-15）· 工具：`inference/i2v/char_sheet.py`
> 原则：**三视图是"角色档案"，不是视频素材**；视频生产用单角色单姿态
> 对接：run_story.py `--character`（IPAdapter 角色锚定）

## 一、三阶段流程（动画公司标准流程）

```
Step 1 角色母版           Step 2 动作资产           Step 3 视频生产
┌────────────────┐   ┌───────────────────┐   ┌──────────────────────┐
│ character_sheet │   │ front.png 正面定妆  │   │ 每个镜头：             │
│ _3view.png      │──▶│ 站立/坐姿/奔跑/表情  │──▶│ 锚定图(front.png)     │
│ 三视图档案       │   │ → RMBG 抠图透明 PNG │   │ + 首帧 + 动作描述      │
│ （IP 身份证）    │   │ poses/*_alpha.png  │   │ → I2V → 动画片段       │
└────────────────┘   └───────────────────┘   └──────────────────────┘
```

| 阶段 | 产物 | 用途 | 说明 |
|---|---|---|---|
| Step 1 | `character_sheet_3view.png` | **档案** | 正/侧/背视图，看比例/服装结构/装饰位置；白底保留 |
| Step 2 | `front.png` + `poses/*.png` | **生产素材** | 白底全身图（I2V 锚定/首帧） |
| Step 2+ | `*_alpha.png` | **合成储备** | RMBG-1.4 抠图透明 PNG（后期贴片/合成用） |
| Step 3 | `character.json` | 元数据 | 角色名/描述/画风/资产索引，run_story `--character` 直接吃 |

## 二、使用

```bash
# 完整角色设定（三视图 + 正面 + 4 姿态 + 全部抠图）
python3 inference/i2v/char_sheet.py --name lumo \
    --desc "a cute firefly fairy with a glowing chest core and purple cloak" \
    --style "cute cartoon, soft lighting" \
    --poses 站立 坐姿 奔跑 微笑

# 只出档案（三视图 + 正面，不抠图）
python3 inference/i2v/char_sheet.py --name lumo --desc "..." --no-alpha

# 视频流水线接线
python3 inference/i2v/run_story.py --sb outputs_video/story001/storyboard.json \
    --character outputs_character/lumo/character.json
```

### 姿态表（--poses 可选值，v1.1 扩至 14 个动作）

| 中文 | 英文提示词 | | 中文 | 英文提示词 |
|---|---|---|---|---|
| 待机 | idle standing pose, relaxed | | 施法 | casting pose, hands glowing with magic |
| 站立 | standing pose, facing forward | | 坐下 | sitting pose |
| 行走 | walking pose | | 躺下 | lying down pose |
| 奔跑 | running pose, dynamic | | 胜利 | victory pose, arms raised cheering |
| 跳跃 | jumping pose, mid-air | | 失败 | defeated pose, sitting tired |
| 攻击 | attacking pose, weapon swing | | 挥手 | waving one hand, friendly |
| 受击 | hit pose, knocked back | | 思考 | thinking pose, hand on chin |

### 表情表（--expressions 可选值，v1.1 新增，脸部特写）

| 中文 | 英文提示词 | | 中文 | 英文提示词 |
|---|---|---|---|---|
| 微笑 | smiling, happy | | 生气 | angry, frowning |
| 惊讶 | surprised, eyes wide open | | 悲伤 | sad, crying slightly |
| 害羞 | shy, blushing | | 自信 | confident, determined |

### 三视图（--no-3view 跳过；v1.1 改单视图拼版）

front / side / back 各生成一张单视图 → `views/` → PIL 拼版 `character_sheet_3view.png`

## 三、输出目录

```
outputs_character/{name}/
├── character.json              # 元数据（desc/style/资产索引）
├── character_sheet_3view.png   # Step1 三视图档案（白底）
├── front.png                   # Step2 正面定妆（白底）← I2V 锚定图
├── front_alpha.png             # 透明 PNG（合成储备）
└── poses/
    ├── 站立.png / 站立_alpha.png
    ├── 坐姿.png / 坐姿_alpha.png
    └── ...
```

## 四、Prompt 模板（SDXL）

| 产物 | 模板 |
|---|---|
| 三视图（单张） | `{desc}, {style}, full body, {front/side/back view}, plain white background, single character, single view, high quality` |
| 正面 | `{desc}, {style}, full body, front view, standing pose, plain white background, single character, high quality` |
| 动作 | `{desc}, {style}, full body, {动作英文}, plain white background, single character, high quality` |
| 表情（特写） | `{desc}, {style}, close-up portrait, {表情英文} facial expression, face filling the frame, plain white background, high quality` |

- `{desc}` 即角色一致性铁律的"固定复用短语"，所有镜头/姿态共用
- 统一 `plain white background`：便于 RMBG 抠图（白底是最稳的抠图场景）

## 五、重要说明

1. **透明 PNG 与 I2V 的关系**：AnimateDiff 是整图扩散（img2img 全画面重绘），透明 PNG 直接输入会变黑底 → **I2V 锚定/首帧一律用白底图**（`front.png`）；透明 PNG 是后期合成/贴片资产，当前管线暂不消费
2. **三视图不进视频**：三视图是档案，喂给 I2V 会产生"三个角色同时出现/视角混乱"（用户调研结论）
3. **定妆照人工可换**：`front.png` 不满意的直接替换文件（保留同名），run_story 无需改动
4. **视角能力边界（v1.2，按能力裁剪）**：SDXL 分不清方向（实测 back view 连续两次画成正面，肤色检测 93%/80% 露脸）。**生产只用 FRONT**（锚定必需）；SIDE/BACK 为可选档案，`--views FRONT,SIDE,BACK` 显式传入才生成；**BACK 生成后自动肤色验证，失败即删除跳过**（不重试浪费）
5. **头盔/面具角色表情受限**：表情表靠脸部特写表达，戴头盔/面具的角色（如宇航员）表情出不来——此类角色要么接受弱表情，要么另做脸部立绘
6. **表情与定妆照的脸不一致**：表情图独立文生图，无锚定，脸与 front.png 不同；如需强一致后续用 IPAdapter 锚定生成
7. **成本**：完整版（正面+3 动作+3 表情）≈ 7 张 × 71s ≈ **9 分钟** + 抠图；三视图可选加 1~2 张

## 六、变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| 1.0 | 2026-08-15 | 初版：三阶段工作流 + prompt 模板 + 目录结构 + I2V 接线 |
| 1.1 | 2026-08-15 | 三视图改单视图拼版（一图三视图不可靠）；动作表扩至 14；新增表情表 6（特写）；记录头盔角色/一致性限制 |
| 1.2 | 2026-08-15 | 集成 prompt-hub（提示词唯一真源）；视角按能力裁剪（生产只 FRONT，BACK 自动验证跳过）；desc 长斗篷遮脚 |
| 1.3 | 2026-08-15 | **Lumo 形象定版**：底座 Counterfeit-V3.0_fp16（SD1.5 系安全动漫）+ 纯真约束提示词；候选 seed 801-805 |

## 七、形象定版记录（Lumo v7.1，2026-08-15）

> 用户确认：Counterfeit-V3 候选"还行"（干净无擦边），定为角色底座与提示词基线

### 模型选择（Anything V5 → Counterfeit-V3）

| 底座 | 问题 | 结论 |
|---|---|---|
| SDXL base 1.0 | 写实画动漫 = 恐怖谷/脸崩 | 只做写实备用 |
| Anything V5 | 训练数据杂，**圆身+斗篷易出擦边** | ❌ 弃用 |
| **Counterfeit-V3.0_fp16** | 日本画师安全向数据，SD1.5 系 | ✅ **Lumo 定版**（与 AnimateDiff 同底座） |

### 提示词基线（v7.1）

```
desc: a tiny round glowing light fairy with a round chubby body, a cute face with big bright sparkling eyes, a loose baggy yellow cloak fully covering her feet, a small glowing wick on her chest, holding a large lantern bigger than herself, wingless

正面约束: cute healing fantasy, adorable, innocent, wholesome, pure, childlike cuteness, modest
负面约束: NSFW, suggestive, revealing, mature, sexy, sensual, tight clothing, cleavage, wings, shoes, visible feet, deformed
```

- **宽松斗篷**（loose baggy）防贴身曲线；**纯真词**（innocent/wholesome/modest）压擦边
- 候选 seed：801-805（`candidates_cf/`），抽卡 10s/张（ComfyUI 路径，RAM 安全）
- 输出路径：`outputs_character/lumo/candidates_cf/`
