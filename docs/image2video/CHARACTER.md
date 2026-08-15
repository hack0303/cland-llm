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

### 姿态表（--poses 可选值）

| 中文 | 英文提示词 | | 中文 | 英文提示词 |
|---|---|---|---|---|
| 站立 | standing pose, facing forward | | 微笑 | smiling expression, close-up face |
| 坐姿 | sitting pose | | 惊讶 | surprised expression, close-up face |
| 奔跑 | running pose, dynamic | | 挥手 | waving one hand, friendly |
| 战斗 | battle stance | | 思考 | thinking pose, hand on chin |

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
| 三视图 | `{desc}, {style}, character sheet, turnaround, three views of the same character, front view side view and back view, full body, plain white background, character design sheet, high quality` |
| 正面 | `{desc}, {style}, full body, front view, standing pose, plain white background, single character, high quality` |
| 姿态 | `{desc}, {style}, full body, {姿态英文}, plain white background, single character, high quality` |

- `{desc}` 即角色一致性铁律的"固定复用短语"，所有镜头/姿态共用
- 统一 `plain white background`：便于 RMBG 抠图（白底是最稳的抠图场景）

## 五、重要说明

1. **透明 PNG 与 I2V 的关系**：AnimateDiff 是整图扩散（img2img 全画面重绘），透明 PNG 直接输入会变黑底 → **I2V 锚定/首帧一律用白底图**（`front.png`）；透明 PNG 是后期合成/贴片资产，当前管线暂不消费
2. **三视图不进视频**：三视图是档案，喂给 I2V 会产生"三个角色同时出现/视角混乱"（用户调研结论）
3. **定妆照人工可换**：`front.png` 不满意的直接替换文件（保留同名），run_story 无需改动
4. **质量上限**：SDXL 文生图三视图的视图间一致性一般（同 prompt 不同 seed 会漂），若需强一致可后续加 ControlNet 或角色 LoRA
5. **成本**：默认 6 张图（三视图+正面+4 姿态）≈ 6×71s ≈ 7 分钟 + RMBG 抠图 ~1 分钟

## 六、变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| 1.0 | 2026-08-15 | 初版：三阶段工作流 + prompt 模板 + 目录结构 + I2V 接线 |
