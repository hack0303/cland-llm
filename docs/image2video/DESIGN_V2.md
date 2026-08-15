# V2 工业架构设计（多管线分离）

> 版本：2.0（2026-08-16）· 状态：实施中
> 背景：V1 全能管线（单 AnimateDiff 承担角色+背景+动作）一路踩坑（脸/背景/动作互相妥协）
> 原则：**每个管线单目标、可调优、可替换、可并行**，最后合成合并

## 一、架构总览

```
inputs: storyboard.json + character.json（定妆/脸部锚定）

┌─ 管线 A：角色（pipe_char.py）────────────────────┐
│ 白底出图（脸部锚定 + white bg prompt）            │
│ → RMBG 抠图 → 透明角色图                          │
└──────────────────────────────────────────────────┘
┌─ 管线 B：背景（pipe_bg.py）──────────────────────┐
│ 纯场景出图（无角色 prompt）→ 背景图（512/768）     │
└──────────────────────────────────────────────────┘
┌─ 管线 C：动画（pipe_anim.py）────────────────────┐
│ 白底角色图 → I2V（16帧 + face_anchor 锚定）       │
│ → RMBG 逐帧抠图 → 透明角色视频序列                │
└──────────────────────────────────────────────────┘
┌─ 管线 D：音频（pipe_audio.py）───────────────────┐
│ 预配音（克隆锚定，尽力而为）→ loudnorm 16kHz      │
│ → 时间戳（voice_at + 动态时长）                   │
└──────────────────────────────────────────────────┘
        ↓ 管线 E：合成（merge.py）
┌─ 合成 ──────────────────────────────────────────┐
│ 透明角色视频 overlay 到背景（位置/缩放/阴影）      │
│ + 音频混流 → 成片                                │
└──────────────────────────────────────────────────┘
```

## 二、管线职责与接口

| 管线 | 输入 | 输出 | 关键点 |
|---|---|---|---|
| A 角色 | character.json + 分镜 | `assets/char_white/*.png`（白底）+ `assets/char_alpha/*.png`（透明） | 脸部锚定；白底与定妆一致 → 锚定完美 |
| B 背景 | 分镜（场景 prompt） | `assets/bg/*.png` | 无角色 prompt；质量独立拉满 |
| C 动画 | 白底角色图 | `assets/anim/{scene}_alpha_%03d.png`（透明序列） | 16 帧固定（140s）；逐帧 RMBG |
| D 音频 | 分镜（台词） | `assets/audio/*.wav` + 时间戳表 | 预配音；PTS 修复链复用 |
| E 合成 | 上述全部 | `{prefix}_final.mp4` | ffmpeg overlay；位置/缩放/阴影 |

## 三、各管线细节

### 3.1 角色管线（A）

- 出图：Counterfeit + face_anchor 锚定 + `plain white background` prompt（与定妆同底色）
- 每个镜头一张：角色姿态由分镜 motion 描述 + 动作资产（waving/holding...）
- RMBG：复用 `to_alpha_png`（RMBG-1.4）
- 输出：白底图（合成前预览用）+ 透明图（动画输入）

### 3.2 背景管线（B）

- 出图：Counterfeit/SDXL 纯场景 prompt（分镜 image_prompt 去掉角色前缀，场景段保留+强化）
- 无角色 → 无锚定 → 背景自由发挥（废墟/雪地/月光细节拉满）
- 分辨率 512（与动画一致）或 768（合成时缩放，细节更好）

### 3.3 动画管线（C）

- 输入：透明角色图 → 贴白底（或直接白底图）→ I2V 16 帧（face_anchor 锚定）
- 输出：16 帧动画 → **逐帧 RMBG 抠图** → 透明视频序列（PNG 序列 + ffmpeg 拼透明视频）
- 帧数固定 16（速度）；时长靠合成 tpad

### 3.4 音频管线（D）

- 复用 V1 Phase 0：预配音（克隆锚定尽力而为）+ loudnorm + aresample 16000 + PTS 修复
- 输出：wav + 全局时间戳（voice_at + 动态时长累计）

### 3.5 合成管线（E）——核心新代码

```python
merge.py --bg bg/scene001.png --chars anim/scene001_alpha_%03d.png \
         --fps 8 --frames 16 --position center --scale 1.0 \
         --shadow true --audio voice.wav --audio-at 0.5 --duration 4.0
```

- overlay：`ffmpeg -i bg.png -i char_video.mov -filter_complex "[1:v]scale=W:H[ch];[0:v][ch]overlay=x:y"` 
- 位置/缩放：--position（center/bottom-left/...）+ --scale（角色占背景比例）
- 阴影：椭圆黑色模糊层（可选，提升贴合感）
- 时长：tpad 补帧（16 帧 → duration）
- 音频：复用 compose 的音频链（loudnorm + asetpts + adelay + amix）

## 四、编排（pipeline.py）

```python
pipeline.py --sb storyboard.json --character character.json --prefix lumo
1. 管线 A+B（出图，可并行：白底角色 + 背景）
2. 管线 C（动画：逐镜头 I2V 16帧 + 抠图）
3. 管线 D（音频：预配音，可与 1 并行）
4. 管线 E（合成：overlay + 音频 → 成片）
```

- 幂等：每管线产物检查（文件存在跳过）——断点续跑
- 每管线可独立运行/替换（`--only char/bg/anim/audio/merge`）

## 五、与 V1 的关系

- V1 保留可用（inference/i2v/）——V2 修复了 V1 的结构性问题
- 复用：RMBG（char_sheet）、预配音逻辑（run_story Phase 0）、音频链（compose）、出图工作流（t2i_anchor）
- 新增：背景管线、透明动画序列、合成（merge.py）

## 六、验收标准

- [ ] 单镜头全链路：白底角色 + 背景 + 动画 + 合成 → 角色清晰（811 脸）、背景完整、动作 16 帧
- [ ] 速度：单镜头 < 3 分钟（出图 12s×2 + I2V 140s + 抠图 32s + 合成 10s）
- [ ] 音频：16kHz 正常、响度一致、PTS 无爆炸
- [ ] 幂等断点：中断重跑跳过已完成
