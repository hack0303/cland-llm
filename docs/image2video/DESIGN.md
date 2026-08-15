# 图生视频系统设计文档（C-Land I2V Pipeline）

> 版本：1.0（2026-08-15）· 硬件：2× Tesla P40 (24GB) + 15GB RAM
> 相关规范：`STORYBOARD.md`（分镜格式 v1.1）· 选型背景：`RESEARCH.md` · 案例：`example/case001.md`
> 服务使用手册：`../comfyui/USAGE.md`

## 一、系统目标

把一段故事文本，自动变成**带配音、音效、BGM 的多镜头视频**：

```
故事文本 → 分镜 JSON → 逐镜头生产（画面+动画+声音）→ 合成大视频
```

约束：P40 无 Tensor Core（慢）、15GB RAM（模型必须轻）、AnimateDiff 第一代能力（轻微动画）。

## 二、架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                       storyboard.py                          │
│   故事 ──Gemma-4 (10303)──▶ storyboard.json（LLM 分镜）       │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      run_story.py                            │
│   逐镜头循环：                                               │
│   ┌───────────┐   ┌──────────────┐   ┌──────────────────┐   │
│   │ SDXL 10331 │──▶│ AnimateDiff   │──▶│ TTS 10333        │   │
│   │ 出参考图   │   │ I2V 10337     │   │ SFX 10336        │   │
│   │ frames/    │   │ clips/        │   │ audio/           │   │
│   └───────────┘   └──────────────┘   └──────────────────┘   │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      compose.py                              │
│   片段拼接（concat/xfade） + 音频 adelay 对齐 + amix 混音      │
│   → {prefix}_final.mp4                                       │
└─────────────────────────────────────────────────────────────┘
```

### 服务矩阵

| 端口 | 服务 | 模型 | GPU | 角色 |
|---|---|---|---|---|
| 10303 | Gemma-4 26B（llama.cpp） | UD-Q4_K_M 16GB | 1（分时） | 故事→分镜（LLM） |
| 10331 | SDXL 文生图 | SDXL base 1.0 | 0 | 每镜头参考图 |
| 10337 | ComfyUI 图生视频 | SD1.5 + mm_sd_v15_v2 + IP-Adapter | 1 | 动画片段 |
| 10333 | TTS | Spark-TTS 0.5B | 1 | 台词/旁白 |
| 10336 | SFX | AudioGen 1.5B | 1 | 事件音效 |

> ⚠️ 显存分时：Gemma（16GB）与 ComfyUI（3.1GB 常驻）同卡 GPU 1，分镜生成与 I2V 出片**必须错开**（先出分镜 → 停 Gemma → 跑镜头）。

## 三、一致性保证（三层机制）

| 层 | 机制 | 解决什么 | 实现 |
|---|---|---|---|
| 1. 分镜文本 | LLM 提示词铁律：角色短语跨镜头固定复用 | 提示词层面漂移 | storyboard.py SYSTEM_PROMPT |
| 2. **角色锚定图** | IPAdapter 单独吃 `assets/character.png`（镜头 1 出图即定妆照），img2img 吃镜头首帧 | **跨镜头长相漂移**（核心） | workflow 节点 14 + generate.py `--ref-image` |
| 3. 首帧接力（可选） | `--chain-frames`：镜头 N 首帧 = 镜头 N-1 末帧 | 同场景跨镜头连续性 | run_story.py `extract_last_frame` |

**锚定图工作流**（改造后）：

```
assets/character.png ──▶ LoadImage(14) ──▶ IPAdapterAdvanced ──┐
frames/scene00X.png ───▶ LoadImage(2) ──▶ VAEEncode ──▶ RepeatLatentBatch(16) ──▶ KSampler
                                                                                    ▲
                                                    ADE_AnimateDiffLoaderWithContext ┘
```

- 默认：镜头 1 出图自动复制为定妆照，后续镜头 IPAdapter 全部锚定它（换场景也不丢脸）
- `--no-character-lock`：回退旧行为（IPAdapter 用各自首帧，适合单镜头）
- `--chain-frames`：与锚定图叠加（IPAdapter=定妆照 + img2img=上镜末帧）

## 四、数据流与目录

分镜格式与目录结构见 `STORYBOARD.md`（v1.1），要点：

```
outputs_video/{prefix}/
├── storyboard.json        # 唯一真源（含 output 回填）
├── assets/character.png   # 角色定妆照（镜头 1 出图，自动生成）
├── frames/scene001.png    # SDXL 参考图
├── clips/scene001.mp4     # I2V 片段 512×512 @8fps
├── audio/scene001_voice.wav / scene001_sfx.wav
└── {prefix}_final.mp4     # 合成大视频
```

时间轴：镜头 k 全局时间 = Σ前序时长 + 镜头内偏移；compose 用全局时间戳放置配音/音效。

## 五、工具链接口

```bash
# 1. 分镜生成（需 Gemma 10303）
python3 inference/i2v/storyboard.py --story "..." --prefix story001
python3 inference/i2v/storyboard.py --story-file story.txt --prefix story001 --clips 6

# 2. 镜头生产（需 10331/10337，TTS/SFX 可选自动跳过）
python3 inference/i2v/run_story.py --sb outputs_video/story001/storyboard.json
#     --no-character-lock  关闭角色锚定
#     --chain-frames       首帧接力（同场景连续）
#     --skip-audio         跳过声音
#     --skip-video         仅合成

# 3. 单镜头/合成（底层）
python3 inference/i2v/generate.py --image ref.png --ref-image character.png --prefix scene001
python3 inference/i2v/compose.py --clips a.mp4 b.mp4 --voice v.wav --voice-at 1.0 --bgm b.mp3 --transition 0.5
```

## 六、性能基准（P40 实测，case001）

| 环节 | 耗时 | 备注 |
|---|---|---|
| SDXL 出图（1024²，30 步） | ~71s | GPU 0 |
| I2V 单镜头（512²，16 帧，20 步） | **140s** | GPU 1 常驻 3.1GB |
| I2V 单镜头（1024²） | ~15.5min | 不推荐（超 SD1.5 训练分布） |
| Gemma 分镜（26B，~400 token） | 1~3min | 与 I2V 错开 |
| 合成（5 镜头硬切） | <30s | ffmpeg CPU |

**产能预估**：8 镜头故事 ≈ 8×(71s+140s) + 分镜 + 合成 ≈ **35~40 分钟/条**。

## 七、已知限制与后续

| 限制 | 影响 | 后续方案 |
|---|---|---|
| 轻微动画（AnimateDiff 第一代） | 无大幅度运动 | LTX-Video 2B（写实向，14GB 下载） |
| 8fps / 2s 每镜 | 节奏偏缓 | RIFE 帧插值 16→48 帧 @24fps |
| 定妆照=镜头 1 出图 | 若镜头 1 崩则全崩 | 支持人工替换 assets/character.png（改图即生效） |
| 场景变化时角色适配 | 锚定图构图可能被强拉 | IPAdapter weight 调参 / ControlNet 构图对齐 |
| 中文台词 | TTS 效果已达标 | 情绪/音色参数化 |
| BGM | 未部署 | ACE-Step（10335 规划）或 CC0 |

## 八、变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| 1.0 | 2026-08-15 | 初版：管线架构 + 三层一致性 + 工具链 + 性能基准 |
