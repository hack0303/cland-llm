# 分镜格式与目录结构规范（v1.0）

> 流水线：故事 → 分镜 JSON → 逐镜头生产（SDXL 出图 → I2V 动画 → TTS 配音 / SFX 音效）→ compose 合成大视频
> 工具：`inference/i2v/storyboard.py`（生成）、`inference/i2v/run_story.py`（执行）、`inference/i2v/compose.py`（合成）
> 本规范是 storyboard JSON 与输出目录的唯一真源，工具输出必须遵守

## 一、分镜 JSON 格式（storyboard.json）

```json
{
  "schema_version": "1.0",
  "title": "猫猫登月",
  "style": "cute cartoon, soft lighting, vibrant colors",
  "story": "原始故事文本（可选，保留溯源）",
  "total_duration": 8.0,
  "clips": [
    {
      "scene": 1,
      "duration": 2.0,
      "image_prompt": "a cute cat astronaut standing on the moon surface, waving one paw, cinematic lighting, vibrant colors",
      "motion_prompt": "subtle motion, gentle breeze, slight paw waving",
      "voice": "我们终于到达月球了！",
      "voice_at": 0.5,
      "sfx": "风声、宇航服呼吸声",
      "sfx_at": 0.0,
      "output": {
        "frame": "frames/scene001.png",
        "clip": "clips/scene001.mp4",
        "voice": "audio/scene001_voice.wav",
        "sfx": "audio/scene001_sfx.wav"
      }
    }
  ]
}
```

### 字段规则

| 字段 | 类型 | 必填 | 约束 |
|---|---|---|---|
| `schema_version` | string | ✅ | 固定 `"1.0"` |
| `title` | string | ✅ | 中文标题 |
| `style` | string | ✅ | 英文画风描述（进 SDXL 提示词） |
| `story` | string | 可选 | 原始故事文本溯源 |
| `total_duration` | float | ✅ | = Σ clips.duration（合成校验用） |
| `clips[]` | array | ✅ | 1~N 个镜头，按 scene 升序 |
| `clips[].scene` | int | ✅ | 从 1 起递增 |
| `clips[].duration` | float | ✅ | **1~6 秒**；推荐 2s（16 帧@8fps）或 4s（32 帧）；超出 AnimateDiff 帧数安全区需实测 |
| `clips[].image_prompt` | string | ✅ | 英文 SDXL 画面提示词；连续镜头保持角色/场景一致 |
| `clips[].motion_prompt` | string | ✅ | 英文 I2V 运动提示词（subtle motion 类） |
| `clips[].voice` | string | 可选 | 中文台词/旁白；空串=无配音 |
| `clips[].voice_at` | float | 条件必填 | 镜头内偏移秒，须 `< duration` |
| `clips[].sfx` | string | 可选 | 中文音效描述；`"无"`=无音效 |
| `clips[].sfx_at` | float | 条件必填 | 镜头内偏移秒，须 `< duration` |
| `clips[].output` | object | 可选 | **流水线执行后回填**的实际文件路径（相对故事目录），生产前可为空 |

### 时间轴换算

- 镜头 k 的**全局起始时间** = Σ(clips[0..k-1].duration)
- 配音/音效的全局时间 = 镜头全局起始 + 镜头内 `*_at` 偏移（compose.py 的 `--voice-at/--sfx-at` 用全局时间）

## 二、目录结构（outputs_video/）

```
/mnt/data/ai_workspace/outputs_video/
└── {story_prefix}/                    # 每个故事一个目录（storyboard --prefix 决定）
    ├── storyboard.json                # 分镜定义（唯一真源）
    ├── frames/                        # SDXL 参考图
    │   ├── scene001.png
    │   └── scene002.png
    ├── clips/                         # I2V 动画片段（512×512 @8fps）
    │   ├── scene001.mp4
    │   └── scene002.mp4
    ├── audio/                         # 声音素材
    │   ├── scene001_voice.wav         # TTS 配音（Spark-TTS 10333）
    │   └── scene001_sfx.wav           # 音效（AudioGen 10336）
    └── {story_prefix}_final.mp4       # 合成大视频（compose 输出）
```

### 规则

1. 故事目录名 = `storyboard.py --prefix`（小写字母数字下划线，如 `story001`）
2. 镜头文件名 = `scene{编号:03d}`，从 `scene001` 开始
3. `clips[].output` 回填的是**相对路径**（相对故事目录），保证目录可整体迁移
4. 合成产物 `{story_prefix}_final.mp4` 与 storyboard.json 同级
5. 中间产物（webp/临时转码）不落故事目录，由工具在临时目录处理

## 三、工具接口

```bash
# 1. 生成分镜（需 Gemma-4 10303 运行）
python3 inference/i2v/storyboard.py --story "故事..." --prefix story001

# 2. 执行流水线（逐镜头：SDXL 出图 → I2V → TTS/SFX；需 10331/10337/10333/10336 运行）
python3 inference/i2v/run_story.py --storyboard outputs_video/story001/storyboard.json

# 3. 合成大视频（复用已有素材，纯 ffmpeg）
python3 inference/i2v/compose.py --clips ... --voice ... --prefix story001_final
```

## 四、变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| 1.0 | 2026-08-15 | 初版：schema + 目录结构 + 工具接口定义 |
