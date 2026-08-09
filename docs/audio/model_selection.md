# 音乐/音效生成选型文档（C-Land 本机部署）

> 硬件：2× Tesla P40 (24GB) · 现状：GPU 0=SDXL(13GB)，GPU 1=TripoSG(4.15GB，余 14GB)
> 补充：AI 生成模型仅覆盖"完整曲目/复杂音效"；短促 UI 音效、环境垫底仍优先程序化/CC0（见 cland-game-asset-gen skill B/C 路线）

## 一、需求分类

| 场景 | 类型 | 说明 |
|---|---|---|
| 完整 BGM 曲目 | 音乐生成 | 文本提示 → 数十秒~数分钟音乐 |
| 带人声歌曲 | 音乐生成+人声 | 歌词/哼唱 → 歌曲 |
| 事件音效（脚步声/雨/爆炸） | SFX 生成 | 文本 → 音效片段 |
| 短促 UI 音效 | 程序化/CC0 | 0 成本，不走模型（已有路线） |
| 环境氛围垫底 | CC0 / 程序化 | 已有路线 |

## 二、候选模型对比（2026-08 实测 HF 数据）

### 音乐生成（BGM/歌曲）

| 模型 | 参数量 | 输出 | 人声 | 中文提示 | License | HF 热度 | P40 可行性 |
|---|---|---|---|---|---|---|---|
| **ACE-Step**（阿里） | ~4B | 44.1kHz 音乐+人声 | ✅ | ✅ | ✅ **Apache 2.0** | ModelScope/GitHub 分发 | ✅ fp16 ~8GB |
| **MusicGen-medium**（Meta） | 1.5B | 32kHz 音乐 | ❌ | ⚠️ 英文佳 | ✅ Apache 2.0 | **187 万** | ✅ fp16 ~3GB |
| **MusicGen-large** | 3.3B | 32kHz 音乐 | ❌ | ⚠️ | ✅ Apache 2.0 | 8.9 万 | ✅ fp16 ~7GB |
| **Stable Audio Open**（Stability） | 1.2B | 44.1kHz 立体声 | ❌ | ⚠️ | ⚠️ **收入<100 万美元限制** | 1.8 万 | ✅ fp16 ~3GB |

### 音效生成（SFX）

| 模型 | 参数量 | 输出 | License | 热度 | P40 可行性 |
|---|---|---|---|---|---|
| **AudioGen-medium**（Meta） | 1.5B | 16kHz 事件音效 | ✅ Apache 2.0 | 2.8 万 | ✅ |
| **Tango2**（SALMONN 团队） | 0.86B | 16kHz 音效 | ✅ Apache 2.0 | 冷门 | ✅ |
| Stable Audio Open | 1.2B | 也可做音效 | ⚠️ 同上 | - | ✅ |

### 在线付费备选

- **Suno**：音乐生成天花板，按量付费（有免费额度）
- **ElevenLabs SFX**：音效 API，按秒计费

## 三、决策树

```
需求：游戏/项目音乐音效
├─ 完整 BGM / 带人声歌曲 ──→ ACE-Step（Apache 可商用 + 中文 + 人声）
│     └ 要稳定成熟 / 纯音乐 ─→ MusicGen-medium（生态最成熟，187 万下载）
├─ 事件音效（脚步声/雨/战斗）→ AudioGen-medium
├─ 短促 UI 音效 / 提示音 ───→ 程序化合成（0 成本秒级，不走模型）
├─ 环境氛围垫底 ──────────→ CC0 素材站 / 程序化垫底
└─ 追求顶级质量（可付费）──→ Suno（音乐）/ ElevenLabs SFX（音效）
```

## 四、推荐结论

| 用途 | 选型 | 理由 |
|---|---|---|
| **BGM/歌曲** | **ACE-Step**（首选） | Apache 2.0 商用友好 + 中文 + 人声；4B P40 可跑 |
| **BGM 稳定备选** | MusicGen-medium | 生态最成熟（audiocraft 工具链完善），质量稳定 |
| **事件音效** | **AudioGen-medium** | Apache + Meta 出品，与 MusicGen 同工具链 |
| UI 音效/垫底 | 程序化 + CC0 | 已有路线，0 成本（不部署模型） |

### License 红线

- **Stable Audio Open**：Stability Community License，**年收入 ≥100 万美元不可商用** → 商用项目排除
- ACE-Step / MusicGen / AudioGen 均 Apache 2.0，无商用限制

## 五、部署规划（确认后执行）

| 项 | 规划 |
|---|---|
| GPU | GPU 1 余量（14GB），ACE-Step ~8GB + AudioGen ~3GB 可共存 |
| 端口 | **10335**（音乐 ACE-Step）/ **10336**（音效 AudioGen） |
| 环境 | 新建 `audio_env`（torch 2.6/2.7 + cu118，audiocraft 依赖） |
| 服务形态 | 常驻 FastAPI（复用现有 server 模式） |
| 输出 | `/mnt/data/ai_workspace/outputs_audio/`（wav） |
| 显存预算 | 音乐 ~8GB + 音效 ~3GB = ~11GB（GPU 1 余 14GB ✅） |

### 端口分配总表（更新）

| 端口 | 服务 | GPU |
|---|---|---|
| 10303 | vllm（gemma，未常驻） | 双卡 |
| 10331 | SDXL 生图 | 0 |
| 10332 | TripoSG 图生 3D | 1 |
| 10333/10334 | TTS/ASR（语音，规划） | 1 |
| **10335/10336** | **音乐/音效（规划）** | 1 |

## 六、风险提示

- MusicGen 生成音乐 32kHz 采样率（低于 CD 44.1kHz），要求高需 ACE-Step/Stable Audio
- 音乐生成模型对提示词风格依赖强（"epic orchestral, 120bpm, minor key"），英文提示词效果更稳
- ACE-Step 无 HF 分发，从 GitHub/ModelScope 获取，部署需按官方文档装依赖
