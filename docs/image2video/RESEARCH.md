# 图生视频（Image-to-Video）选型研究报告（C-Land 本机部署）

> 硬件：2× Tesla P40 (24GB, Pascal sm_61) + 15GB RAM + 3GB swap
> 现状：GPU 0 = SDXL（13GB 常驻），GPU 1 = TripoSG（4.15GB 常驻）+ TTS/SFX（~5GB），视频服务规划用 **GPU 1 余量（~15GB）**
> 载体：ComfyUI（2026-08-15 全新安装于 `/mnt/data/ai_workspace/ComfyUI`，master a7365071，base 环境 torch 2.7.1+cu118 可直接跑 ✅）
> 结论先行：社区"一般"选择确实是 **AnimateDiff**（SD 生态 + 运动模块，最成熟轻量）；**游戏资产生成场景（风格化 + 轻微动画）主选 AnimateDiff**，写实/大幅度运动才上 LTX-Video 2B；写实级长片段走付费 API（Grok 视频 5¢/s，见 cland-game-asset-gen D 路线）

## 一、需求分类

| 场景 | 类型 | 典型用途 |
|---|---|---|
| 图片动起来（轻微动画） | I2V 短视频 | 立绘呼吸/飘发、场景运镜、UI 动效参考 |
| 角色动作短片（5s 内） | I2V 动作 | 技能演示、待机动画参考、过场素材 |
| 高质量写实片段 | 付费 API | 宣传片、CG 级素材（本地算力不可达） |
| 快速占位/验证构图 | 本地小模型 | 动画分镜预演、构图验证（秒~分钟级出片） |

## 二、硬件约束（视频特有病根）

| 约束 | 对视频生成的影响 |
|---|---|
| P40 无 Tensor Core | FP16 仅为 23 TFLOPS（4090 的 ~1/10），视频扩散需数百步时序注意力 → **出片慢 10~20 倍**，5s 片段按"分钟~十几分钟"预期 |
| 无 flash-attn | 长时序注意力（81~121 帧）走慢路径，显存占用更高；依赖 flash-attn 的模型直接排除 |
| 内存仅 15GB | 大文本编码器（T5-XXL 系 ~10GB fp16）必须 CPU offload；**长帧数 + 高分辨率会爆内存**，需限帧数/分辨率 |
| 双卡 48GB 显存 | GPU 0 被 SDXL 占 13GB；GPU 1 余 ~15GB → 只能容纳 **2B 级 fp16 模型（~5-6GB）+ CPU offload 文本编码器** |
| 视频输出 | 需 ffmpeg 合成 mp4（ComfyUI 内置）；P40 上 VAE decode 也是大头耗时 |

## 三、候选模型对比（开源可自部署，2026-08）

### 能跑的（≤2B，P40 可行）

| 模型 | 代际（发布） | 参数量 | 输出规格 | 文本编码器 | License | P40 可行性 | 备注 |
|---|---|---|---|---|---|---|---|
| **AnimateDiff**（SD1.5/SDXL 运动模块） | **第一代**（2023.07） | 底座 1.4B + 模块 0.5B | 16~32 帧@8fps（≈2~4s），512/1024px | CLIP（小） | ✅ 运动模块 Apache 2.0 | ✅ 全方案最轻 | **社区主流**：LoRA 海量 + 工作流成熟 + 与 SD 生图无缝；运动幅度小，I2V 需 img2img + IP-Adapter 拐弯 |
| **LTX-Video 2B**（Lightricks） | **第三代**（2024.11） | 2B | 768×512@24fps，最长 5s（121 帧） | T5-XXL（大，CPU offload） | ✅ **Apache 2.0** | ✅ fp16 ~4GB + VAE | 高帧率/生成最快（4090 上近实时），ComfyUI 原生支持；写实/通用向首选 |
| **Wan2.1-I2V-1.3B**（阿里） | **第三代**（2025.02） | 1.3B | 480p，81 帧 | UMT5-XXL（大，CPU offload） | ✅ **Apache 2.0** | ✅ fp16 ~3GB | 1.3B 尺寸下质量口碑最佳，写实向；14B 版远超硬件上限 |
| **CogVideoX-2B**（智谱） | 第二代（2024.08） | 2B | 720×480@8fps，49 帧（6s） | CogVideoX-T5（~3B） | ⚠️ **需核验**（2B/5B 协议不同，商用前查 HF 卡） | ✅ fp16 ~5GB | 中文提示词友好，生态成熟；帧率低（8fps）动作偏缓 |
| **SVD / SVD-XT**（Stability） | 第二代（2023.11） | 1.4B UNet | 576×1024，14/25 帧（2~4s@6fps） | CLIP-L + OpenCLIP-H（小） | ⚠️ Stability 社区协议（年收入 <$100万 可商用） | ✅ fp16 ~4GB 全包 | 2023 老将但**最稳**，文本编码器小、内存压力最小，ComfyUI 原生 |
| I2VGen-XL（阿里） | 第二代（2023） | 3.5B | 720p，16 帧（4s） | - | ⚠️ 需核验 | ⚠️ fp16 ~7GB，GPU 1 勉强 | 老，仅对比 |
| DynamiCrafter（腾讯） | 第二代（2024.02） | 1.4B | 576×1024，16 帧 | - | ⚠️ 研究用途限制 | ✅ | 老，仅对比 |

### 代际速览（为什么帧数/fps 差这么多）

| 代际 | 时期 | 代表 | 架构思路 | 帧数能力 |
|---|---|---|---|---|
| **第一代** | 2023 | AnimateDiff / Deforum / I2VGen-XL | 图模型 + 外挂运动模块 | 16 帧 / 6~8fps |
| **第二代** | 2023.11~2024 | SVD / CogVideoX-2B / DynamiCrafter | 原生视频扩散，规模仍小 | 14~49 帧 / 6~8fps |
| **第三代** | 2024.11~2025 | LTX-Video / Wan2.1 / HunyuanVideo / Mochi | 大 DiT + T5-XXL 级文本编码器 | 81~121 帧 / 16~24fps |

> 代差本质：文本编码器（CLIP 0.25GB → T5-XXL ~10GB）+ 帧数能力（16 → 121）+ 运动幅度（轻微 → 真实物理）。越新越强，但 RAM/显存/算力要求同步飙升——本机（P40 + 15GB RAM）上限恰好卡在"第一代最轻 ↔ 第三代小模型"之间，这就是选型必须跨代对比的原因。

### 跑不动的（排除）

| 模型 | 代际（发布） | 参数量 | 排除原因 |
|---|---|---|---|
| HunyuanVideo（腾讯） | 第三代（2024.12） | 13B | fp16 权重 ~26GB + 长时序注意力，双卡拼接也无 flash-attn；15GB RAM 直接出局 |
| Wan2.1-I2V-14B | 第三代（2025.02） | 14B | 同上，权重 ~29GB，非 4090 级不可用 |
| CogVideoX-5B | 第二代（2024.08） | 5B | fp16 ~10GB 贴线，无余量给激活/VAE；P40 出片时间不可接受 |
| Mochi-1 / Open-Sora 2.0 | 第二/三代（2024.10 / 2025.03） | 10-11B | 同上，出局 |
| Kling / Runway / Vidu / 即梦 | 闭源（2024~） | 闭源 | 无本地部署选项 → 走付费 API 对比（下） |

### 付费 API 备选（质量天花板，按量付费）

| 服务 | 价格参考 | 备注 |
|---|---|---|
| **Grok 视频**（asset-gen `video --image pose.png --duration 2`） | **5¢/s** | 技能 D 路线已打通：Gemini 参考图 + Grok 视频；好看但不可控，超时≠失败勿重提交 |
| Gemini（Veo 系） | 按 API 计费 | 精确需求首选（与出图同生态） |
| Kling 可灵 / Runway Gen-3+ / Vidu / 即梦 | 会员/积分制 | 中文生态即梦/Kling 好，无 API 或需审核 |

## 四、决策树

```
需求：图生视频
├─ 游戏资产生成 / 风格化 / 轻微动画 → AnimateDiff（✅ 首选：SD 生态 + LoRA 海量 + 最轻量）
├─ 写实 / 大幅度运动 / 高帧率 ────→ LTX-Video 2B（Apache + 最快 + 原生 I2V）
├─ 本地写实向 / 480p 足够 ───────→ Wan2.1-I2V-1.3B（Apache，1.3B 质量标杆）
├─ 本地中文提示词 / 6s 长一点 ────→ CogVideoX-2B（先核验 License）
├─ 快速占位 / 内存最紧 ──────────→ SVD（文本编码器最小，最稳）
└─ 写实 CG 级 / 长片段 ────────→ 付费 API（Grok 5¢/s / Veo / Kling），调用前确认花费
```

## 五、推荐结论（2026-08-15 研究稿，未部署）

| 用途 | 选型 | 理由 |
|---|---|---|
| **本地主选（游戏资产）** | **AnimateDiff** | 风格化/卡通 + LoRA 生态 + 与 SDXL 管线无缝 + 显存/RAM 最轻；运动幅度小是特性不是缺陷（待机/呼吸/飘动正合适） |
| **本地主选（写实/通用）** | LTX-Video 2B | Apache + 24fps 高帧率 + 原生 I2V；代价：T5-XXL ~10GB 文本编码器占 RAM |
| **本地备选** | Wan2.1-I2V-1.3B | Apache + 1.3B 尺寸质量口碑最佳，写实向；与 LTX 互补 |
| **中文/慢动作** | CogVideoX-2B | 中文提示友好，但 License 需先核验 |
| **内存最紧场景** | SVD | 文本编码器小，15GB RAM 压力最小 |
| **高质量长片段** | Grok 视频（付费） | 已有 asset-gen 管线（5¢/s），本地不可达场景直接付费 |

### License 红线

- **SVD**：Stability 社区协议，**年收入 ≥$100万 不可商用** → 商用项目慎用
- **CogVideoX**：2B/5B 协议不同（HF 卡标注），商用前必须核验
- **DynamiCrafter**：研究用途限制，直接排除商用
- **LTX-Video / Wan2.1-1.3B**：Apache 2.0，无商用限制 ✅

## 六、部署记录（已完成 2026-08-15）

| 项 | 实际值 |
|---|---|
| 载体 | ComfyUI headless（`/mnt/data/ai_workspace/ComfyUI`，master a7365071，setsid 隔离启动）✅ |
| GPU | GPU 1（CUDA_VISIBLE_DEVICES=1）✅，常驻显存 **3.1GB** |
| 端口 | **10337**（ComfyUI API：/prompt + /system_stats）✅ |
| 环境 | base（torch 2.7.1+cu118）✅，无新 conda |
| 模型 | SD1.5 底座 2.06GB + mm_sd_v15_v2 1.82GB + IP-Adapter 45MB + CLIP-ViT-H 2.53GB（共 4.4GB，hf-mirror）✅ |
| 节点 | ComfyUI-AnimateDiff-Evolved（144 节点）+ ComfyUI_IPAdapter_plus（35 节点）✅ |
| 客户端 | `inference/i2v/generate.py`（图片→mp4，webp→PIL 帧提取→ffmpeg）✅ |
| 输出 | `/mnt/data/ai_workspace/outputs_video/*.mp4` ✅ |
| **实测耗时** | **512×512 16 帧 20 步 = 140s**（2m20s）；1024×1024 = ~15.5min（45s/步，超 SD1.5 训练分布）| 
| 首条出片 | `i2v_cat_paw.mp4`（1024²）+ `i2v_cat_paw_512.mp4`（512²），2s@8fps 轻微动画 ✅ |
| 兼容补丁 | IPAdapter CrossAttentionPatch dtype 对齐（P40 fp32 query × fp16 k/v）；见 PITFAILLOG |

### 端口分配总表（更新）

| 端口 | 服务 | GPU |
|---|---|---|
| 10303 | vllm（gemma，未常驻） | 双卡 |
| 10331 | SDXL 生图 | 0 |
| 10332 | TripoSG 图生 3D | 1 |
| 10333/10334 | TTS/ASR（语音） | 1 |
| 10335/10336 | 音乐/音效（规划） | 1 |
| **10337** | **图生视频 AnimateDiff（已部署 ✅）** | **1** |

### 联动管线（图 → 视频 → 配音）

```
SDXL 生图 (10331) → 参考图 → 图生视频 (10337 AnimateDiff/LTX) → mp4
Spark-TTS (10333) → 配音 → ffmpeg 合成最终素材
```

## 七、风险提示

- **AnimateDiff 边界**：运动幅度天生有限（适合轻微动画/待机/飘动），大幅度动作/写实会露馅；8fps 偏低、长序列（>24 帧）易漂移 → 帧数上限 16~24
- **出片速度**：P40 无 Tensor Core，视频扩散是现有服务中最重的负载；首版务必实测 5s 片段耗时与显存峰值再定帧数上限
- **15GB RAM 是硬瓶颈**：T5-XXL/UMT5-XXL 文本编码器 ~10GB 必须 CPU offload（ComfyUI 默认支持），但首次 encode 会慢；长片段（>121 帧）+ 高分辨率有 OOM/swap 风险，建议 **≤121 帧、≤768×512** 起步
- **flash-attn 缺失**：选型已避开强依赖 flash-attn 的模型；LTX/Wan 走标准注意力，P40 可用但慢
- **ComfyUI 依赖**：custom_nodes 目前仅示例；视频工作流需下载官方 workflow JSON（LTX/Wan 官方仓库自带）
- **与付费路线分工**：本地只做 ≤5s 短片/占位；宣传级素材直接走 Grok（5¢/s），勿在 P40 上硬耗数小时
