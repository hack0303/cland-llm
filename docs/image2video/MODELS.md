---
title: "模型资产清单（下载记录）"
summary: "本机已下载模型全清单：路径、大小、来源、用途、状态（在用/备用/弃用）"
read_when:
  - "确认某个模型是否已下载/在哪里"
  - "磁盘清理决策（哪些可删）"
  - "新增模型前查重"
scope:
  - cland-llm
status: "active"
updated: "2026-08-15"
---

# 模型资产清单

> 全部经 hf-mirror 下载（`HF_ENDPOINT=https://hf-mirror.com`，实测 ~2.9MB/s）
> 权重不入 git（.gitignore），只此清单入库

## 一、图生视频链路（角色/出图/动画）

| 模型 | 路径 | 大小 | 来源 repo | 用途 | 状态 |
|---|---|---|---|---|---|
| **Counterfeit-V3.0_fp16** | `ComfyUI/models/checkpoints/` | 4.0G | gsdf/Counterfeit-V3.0 | **Lumo 定版底座**（SD1.5 系安全动漫，与 AnimateDiff 同源） | ✅ 在用 |
| v1-5-pruned-emaonly | `ComfyUI/models/checkpoints/` | 4.0G | stable-diffusion-v1-5 | AnimateDiff 底座（通用） | ✅ 在用 |
| Anything V5 | `ComfyUI/models/checkpoints/anything-v5-3d.safetensors` | 2.0G | ckpt/anything-v5.0 | 动漫底座 | ❌ **弃用**（擦边问题，见 CHARACTER.md v1.3） |
| mm_sd_v15_v2 | `ComfyUI/models/animatediff_models/` | 1.7G | guoyww/animatediff | AnimateDiff 运动模块 | ✅ 在用 |
| ip-adapter_sd15 | `ComfyUI/models/ipadapter/` | 43M | h94/IP-Adapter | IPAdapter 投影器（角色锚定） | ✅ 在用 |
| CLIP-ViT-H-14-laion2B | `ComfyUI/models/clip_vision/` | 2.4G | h94/IP-Adapter | IPAdapter 图像编码器 | ✅ 在用 |
| RMBG-1.4 | `inference/triposg/pretrained_weights/RMBG-1.4` | 804M | briaai/RMBG-1.4 | 抠图（透明 PNG 资产） | ✅ 在用 |

## 二、出图服务（SDXL 10331）

| 模型 | 路径 | 大小 | 用途 | 状态 |
|---|---|---|---|---|
| stable-diffusion-xl-base-1.0 | `models/` | 25G | SDXL 服务（写实/复杂场景备用） | 📦 备用（动漫主力已切 Counterfeit） |
| stable-diffusion-xl-base-1.0-fp16 | `models/` | 6.5G | fp16 权重备份 | 📦 备用 |

## 三、其他服务（语音/3D/文本，带过）

| 模型 | 大小 | 用途 | 状态 |
|---|---|---|---|
| Spark-TTS-0.5B | ~2G | 配音（10333） | ✅ 在用 |
| SenseVoiceSmall | ~1G | ASR（10334，按需） | 📦 按需 |
| AudioGen-medium | ~3G | 音效（10336，RAM 限制跳过） | 📦 按需 |
| TripoSG + RMBG | 7.5G+0.8G | 图生 3D（10332） | ✅ 在用 |
| gemma-4-26B（AWQ/UD-Q4）+ 31B + E4B | ~16G×2 | 文本/分镜（10303） | 📦 按需 |

## 四、总账

- **图生视频链路**：4.0+4.0+2.0+1.7+0.04+2.4+0.8 ≈ **14.9G**
- 全机模型目录：`/mnt/data/ai_workspace/models/` + `ComfyUI/models/` + 服务内嵌权重，合计 ~60G+

## 五、可清理项（磁盘紧张时）

| 项 | 大小 | 理由 |
|---|---|---|
| Anything V5 | 2.0G | 弃用（擦边）——确定新形象后删 |
| SDXL 原版（fp32 25G） | 25G | 服务跑的是 8bit 加载，fp16 备份 6.5G 已够兜底——确认后删 |
| gemma-4-31B 系列 | ~30G | 分镜用 26B 即可，31B 备用（按需删） |
