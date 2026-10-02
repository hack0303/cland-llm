---
title: cland-llm 模型台账（models inventory）
summary: 本仓推理/修图管线所用模型与权重的集中台账——模型名/来源/大小/本地路径/用途/登记状态，供追溯与运维。
read_when: 部署/迁移本仓推理服务、核对权重来源与许可、新增或替换模型、模型资产登记（OpenCMDB）时。
tags: [models, inventory, 模型台账, sdxl, ddcolor, esrgan, dwpose]
owner: sage
status: "2026-10-02 · 首版"
---

# cland-llm 模型台账（models inventory）

> 权重统一存放于 `/mnt/data/ai_workspace/models/`。本台账为仓库内可追溯来源；同步登记到 OpenCMDB（模型资产）。

## 一、模型清单

| 模型 | 版本 / 来源 | 大小 | 本地路径 | 用途 | 登记 |
|---|---|---|---|---|---|
| **DDColor-L** | modelscope `damo/cv_ddcolor_image-colorization`（ImageNet 训练，作者 Piddnad） | 912 MB | `/mnt/data/ai_workspace/models/ddcolor/ddcolor_modelscope.pt`（仓库代码 `/mnt/data/ai_workspace/DDColor`） | 老照片上色（#225 首选；`--colorizer ddcolor`） | OpenCMDB ✅ `ast-1790952748239` |
| SDXL base 1.0 (fp16) | `stabilityai/stable-diffusion-xl-base-1.0` | 6.5 GB | `/mnt/data/ai_workspace/models/stable-diffusion-xl-base-1.0-fp16` | repair / colorize / product img2img | — |
| RealESRGAN x4plus | RealESRGAN（RRDBNet） | 64 MB | `/mnt/data/ai_workspace/models/upscale/RealESRGAN_x4plus.pth` | 4x 超分（默认） | — |
| 4x-UltraSharp | 4x-UltraSharp（RRDBNet） | 64 MB | `/mnt/data/ai_workspace/models/upscale/4x-UltraSharp.pth` | 4x 超分（备选） | — |
| DWPose YOLOX-L | DWPose 人体检测 | 207 MB | `/mnt/data/ai_workspace/models/dwpose/yolox_l.onnx` | 人体检测（肤色掩膜） | — |
| DWPose dw-ll_ucoco_384 | DWPose 133 点关键点 | 129 MB | `/mnt/data/ai_workspace/models/dwpose/dw-ll_ucoco_384.onnx` | 脸/手/前臂关键点（`skin_fix`） | — |

## 二、引用与覆盖

- **DDColor 路径可覆盖**：环境变量 `DDCOLOR_REPO`（代码仓，默认 `/mnt/data/ai_workspace/DDColor`）、`DDCOLOR_CKPT`（权重，默认见上）。
- **超分权重**：`inference/photo/upscale.py` 的 `WEIGHTS` 表（realesrgan / ultrasharp）。
- **SDXL 路径**：`inference/photo/retouch.py` 的 `SDXL_FP16_DIR`。
- **DWPose 路径**：`inference/sdxl/hand_pipe/dwpose.py` 的 `model_dir`。

## 三、许可与合规

- DDColor：**代码 Apache-2.0**；**权重 Apache-2.0**（HF `piddnad/ddcolor_modelscope` 模型卡，经 hf-mirror 核实）→ **可商用**，保留 LICENSE 与模型卡引用。
- RealESRGAN / 4x-UltraSharp：BSD-3 / 各自许可证。
- DWPose：Apache-2.0。
- SDXL base：CreativeML Open RAIL++-M。
- 样例素材（`materials/`）仅用 **PD/CC0**，见 `docs/photo-retouch-v0.md` §3。

## 四、登记记录

| 日期 | 模型 | 登记位置 | 操作人 |
|---|---|---|---|
| 2026-10-02 | DDColor-L（modelscope `damo/cv_ddcolor_image-colorization`，912MB） | OpenCMDB 模型资产 `ast-1790952748239`（template `tmpl-1790952684373` AI Model）+ 本台账 | sage |
