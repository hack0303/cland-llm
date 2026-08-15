---
description: record your changes
---

# Changelog

## 20260815

### Changes

- 推理/图生视频：落地 AnimateDiff I2V 管线（ComfyUI GPU 1 / 端口 10337 + SD1.5 底座 + mm_sd_v15_v2 运动模块 + IP-Adapter，共 4.4GB；客户端 `inference/i2v/generate.py`）
- 实测：16 帧 512×512 20 步 = **140s**（2m20s），1024×1024 ≈ 15.5min；GPU 1 常驻显存 3.1GB；首条出片 `outputs_video/i2v_cat_paw*.mp4`
- 修复：IPAdapter CrossAttentionPatch dtype 对齐补丁（P40 fp32 query × fp16 k/v）；ffmpeg 动画 webp 转码改 PIL 逐帧提取；常驻服务 setsid 隔离启动
- 文档：新增 `docs/image2video/RESEARCH.md` 图生视频选型研究——游戏资产生成场景主选 AnimateDiff（SD 生态 + 最轻量），写实向备选 LTX-Video 2B / Wan2.1-I2V-1.3B，ComfyUI 承载规划端口 10337；付费兜底 Grok 视频 5¢/s

## 20260809

### Changes

- 推理/SDXL：新增 `stable-diffusion-xl-base-1.0` 文生图常驻服务（GPU 0 / 端口 10331），1024x1024 30 步实测 71s，峰值显存 10.45GB
- 推理/TripoSG：新增 `VAST-AI/TripoSG` 图生 3D 常驻服务（GPU 1 / 端口 10332），50 步推理 7.5min + 网格提取 18min，常驻显存 4.15GB
- 推理/模块：新增 `inference/` 目录，归拢 SDXL（`sdxl/`）、TripoSG（`triposg/`）、Gemma（`gemma/`）三套服务代码
- 推理/diso：编译 `diso-0.1.4` CUDA 扩展（conda gcc-11 + patch setup.py），产出 P40 (sm_61) 可用 `_C.so`
- 环境：新增 conda 环境 `triposg_env`（python 3.10 + torch 2.6.0+cu118 + torchvision 0.21.0）
- 模型：下载 `TripoSG` 权重 7.5GB 至 `models/TripoSG`、`RMBG-1.4` 背景移除模型 804MB
- 文档：新增 `docs/sdxl_USAGE.md`、`docs/3d/model_selection.md`、`inference/triposg/README.md`、`inference/README.md`
- 技能：新增 `cland-image-gen` agent skill（本机 SDXL 服务调用与模型选型）

### Fixes

- 推理/TripoSG：修复 FastAPI `async def` 内同步阻塞导致事件循环卡死的问题（改为 `def` 走线程池）
- 推理/TripoSG：修复 `prepare_image` 接收 `BytesIO` 报 `stat: path should be string` 异常（改为先落盘临时文件）
- 推理/diso：修复 conda gcc 找不到系统 C++ 头文件与 crt 库的问题（setup.py 补齐 include/link 参数）
