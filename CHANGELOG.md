---
description: record your changes
---

# Changelog

## 20260920

### Changes

- 推理/决策模型：新增 `inference/nanojev/`——开源 System One 复现本地部署（NanoJev 10338 / Von 10339）：独立 venv（复用 base torch 2.7.1+cu118）+ `sitecustomize.py`（torch 2.7 的 `torch._native.triton_utils` no-op shim）+ 下载/启动/基准脚本；文档 `inference/nanojev/README.md` 与 `docs/systemone/README.md`
- 基准复现/50×50 maze：NanoJev fp32 **244 attempts / 36 collisions / goal**（与官方 A100-bf16 记录逐项一致），27.5s / 199 前向；未调 Qwen3-0.6B fp32 4134/1768/goal（167.2s），atomic accuracy 43.3% vs NanoJev 85.9%
- 基准复现/Snake 8 局 cohort：NanoJev ——4 trapped/4 survived、总 food 168、mean 21.0，**逐 case 与官方 8/8 一致**（showcase 12:61005 = 27 food/256 步存活）；未调 Qwen3-0.6B ——5 trapped/3 survived、mean 23.75，同样 8/8 一致（showcase 25 food/211 步 trapped）
- 服务性能（P40 实测）：NanoJev fp32 加载 46.2s、显存 2.39GB（峰值 2.55GB）、单 state 4 题 136.7ms p50（HTTP 134.4ms / 7.44 req/s）；Von 单 noul 61ms、3 题 fan-out 150ms、显存 ~1GB
- 模型/权重：下载 NanoJev（local_atomic/games_gold 2.39GB×2）、Von-1.0（1.58GB）、Qwen3-0.6B 基线（固定 revision）、NanoJev-Data games_v4/arcade 小包（官方 cohort+回放）

### Fixes

- 修复：上游评估脚本在 torch 2.7 报 `ModuleNotFoundError: torch._native`（上游记录 torch 2.14）→ `sitecustomize.py` 提供 no-op 等价 shim，不改上游源码
- 修复：hf-mirror 长下载断连（`httpx.RemoteProtocolError`，von-1.0 到 980MB/1.58GB）→ 重跑同命令续传完成
- 优化：P40 上 `is_bf16_supported()`=True 为模拟语义（bf16 实测比 fp32 慢 ~1.5×）→ 服务与基准默认 `--precision fp32`
- 修复：Von 自动 dtype 在 P40 选 bf16 模拟导致决策边界精度回退（官方 `test_fanout` noul 0.498 vs fp32 0.509）→ 新增 `patches/von-p40-fp32.patch`（`VON_DTYPE` 覆盖）+ `start_von.sh` 默认 fp32；CUDA 跑 von 全套测试 **23 passed**，服务端复测 noul 0.5092，fp32 延迟 32.6ms/单题、93.1ms/3 题（比 bf16 快 ~1.9×）

## 20260906

### Changes

- 推理/文本 LLM：gemma4（10303）单卡部署——`gemma4.py` 顶部 `CUDA_VISIBLE_DEVICES=1`（setdefault，可被显式环境变量覆盖），26B-A4B Q4 权重 + 16K KV 全压 1 号 P40（~20GB/24GB），0 号卡留给 OCR 等分卡并行；单卡免去双卡张量并行的 PCIe 卡间同步开销
- 推理/文本 LLM：gemma4 每请求打印 ENTER/DONE 日志（请求编号/线程/总耗时/prompt+completion tokens）到 /tmp/gemma4-server.log，批量任务可据此对账
- 推理/文本 LLM：新增 `inference/gemma/bench10303.py` 吞吐/并发复测脚本；实测基线（单卡 GPU1）：decode 稳态 43-48 tok/s、batch=10 画像 ~10s/请求、7300 人全量约 2.0-2.6h；并发 1/2/4 路无聚合收益（服务端日志实证 ENTER 恒在上一 DONE 后，并发=1），批量管道建议 workers=1 + 单请求做厚

## 20260826

### Changes

- 推理/手脚修复管线：新增 `inference/sdxl/hand_pipe/` 工业级 AI 生图手脚崩坏修复服务（GPU 1 / 端口 10335），五环节：SDXL txt2img(8bit) → ControlNet OpenPose 骨骼约束（手部 21 点）→ DWPose 检测手部 → SDXL-inpaint 局部重绘 → Real-ESRGAN 4x 超分；`POST /pipeline` 端到端 4 场景实测 328-358s/场景，DWPose 客观验收双手 21/21 点
- 推理/检测：新增 `dwpose.py` DWPose 全身 133 点姿态检测（yolox_l onnx 标准 grid+stride decode + RTMPose SimCC 双输出 + top-down 仿射预处理，onnxruntime CPU 不占 GPU）
- 推理/超分：新增 `rrdbnet.py` Real-ESRGAN RRDBNet 纯 torch 实现（官方/ComfyUI 双格式权重 key 适配，4x-UltraSharp + RealESRGAN_x4plus 双模型 1024→4096 验证）
- 推理/LoRA：新增 `test_lora.py` 手部 LoRA 对比测试；调研 5 候选后确认 `Benevolent/Perfect Hands v2`（SDXL 格式）有效（peace_sign 左手关键点 10→16），实现 `load_kohya_lora_manual` 手动注入（绕开 diffusers 0.39 rank 推断 bug）
- 模型：下载 ControlNet SDXL 三件套（openpose/depth/canny 11.8GB）、SDXL-inpaint（20GB，diffusers 镜像）、DWPose（350MB）、ESRGAN（134MB）、手部 LoRA 4 个候选（1.7GB）
- 环境：`onnxruntime`、`controlnet_aux`、`ultralytics`、`onnx` 安装（清华源）；`TORCHINDUCTOR_COMPILE_THREADS=1` 防 15GB 内存 OOM（100 个 compile_worker 子进程）
- 文档：新增 `docs/sdxl/hand_pipe.md` 管线技术手册（环节测试结论/API/踩坑 10 条）；`docs/QUICK_START.md` 服务总览登记 10335；新增 `TODO-hand-pipe.md` 任务看板（18 完成/14 待办）

### Fixes

- 修复：diffusers 0.39 `load_in_8bit` 参数弃用 → `PipelineQuantizationConfig(quant_backend="bitsandbytes_8bit")`；单 CN 传 list 挂 MultiControlNetModel（显式包装 + image/scale 传 list）
- 修复：超分权重加载静默失败（4x-UltraSharp 为 ComfyUI 格式、原实现架构错误），重写 RRDBNet（RDB1/2/3 三子块）+ 双格式 key 适配，输出黑图/噪声修复为正常图
- 修复：DWPose yolox 检测（标准 grid+stride decode 后 conf 0.95）、RTMPose SimCC 解码（非 heatmap）、仿射变换（去掉 mmpose 老版 ×200 因子）
- 修复：显存 OOM（三 CN 常驻 22.4GB → 只常驻 openpose 14.3GB + empty_cache + expandable_segments）
- 修复：整机内存 OOM 重启（100 个 torch compile_worker 吃光 15GB RAM，限制 `TORCHINDUCTOR_COMPILE_THREADS`）

## 20260815

### Changes

- 推理/图生视频：落地 AnimateDiff I2V 管线（ComfyUI GPU 1 / 端口 10337 + SD1.5 底座 + mm_sd_v15_v2 运动模块 + IP-Adapter，共 4.4GB；客户端 `inference/i2v/generate.py`）
- 推理/图生视频：新增 `inference/i2v/compose.py` 大视频合成（多片段硬切/xfade 淡化拼接 + 配音/音效 adelay 对齐 + BGM 铺底 + amix 混流）
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

## 20260815（续）

### Changes

- 推理/图生视频：完成《最后一颗火种》Lumo 全流水线（prompt-hub 提示词 → char_sheet 角色 v4 → 手动分镜 8 镜头 → run_story → 合成 17.6s 含配音），case002
- 修复：SDXL 并发 500（调度器加锁）、I2V 1024 慢 6.6 倍（generate.py --size 512）、run_story 健康检查兼容 /system_stats、负面词去掉 ugly/deformed（反向降质）
- 决策：SFX 跳过（15GB RAM）、三视图只 FRONT（SDXL 方向能力边界）、desc 长斗篷遮脚（脚细节规避）
