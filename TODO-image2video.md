# TODO-image2video: 图生视频服务实施看板

> 遵循 GFM 任务列表规范（todoprompt skill）
> 依据：`docs/image2video/RESEARCH.md`（选型：**AnimateDiff 主方案**，LTX-Video 2B 写实备选）
> 载体：ComfyUI（`/mnt/data/ai_workspace/ComfyUI`，已安装 master a7365071）
> 格式：`- [ ]` 待办 | `- [x]` 已完成 | `- [/]` 执行中 | `- [-]` 取消 | `- [!]` 阻塞

---

# 目标：P40 机器落地图生视频服务（ComfyUI 承载 · GPU 1 · 端口 10337 · AnimateDiff 首条出片）

## 一、环境与载体（ComfyUI）

- [x] ComfyUI 安装（`/mnt/data/ai_workspace/ComfyUI`，base 环境 torch 2.7.1+cu118 匹配 P40）[verify:: python -c "import torch; print(torch.__version__)"]
- [x] ComfyUI headless 启动验证（GPU 1，端口 10337）[verify:: /system_stats ✅ 0.33.0]
- [x] 安装 ComfyUI-AnimateDiff-Evolved 节点 [verify:: 144 节点已注册]
- [x] 安装 ComfyUI_IPAdapter_plus 节点 [verify:: 35 节点已注册]

## 二、模型下载（AnimateDiff 方案 ~4-6GB，hf-mirror 2.9MB/s ≈ 30-40min）

- [x] 模型下载完成（SD1.5 2.06GB + mm_sd_v15_v2 1.82GB + IP-Adapter 45MB + CLIP-ViT-H 2.53GB，共 4.4GB）[priority:: high]
- [x] 运动模块 `mm_sd_v15_v2.ckpt`（~1.7GB）[priority:: high] [verify:: ls models/animatediff_models/ ✅ 1.82GB]
- [x] IP-Adapter SD1.5 权重 + CLIP-ViT-H 图像编码器（~0.7GB）[priority:: high] [verify:: ls models/ipadapter/ models/clip_vision/ ✅]
- [ ] （可选）LTX-Video 2B + LTXV VAE + t5xxl_fp16 写实向补全（~14GB）[priority:: low] [estimate:: 1.5h]

## 三、工作流与首条出片

- [x] 编写 I2V 工作流 JSON（`inference/i2v/workflow_i2v.json` + `generate.py` 提交脚本）[priority:: critical]
- [x] 用 SDXL 服务（10331）生成测试参考图 [priority:: high] ✅ 猫宇航员挥手
- [x] 跑通第一条 16 帧动画 [priority:: critical] [verify:: 512px 140s / 1024px 15.5min，输出 mp4 可播放 ✅]
- [x] ffmpeg 合成 mp4 + 参数调优（分辨率/耗时实测；帧数/步数上限留待批量验证）

## 四、服务化（端口 10337）

- [ ] 薄 FastAPI 网关 `inference/i2v/server.py`（POST /generate：图片→mp4，复用 repo server 模式）[priority:: medium]
- [ ] /health 健康检查 + 启动脚本集成 [priority:: medium] [verify:: curl http://127.0.0.1:10337/health]
- [ ] 性能实测（耗时 / 显存峰值 / 帧数上限）→ 回填 RESEARCH.md 实测数据 [priority:: medium]
- [ ] 更新 QUICK_START 服务总览 / CHANGELOG（按需 PITFAILLOG）

## 五、可选优化

- [ ] RIFE 帧插值 16→48 帧 @24fps 输出（ComfyUI Frame Interpolation 节点）[priority:: low]
- [ ] 动漫向底座 / LoRA（Anything V5 等）接入游戏资产生成风格 [priority:: low]

---

## 状态汇总

| 状态 | 计数 | 说明 |
|------|------|------|
| `- [x]` 已完成 | 10 | 环境/模型/首条出片全部落地 |
| `- [/]` 执行中 | 0 | — |
| `- [ ]` 待执行 | 5 | 网关/LTX 补全/帧插值/动漫底座 |
| `- [!]` 失败/阻塞 | 0 | — |
| **总计** | **15** | — |

> 最后更新：2026-08-15
