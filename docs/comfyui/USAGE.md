# ComfyUI 使用手册（C-Land 图生视频承载）

> 部署：`/mnt/data/ai_workspace/ComfyUI`（master a7365071，ComfyUI 0.33.0）· GPU 1 · 端口 10337
> 用途：图生视频（AnimateDiff 主方案）的推理载体；SDXL/3D/TTS 仍走独立 FastAPI 服务，互不干扰
> 前置：base 环境（torch 2.7.1+cu118，P40 sm_61 兼容），无需新 conda

## 一、为什么用它承载视频

| 对比 | 自研 FastAPI | ComfyUI headless |
|---|---|---|
| 视频模型支持 | 每个模型手写 diffusers 管线 | 原生支持 AnimateDiff/LTX/Wan/CogVideoX 等全部视频模型 |
| 工作流 | 代码硬编码 | JSON 工作流，节点可换（img2img/IPAdapter/ControlNet 拖拽组合） |
| 踩坑成本 | 每个模型重踩一遍 | 社区节点（144+35 个）已解决 |
| 代价 | — | 多一层 API（/prompt + /history 轮询） |

图生视频链路复杂（运动模块注入 + IPAdapter 身份保持 + 首帧 img2img），ComfyUI 节点生态是捷径。

## 二、启动与停止

```bash
# 启动（GPU 1 / 10337，setsid 隔离进程组，防 bash 超时误杀）
cd /mnt/data/ai_workspace/ComfyUI
setsid env CUDA_VISIBLE_DEVICES=1 nohup python main.py --port 10337 --listen 127.0.0.1 > /tmp/comfyui.log 2>&1 < /dev/null &

# 停止（两步走，防 pkill 自匹配自杀）
pgrep -f "main.py --port 10337"   # 拿 PID
kill <PID>

# 健康检查
curl http://127.0.0.1:10337/system_stats | python3 -m json.tool | grep -A4 devices
```

> ⚠️ 坑：`nohup` 不防 SIGTERM，bash 工具超时杀进程组会连坐 ComfyUI → 必须 setsid；`pkill -f "main.py"` 会匹配到自己的命令行 → 两步走杀进程（详见 PITFAILLOG）

## 三、模型目录（models/）

| 目录 | 内容 | 本次已装 |
|---|---|---|
| `checkpoints/` | 底座（SD1.5 等） | ✅ v1-5-pruned-emaonly.safetensors |
| `animatediff_models/` | AnimateDiff 运动模块 | ✅ mm_sd_v15_v2.ckpt |
| `ipadapter/` | IP-Adapter 投影器 | ✅ ip-adapter_sd15.safetensors |
| `clip_vision/` | 图像编码器（ViT-H 等） | ✅ CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors |
| `loras/` / `vae/` / `controlnet/` | 扩展（后续按需） | 空 |

> 新增模型文件后**需重启 ComfyUI** 才会出现在节点下拉列表（启动时扫描目录）。
> 下载统一走 hf-mirror（`HF_ENDPOINT=https://hf-mirror.com`），实测 ~2.9MB/s。

## 四、API 速查

| 端点 | 用途 |
|---|---|
| `GET /system_stats` | 健康/设备/版本 |
| `GET /object_info` | 全部节点定义（类名/输入/输出），对接工作流必查 |
| `POST /prompt` | 提交工作流 `{"prompt": {...}}`，返回 `prompt_id` |
| `GET /history/{prompt_id}` | 轮询结果（completed/error + 输出文件） |
| `GET /queue` | 队列状态 |

```bash
# 提交工作流（workflow JSON 见 inference/i2v/workflow_i2v.json）
curl -X POST http://127.0.0.1:10337/prompt -H "Content-Type: application/json" \
  -d @workflow.json
```

## 五、图生视频使用（推荐走封装脚本）

```bash
cd /mnt/data/ai_workspace/cland-llm

# 一条命令出片（512 甜点 ~2.5 分钟）
python3 inference/i2v/generate.py --image ref.png --prefix my_video --seed 43

# 常用参数
--frames 16      # 帧数（AnimateDiff 安全区 16，上限待测）
--steps 20       # 采样步数
--denoise 0.8    # 首帧保留强度（0.7~0.9 控运动量）
--fps 8.0        # 输出帧率（16 帧 = 2s）
--cfg 7.0        # CFG
```

内部流程：参考图 → ComfyUI input/ → 提交 workflow → 轮询 history → webp → **PIL 逐帧提取 → ffmpeg 拼 mp4**（ffmpeg 直接解动画 webp 会失败，exit 69）

输出：`/mnt/data/ai_workspace/outputs_video/*.mp4`

### 工作流节点链（inference/i2v/workflow_i2v.json）

```
LoadImage → VAEEncode → RepeatLatentBatch(16)          # 首帧复制 16 帧
→ ADE_StandardUniformContextOptions(16,1,4)             # 时序上下文
→ ADE_AnimateDiffLoaderWithContext(mm_sd_v15_v2)        # 运动模块注入
→ IPAdapterUnifiedLoader(STANDARD) → IPAdapterAdvanced  # 身份保持
→ KSampler(20 步 euler denoise 0.8) → VAEDecode
→ SaveAnimatedWEBP
```

## 六、节点管理

```bash
cd /mnt/data/ai_workspace/ComfyUI/custom_nodes
git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved   # 144 节点
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus              # 35 节点
```

- 装节点后重启 ComfyUI；用 `/object_info` 验证注册（`python -m json.tool | grep AnimateDiff`）
- 新版节点（1.6.0）用 ComfyExtension 注册，类名即 workflow 里的 `class_type`，以 `/object_info` 返回为准

## 七、故障排查

| 症状 | 处理 |
|---|---|
| 采样中途进程消失、日志无报错 | 被进程组连坐杀死 → setsid 启动（见上） |
| `query/key dtype 不一致`（P40） | IPAdapter dtype 补丁：`CrossAttentionPatch.py` 已对齐 fp32/fp16；改代码后**必须重启** |
| 新模型不在下拉列表 | 模型文件放对目录 + 重启 |
| 动画 webp 转 mp4 失败（exit 69） | PIL 逐帧提取（generate.py 已内置），勿直接 ffmpeg 解 webp |
| 显存不足 | 降到 512×512、帧数 ≤16；参考图分辨率就是生成分辨率 |
| 首次加载慢 | 正常（SD1.5 全量 ~1-2 分钟），之后模型常驻 3.1GB |

## 八、端口分配（现状）

| 端口 | 服务 | GPU |
|---|---|---|
| 10331 | SDXL 生图 | 0 |
| 10337 | **ComfyUI（图生视频）** | 1 |
| 10332/10333/10334/10336 | 3D/TTS/ASR/音效 | 1 |
