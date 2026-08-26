---
title: "AI 生图手脚崩坏修复 — 工业管线手册"
summary: "SDXL+ControlNet(OpenPose)+局部重绘+超分四环节手脚修复管线，含组件测试结论、API 手册与踩坑记录"
read_when:
  - "搭建或调试 AI 生图手脚修复管线（10335 服务）"
  - "处理生成图手指扭曲/多指/手掌粘连问题"
  - "使用 DWPose 姿态检测或 ControlNet OpenPose 约束生成"
  - "排查 SDXL 8bit 量化、显存 OOM 或 diffusers 0.39 兼容问题"
scope:
  - inference/sdxl/hand_pipe
  - inference/sdxl/server.py
status: "active"
updated: "2026-08-26"
---

# AI 生图手脚崩坏修复 — 工业管线手册

> 依据 ShowDoc「AI生图手脚崩坏原因与解决方案」（LLM 项目 / 251514559）落地的本机管线。
> 代码：`inference/sdxl/hand_pipe/` · 服务：**GPU1 / 端口 10335** · 前置：base 环境（torch 2.7.1+cu118，P40 sm_61）

## 一、背景与目标

AI 生图（SD 系）最常见缺陷：手指扭曲、数量不对、手掌粘连、脚趾畸形。根本原因是**手部训练样本稀缺 + 扩散模型对高拓扑结构的弱约束**，提示词只能改善简单姿态。工业级方案是组合管线：

```
基础模型 + 手部LoRA → ControlNet OpenPose(手部21点) 骨骼约束 → 生成大图
  → DWPose 检测手脚区域 → 局部重绘精修 → 最后全局超分（顺序不可颠倒）
```

本管线落地 ShowDoc 方案 1/3/4/6 四个环节（方案 2 LoRA 与方案 5 HandRefiner 为可选增强，见文末）。

## 二、组件架构

```
┌────────────────────────── GPU1 :10335 (pipe_server.py) ──────────────────────────┐
│                                                                                   │
│  POST /generate      SDXL base 1.0 (8bit)          txt2img 基线/首轮生成           │
│  POST /pose          DWPose (yolox+rtmpose onnx)   全身 133 点 + 骨架图 + 手部区域 │
│  POST /cn_generate   ControlNet OpenPose/Depth/Canny 骨骼约束生成（管线核心）      │
│  POST /inpaint       SDXL-inpaint (8bit)           手部局部重绘（自动检测区域）     │
│  POST /upscale       Real-ESRGAN (4x-UltraSharp/x4plus) 全局超分（最后一步）       │
│  POST /pipeline      端到端编排：txt2img→CN→inpaint→upscale                        │
│                                                                                   │
│  检测器 DWPose 走 CPU (onnxruntime)，不占 GPU                                      │
└───────────────────────────────────────────────────────────────────────────────────┘
```

GPU 分工：GPU0 = 10331 基础文生图服务（原 SDXL）；GPU1 = 10335 管线服务（base+CN+inpaint+ESRGAN 常驻）。

## 三、模型资产清单

| 资产 | 位置 | 大小 | 来源（hf-mirror） |
|---|---|---|---|
| SDXL base 1.0 | `models/stable-diffusion-xl-base-1.0/` | 7.3GB | 已有 |
| ControlNet OpenPose SDXL | `models/sdxl_controlnet/openpose/` | 4.7GB | `xinsir/controlnet-openpose-sdxl-1.0` |
| ControlNet Depth SDXL | `models/sdxl_controlnet/depth/` | 2.4GB | `xinsir/controlnet-depth-sdxl-1.0` |
| ControlNet Canny SDXL | `models/sdxl_controlnet/canny/` | 4.7GB | `xinsir/controlnet-canny-sdxl-1.0` |
| SDXL-inpaint | `models/sdxl-inpaint/` | 20GB | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`（官方仓库 gated，必须用 diffusers 镜像） |
| DWPose 检测器 | `models/dwpose/` | 350MB | `yzd-v/DWPose`（yolox_l.onnx + dw-ll_ucoco_384.onnx） |
| 超分模型 | `models/upscale/` | 134MB | `lokCX/4x-Ultrasharp`；`lllyasviel/Annotators`（RealESRGAN_x4plus，注意 ai-forever 仓库文件已删） |

> ⚠️ `stabilityai/stable-diffusion-xl-1.0-inpainting-0.1` 是 gated 仓库，直接下载报 `Invalid username or password`，必须用 `diffusers/` 官方镜像。

## 四、环节测试结论（2026-08-26 实测）

### 4.1 提示词方案（ShowDoc 方案 1）— 有改善但不稳定

4 场景 × 2 方案对比（seed 固定），用 DWPose 手部关键点检出数评估：

| 场景 | 基线 handPts | 提示词方案 handPts | 结论 |
|---|---|---|---|
| 人像托腮 | 21（仅右手检出） | 42（双手检出） | ✅ 改善 |
| 比耶手势 | 21 | 16 | ⚠️ 不稳定 |
| 双手持杯 | 42 | 62（双人） | ⚠️ 场景相关 |
| 芭蕾舞者 | 42 | 38 | ⚠️ 不稳定 |

**结论**：正向词 `perfect hands, detailed fingers, five fingers per hand...` + 负面词 `bad hands, missing fingers, extra fingers, deformed hands...` 仅改善简单姿态，与 ShowDoc 判断一致。**ControlNet 骨骼约束才是主方案**。

### 4.2 DWPose 姿态检测（管线基石）— ✅ 通过

- 人体检测：yolox_l.onnx，标准 grid+stride decode + NMS（**不能用直接 resize + 简单阈值**，见踩坑 2）
- 关键点：dw-ll_ucoco_384.onnx，RTMPose SimCC 表示（simcc_x/simcc_y 双输出，**非 heatmap**）
- 预处理必须：bbox→center/scale(×1.25)→仿射变换（**无黑边**）+ ImageNet 归一化
- 实测：1 人检测 conf 0.95，body 关键点 conf 0.87，右手 21 点全检出

### 4.3 ControlNet OpenPose 生成（核心环节）— ✅ 通过

`POST /cn_generate`（openpose 条件）实测 373s/张（首次含 kernel 编译），出图正常、无 OOM。条件图为 DWPose 骨架（含手部 21 点连线，绿=身体 红=左手 蓝=右手）。

### 4.4 局部重绘 Inpaint — ✅ 通过

`POST /inpaint` 实测 149.7s：自动 DWPose 检测手部 bbox（含 35% 膨胀 + 保底 96px）→ 羽化 mask → SDXL-inpaint 重绘（denoise 0.45）。

### 4.5 超分 — ✅ 通过（显存优化后）

- 独立脚本 `rrdbnet.py`：1024→4096 成功（4x-UltraSharp 与 RealESRGAN_x4plus 均验证）
- 服务内 `/upscale` 在**显存优化后通过**：静态占用 18.2GB→14.3GB，不再 OOM

### 4.6 端到端 /pipeline — ✅ 全部通过（2026-08-26 实测）

4 场景（比耶/持杯/托腮/舞者，seed 42）全部 4 步跑通，无 OOM：

| 场景 | step1 txt2img | step2 CN | step3 inpaint | step4 upscale | 总计 |
|---|---|---|---|---|---|
| peace_sign | 106s | 309s | 354s | 358s | **357.7s** |
| holding_cup | 83s | 282s | 327s | 330s | **329.6s** |
| chin_hands | 81s | 281s | 325s | 328s | **328.1s** |
| dancer | 81s | 281s | 326s | 328s | **328.2s** |

**质量评估**（DWPose 手部关键点，对比基线测试的 conf 0.53/单侧检出）：

| 指标 | step1 后 | step3 后 |
|---|---|---|
| 身体 17 点 | 17/17 | 17/17 |
| 左手 21 点 | 21/21 | 21/21 |
| 右手 21 点 | 21/21 | 21/21 |
| 手部最大 conf | 0.97 | 0.98 |

ControlNet 环节（step2）即锁定双手姿态；inpaint（step3）进一步提升细节置信度。

## 五、服务使用手册

### 1. 启动

```bash
cd /mnt/data/ai_workspace/cland-llm/inference/sdxl/hand_pipe
# 必须限制 torch compile worker 数量（机器仅 15GB RAM，默认 32 个 worker 直接 OOM 重启）
TORCHINDUCTOR_COMPILE_THREADS=1 nohup python3 -u pipe_server.py --port 10335 > /tmp/pipe_server.log 2>&1 &
# 加载约 9.5 分钟（base 8bit + 3×ControlNet + inpaint 8bit + ESRGAN）
curl http://127.0.0.1:10335/health   # → {"status":"ok",...}
```

### 2. API 速查

| 接口 | 关键参数 | 说明 |
|---|---|---|
| `POST /generate` | prompt, seed, steps, width, height | txt2img 基线 |
| `POST /pose` | image(路径) | DWPose 检测，返回骨架图+手部 bbox |
| `POST /cn_generate` | prompt, condition(openpose/depth/canny), source_image 或 image, cn_strength | ControlNet 约束生成 |
| `POST /inpaint` | image, denoise(0.3-0.5), hand_box(可空=自动) | 手部局部重绘 |
| `POST /upscale` | image, model(ultrasharp/esrgan) | 超分（当前 OOM，见已知问题） |
| `POST /pipeline` | prompt, use_cn/use_inpaint/use_upscale | 端到端编排 |

`/pipeline` 编排顺序（ShowDoc 方案 6 的**正确顺序**）：

```
step1 txt2img → step2 ControlNet OpenPose 重生成（姿态锁定）
  → step3 DWPose 检测手部 → inpaint 精修 → step4 全局超分
```

> 先修手脚再超分；先放大再修会固化畸形。

### 3. 测试脚本

```bash
python3 hand_pipe/test_cases.py          # 基线+提示词方案 8 图对比
python3 hand_pipe/run_pipeline.py --scenes 0,1,2,3   # 端到端管线 4 场景
python3 hand_pipe/dwpose.py <img> <out>  # DWPose 单图检测
python3 hand_pipe/rrdbnet.py <pth> <img> <out>       # 超分单测
```

### 4. 手部质量客观评估

无人工看图时用 DWPose 手部关键点数量/置信度量化"手部质量"：

```python
# 21 点全部 conf>0.3 → 手部可检测；conf 均值低 → 疑似畸形
det = DWPose(); res = det.detect_full(cv2.imread(img))
kps = res[0]['kps']; hand_pts = (kps[LEFT_HAND, 2] > 0.3).sum()
```

## 六、已知问题与优化方向

### 6.1 显存（已优化，2026-08-26）

- **现状**：只常驻 openpose CN（depth/canny 懒加载）+ `torch.cuda.empty_cache()` + `expandable_segments`，静态 19.5→14.3GB，4 场景端到端含超分全部通过（峰值 24GB 内）
- 风险：推理峰值接近 24GB，长期运行建议观察；后续可 CN 也走 8bit

### 6.2 机器内存 15GB 限制

- 100 个 torch compile_worker 子进程（每次 import torch spawn 32 个）吃光内存 → **整机 OOM 重启**
- 已用 `TORCHINDUCTOR_COMPILE_THREADS=1` 规避；多服务同时加载模型时注意内存峰值
- 10331（GPU0）与 10335（GPU1）同时加载会内存紧张，建议串行启动

### 6.3 性能

- ControlNet 生成 373s/张（P40 无 Tensor Core + fp16 CN + 8bit unet），可接受但慢
- 后续可测：CN 8bit 量化提速、`--scenes` 批量预热 kernel 缓存

### 6.4 未落地环节（ShowDoc 方案 2/5）

| 方案 | 状态 | 说明 |
|---|---|---|
| 手部 LoRA | 未做 | SDXL 生态手部 LoRA 稀缺（HandFix 等为 SD1.5），权重 0.3-0.6 为辅助增益 |
| HandRefiner | 未做 | 需 SD1.5+ControlNet 专用节点；现用 inpaint 局部重绘替代，效果接近 |

## 七、踩坑记录（详见 PITFAILLOG）

1. **diffusers 0.39 弃用 `load_in_8bit=` 直接传参**：pipeline 层需 `PipelineQuantizationConfig(quant_backend="bitsandbytes_8bit", quant_kwargs={"load_in_8bit": True}, components_to_quantize=[...])`；模型层（ControlNetModel）不接受 PipelineQuantizationConfig
2. **DWPose yolox onnx 必须标准 grid+stride decode**：直接 resize+阈值检不出人（conf 全低）；标准 YOLOX decode 后 conf 0.95
3. **RTMPose onnx 是 SimCC 双输出**（simcc_x/simcc_y），非 heatmap；置信度 = (max_x+max_y)/2，位置 argmax 后 ÷split_ratio(2)
4. **仿射变换不能带 mmpose 老版 `scale*200` 因子**：会放大 250 倍裁剪窗口导致人物缩成背景点、SimCC 全低置信度；以 sd-webui-controlnet 新版 `get_warp_matrix`（无 200 因子 + `_fix_aspect_ratio`）为准
5. **onnx 输入类型必须 float32**：ImageNet 归一化（float64 mean/std）后需 `.astype(np.float32)`
6. **8bit 下 VAE 在 CPU**：generator 必须 `torch.Generator("cpu")`（同 10331 服务经验）
7. **多 ControlNet 传 None 会报 TypeError**：未激活的 CN 也要传黑图 + scale=0
8. **gated 仓库**：inpaint 用 `diffusers/` 镜像，`stabilityai/` 原仓库需 token
9. **diffusers 0.39 的 from_pretrained 传 controlnet 实例 list 会挂**：`controlnet=[cn]` 在量化路径下 ModuleList 报「pipeline is not a Module subclass」；**必须显式 `MultiControlNetModel([cn])` 后传实例**，且调用时 image 传 `[cond_img]`、scale 传 `[strength]`（MultiControlNetModel 约定）
10. **load_cn 返回类型陷阱**：懒加载函数若返回重建后的 cn_pipe，启动处当 ControlNetModel 用会连环报错；启动路径直接 `ControlNetModel.from_pretrained` 加载
