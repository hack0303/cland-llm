# Case 001：图生视频首条出片（AnimateDiff · P40 · 2s 轻微动画）

> 日期：2026-08-15 · 管线：SDXL(10331) → ComfyUI I2V(10337) → MP4
> 目的：验证 AnimateDiff 图生视频管线全链路跑通，标定 P40 实测性能
> 结论先行：**512×512 16 帧 20 步 = 140s 出片**，主体保持 + 渐进轻微动画；1024×1024 慢 6.6 倍（~15.5min）且超 SD1.5 训练分布，不推荐

## 一、输入

### 1. 参考图（SDXL 生成）

| 项 | 值 |
|---|---|
| 文件 | `/mnt/data/ai_workspace/outputs/sdxl_1786771048_42.png` |
| 内容 | 猫宇航员挥手（月球场景） |
| Prompt | `a cute cat astronaut on the moon, waving one paw, cinematic lighting, vibrant colors` |
| 参数 | steps=30 · seed=42 · SDXL base 1.0（GPU 0） |

### 2. 生成参数

| 参数 | 值 | 说明 |
|---|---|---|
| 帧数 | 16 | context_length=16（AnimateDiff 安全区） |
| 步数 | 20 | euler / normal |
| CFG | 7.0 | — |
| denoise | 0.8 | img2img 首帧保留强度 |
| fps | 8.0 | 16 帧 = 2s |
| IPAdapter weight | 0.7 | 身份保持（STANDARD preset） |
| seed | 42（1024²）/ 43（512²） | — |

## 二、执行过程

```
SDXL 文生图 (10331) → 参考图 1024²
  → ComfyUI (10337, GPU 1)：
      LoadImage → VAEEncode → RepeatLatentBatch(16)
      → ADE_StandardUniformContextOptions(16,1,4)
      → ADE_AnimateDiffLoaderWithContext(mm_sd_v15_v2, sqrt_linear)
      → IPAdapterUnifiedLoader(STANDARD) → IPAdapterAdvanced(0.7)
      → KSampler(20步, euler, denoise 0.8) → VAEDecode
      → SaveAnimatedWEBP → PIL 逐帧提取 → ffmpeg → MP4
```

### 实测耗时与资源

| 分辨率 | 耗时 | 备注 |
|---|---|---|
| **512×512** | **140s（2m20s）** | SD1.5 训练分辨率，**推荐甜点** |
| 1024×1024 | ~15.5min（45s/步） | 超训练分布，慢 6.6 倍，不推荐 |

- GPU 1 常驻显存：3.1GB（SD1.5 + 运动模块 + IPAdapter + CLIP-ViT-H）
- 与 SDXL（GPU 0，13GB）并行无冲突

## 三、输出与质量评估

| 文件 | 规格 | 大小 |
|---|---|---|
| `outputs_video/i2v_cat_paw_512.mp4` | 512×512 · 16 帧 @8fps · 2s | 137KB |
| `outputs_video/i2v_cat_paw.mp4` | 1024×1024 · 16 帧 @8fps · 2s | 348KB |

### 帧差异量化（64×64 降采样平均 RGB 差）

| 对比 | 差异 | 解读 |
|---|---|---|
| 参考图 vs 首帧 | 16.7 | denoise 0.8 重绘，主体保留（IPAdapter 0.7 生效） |
| 首帧 vs 中帧 | 1.7 | 运动启动 |
| 首帧 vs 尾帧 | 3.3 | 渐进式轻微动画 |

- **运动特征**：幅度小、渐进、无跳变——符合 AnimateDiff 第一代定位（待机/呼吸/飘动类）
- **身份保持**：IPAdapter 0.7 + img2img 首帧，猫主体轮廓稳定

## 四、结论与经验

1. **512×512 是 P40 甜点**：140s/条（vs 1024 的 15.5min），游戏待机动画场景建议固定 512
2. **AnimateDiff 输出 = 轻微动画**：2s@8fps 适合占位/待机/飘动；要大幅度运动/24fps 需 LTX-Video（写实向）或帧插值（RIFE 16→48 帧）
3. **可调参数方向**（后续批量验证）：denoise 0.7~0.9 控运动量、IPAdapter weight 0.5~0.8 控身份、帧数上限（24/32 帧显存与稳定性待测）、运动幅度可用 motion_scale
4. **管线已固化**：`inference/i2v/generate.py --image X.png --frames 16 --steps 20` 一条命令出片

## 五、复现

```bash
# 1. SDXL 出参考图（如已存在可跳过）
curl -X POST http://127.0.0.1:10331/generate -H "Content-Type: application/json" \
  -d '{"prompt":"a cute cat astronaut on the moon, waving one paw, cinematic lighting","steps":30,"seed":42}'

# 2. 图生视频（512 甜点，~2.5 分钟）
cd /mnt/data/ai_workspace/cland-llm
python3 inference/i2v/generate.py --image <参考图> --prefix case001 --seed 43
# 输出：/mnt/data/ai_workspace/outputs_video/case001.mp4
```
