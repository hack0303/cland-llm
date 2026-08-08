# SDXL 图像生成服务使用手册

> 基于 `stabilityai/stable-diffusion-xl-base-1.0` 的常驻文本生图服务，部署于 2× Tesla P40 机器，代码见 `inference/sdxl/`。

## 一、快速开始

### 1. 启动服务

```bash
cd /mnt/data/ai_workspace/cland-llm/inference/sdxl
python3 server.py --host 0.0.0.0 --port 10331
```

- 首次启动需加载模型（约 2~10 分钟，CPU 量化阶段 GPU 空闲属正常）
- 加载完成后模型常驻显存（约 13GB），之后所有请求零加载开销
- 建议用 nohup 后台运行：`nohup python3 server.py --port 10331 > /tmp/sdxl_server.log 2>&1 &`

### 2. 验证服务

```bash
curl http://127.0.0.1:10331/health
# → {"status":"ok","model":"stable-diffusion-xl-base-1.0"}
```

### 3. 生成第一张图

```bash
curl -X POST http://127.0.0.1:10331/generate -H "Content-Type: application/json" \
  -d '{
    "prompt": "A red fox in a snowy pine forest at golden hour, photorealistic, sharp focus, soft bokeh",
    "steps": 30,
    "seed": 42
  }'
```

响应示例：

```json
{"image": "/mnt/data/ai_workspace/outputs/sdxl_1786232455_42.png", "seconds": 70.9, "vram_gb": 10.45, "seed": 42}
```

图片输出到 `/mnt/data/ai_workspace/outputs/`，文件名格式 `sdxl_<时间戳>_<seed>.png`。

## 二、API 参数详解

`POST /generate` 请求体（JSON）：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `prompt` | string | **必填** | 提示词。英文效果最佳，可含构图/光线/风格描述 |
| `negative_prompt` | string | `blurry, low quality, distorted, watermark` | 负面提示词，排除不想要的内容 |
| `steps` | int | 30 | 采样步数。20~40 区间质量/速度平衡；越多越精细但非线性收益 |
| `width` / `height` | int | 1024 / 1024 | 尺寸。**必须为 16 的倍数**，否则自动缩放 |
| `seed` | int | 42 | 随机种子。同 seed 同参数可复现同一张图 |
| `guidance_scale` | float | 7.5 | CFG 强度。越高越贴近提示词，过高会过饱和/伪影（建议 5~9） |

## 三、CLI 单次生成（不开服务）

```bash
python3 gen.py \
  --prompt "a cat astronaut on the moon, cinematic" \
  --negative "blurry, low quality" \
  --steps 30 --width 1024 --height 1024 \
  --seed 42 --output cat.png
```

适合批量脚本内调用；高频场景仍建议走常驻服务（省掉每次 ~2 分钟加载）。

## 四、提示词技巧

```text
# 结构模板：主体 + 场景 + 光线 + 风格 + 质量词
[主体], [环境/场景], [光线], [风格/媒介], [质量修饰]

# 示例
a samurai cat in a cyberpunk alley, neon lights, rain, cinematic lighting, 4k, sharp focus
ancient chinese palace in autumn, misty mountains, golden hour, oil painting style, highly detailed
```

- 质量词：`photorealistic, highly detailed, sharp focus, 8k, masterpiece`
- 风格词：`oil painting, watercolor, anime, cyberpunk, cinematic, isometric`
- 负面词兜底：`blurry, low quality, distorted, watermark, extra fingers, deformed`

## 五、实测性能（单张 Tesla P40）

| 配置 | 耗时 | 峰值显存 |
|---|---|---|
| 1024×1024 / 30 步 | **~71s**（2.2s/步） | 10.45GB |
| 1024×1024 / 20 步 | ~48s | ~10.5GB |
| 512×512 / 30 步 | ~30s | ~9GB |

- 单卡 24GB 余量充足，可与其他服务（如 gemma）分卡并行
- 已启用 `attention_slicing` + `vae_slicing`（Pascal 架构稳定性配置）

## 六、环境与依赖（重要）

```bash
# 必须使用 base 环境！gemma_env 的 torch 2.10 (cu128) 不支持 P40 (sm_61)
conda activate base
pip install torch==2.7.1+cu118 diffusers accelerate safetensors fastapi uvicorn pillow
```

| 环境 | torch | P40 支持 | 用途 |
|---|---|---|---|
| base | 2.7.1+cu118 | ✅ sm_61 | SDXL（本服务） |
| gemma_env | 2.10.0+cu128 | ❌ 最低 sm_70 | Gemma llama.cpp（自带内核） |

## 七、故障排查

| 现象 | 原因 | 解决 |
|---|---|---|
| `Cannot generate a cpu tensor from a generator of type cuda` | 8bit 模式下 VAE 在 CPU，generator 需匹配 | 用 `torch.Generator("cpu")`（server.py 已处理） |
| 启动慢（分钟级） | 首次 CPU 量化/权重读取 | 正常，常驻后无此问题 |
| `no kernel image is available` | torch 版本不支持 sm_61 | 换 base 环境（cu118） |
| 出图全黑/花屏 | 精度问题或显存不足 | 检查 seed/显存，确认 `attention_slicing` 已启用 |
| 服务端口占用 | 重复启动 | `lsof -i :10331` 查占用进程 |

## 八、相关文件

| 路径 | 说明 |
|---|---|
| `inference/sdxl/server.py` | 常驻 API 服务（FastAPI） |
| `inference/sdxl/gen.py` | CLI 单次生成 |
| `/mnt/data/ai_workspace/models/stable-diffusion-xl-base-1.0/` | 模型权重（7.3GB） |
| `/mnt/data/ai_workspace/outputs/` | 出图目录 |
