# SDXL 图像生成服务

基于 `stabilityai/stable-diffusion-xl-base-1.0` 的文本生图服务，适配 Tesla P40。

## 依赖（base 环境）

```bash
pip install torch==2.7.1+cu118 diffusers accelerate safetensors fastapi uvicorn pillow
```

⚠️ 必须用 base 环境（torch 2.7.1+cu118 支持 Pascal sm_61）；gemma_env 的 torch 2.10 (cu128) 不支持 P40。

## 常驻服务

```bash
# 启动（模型加载约 2-10 分钟，之后常驻显存）
python3 server.py --host 0.0.0.0 --port 10331

# 健康检查
curl http://127.0.0.1:10331/health

# 生成图片
curl -X POST http://127.0.0.1:10331/generate -H "Content-Type: application/json" \
  -d '{"prompt":"a cat astronaut on the moon, cinematic","steps":30,"seed":42}'
```

### 请求参数

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| prompt | str | 必填 | 提示词（英文效果最佳） |
| negative_prompt | str | blurry, low quality... | 负面提示词 |
| steps | int | 30 | 采样步数 |
| width / height | int | 1024 | 尺寸（16 的倍数） |
| seed | int | 42 | 随机种子 |
| guidance_scale | float | 7.5 | CFG 强度 |

### 响应

```json
{"image": "/mnt/data/ai_workspace/outputs/sdxl_xxx.png", "seconds": 70.9, "vram_gb": 10.45, "seed": 42}
```

## 命令行单次生成

```bash
python3 gen.py --prompt "a red fox in snowy forest" --steps 30 --seed 42 --output out.png
```

## 实测性能（单张 P40）

- 1024×1024 / 30 步：**~71s**（2.2s/步）
- 峰值显存：**10.45GB**（24GB 卡余量充足，可双服务并行）
- 已启用 `attention_slicing` + `vae_slicing`（Pascal 稳定性配置）

## 模型路径

- 权重：`/mnt/data/ai_workspace/models/stable-diffusion-xl-base-1.0`（7.3GB，fp32 官方版）
- 输出：`/mnt/data/ai_workspace/outputs/`

## 备注

- 8bit 加速（bitsandbytes）在 P40 上实测无增益（71s vs 73s），保持 FP16 即可
- 更高画质可换 FLUX.1-schnell（4 步蒸馏）或 Qwen-Image 系列，需 4bit 量化
