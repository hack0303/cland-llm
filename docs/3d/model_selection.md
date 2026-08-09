# 3D 模型选型文档（C-Land 本机部署）

> 硬件：2× Tesla P40 (24GB, Pascal sm_61) + 15GB RAM
> 现状：GPU 0 已被 SDXL 占用（13GB，端口 10331）；**3D 服务规划部署在 GPU 1（空闲），使用独立端口**

## 一、硬件约束（选型铁律）

| 约束 | 影响 |
|---|---|
| P40 无 Tensor Core | BF16 降速，无 flash-attn 可用（编译失败），依赖 xformers/慢路径 |
| torch 需 cu118 | gemma_env 的 cu128 torch 不支持 sm_61，**必须 base 环境** |
| 内存仅 15GB | 7B+ 模型 fp16 加载紧张；优先 ≤3B 模型 |
| 双卡 48GB 总显存 | GPU 0 已用 13GB，3D 服务独享 GPU 1 (24GB) |

## 二、候选模型对比（开源可自部署）

| 模型 | 参数量 | 输入 | 显存需求 | 质量 | 安装难度 | 关键风险 |
|---|---|---|---|---|---|---|
| **Hunyuan3D-2.5** | 形状 2.6B + 纹理 1.3B | 文/图 → 3D | ~10-12GB | ⭐⭐⭐⭐⭐ | 中 | 依赖多（tome/dpm-solver），中文/英文均佳 |
| **TripoSG** | 1.5B | 图 → 3D | ~8GB | ⭐⭐⭐⭐ | 低 | ✅ **已部署**（GPU1:10332） |
| **SF3D** | 0.7B | 图 → 3D | ~5GB | ⭐⭐⭐⭐ | 低（pip 即用） | 单视图，背面质量一般；**秒级出图** |
| **TripoSR** | 0.7B | 图 → 3D | ~4GB | ⭐⭐⭐ | 低 | 老，仅占位级 |
| **TRELLIS** | 7B | 图 → 3D | 20-24GB | ⭐⭐⭐⭐⭐ | 高 | flash-attn 在 P40 编译失败，需 xformers 降级；GPU 1 单卡刚好 |
| **Unique3D** | ~2B | 图 → 3D | ~14GB | ⭐⭐⭐⭐ | 中 | 多视图扩散，依赖 xformers |
| Shap-E / Point-E | 1B | 文/图 → 3D | ~4GB | ⭐⭐ | 低 | 已过时，仅对比 |

## 三、决策树

```
需求：游戏/产品 3D 资产
├─ 文生 3D（无参考图）────────→ Hunyuan3D-2.5（✅ 唯一成熟选择，待部署）
├─ 图生 3D 高质量 ────────────→ TripoSG（✅ 已部署 GPU1:10332）
├─ 图生 3D 快速占位（秒级）────→ SF3D
└─ 低模原创（已有成熟流程）────→ Blender 程序化（cland-game-asset-gen E 路线）
```

## 四、推荐结论（2026-08-09 更新：已落地 TripoSG）

### ✅ 已部署：TripoSG（图生 3D）

- 用户最终决策：部署 TripoSG（1.5B rectified-flow，图生 3D）
- 服务：GPU 1 / 端口 10332，常驻显存 4.15GB，详见 `inference/triposg/README.md`
- 全流程 ~26min/图（推理 7.5min + SDF→mesh 提取 18min）

### 📌 待补：Hunyuan3D-2.5（文生 3D）

- TripoSG 仅支持**图生** 3D；文生 3D 场景（无参考图）需要 Hunyuan3D-2.5 补齐
- 图生 3D 质量同为开源第一梯队
- 显存 ~10-12GB，GPU 1 (24GB) 富余，可与后续服务共存
- 官方支持 4090 级消费卡推理，P40 算力略低但架构兼容
- 两阶段：形状生成（2.6B transformer）→ 纹理生成（1.3B），均可独立部署

### 🥈 备选：SF3D（快速通道）

- 秒级出图，适合批量占位、快速验证构图
- pip 安装 5 分钟，风险最低

### ❌ 不选理由

| 模型 | 否决原因 |
|---|---|
| TRELLIS | flash-attn 依赖在 P40 无法编译（xformers 降级效果未知）；24GB 显存贴线无余量 |
| Unique3D | 与 Hunyuan3D-2.5 同梯队但社区/文档较弱 |
| TripoSR | 质量过时 |

## 五、部署记录（已完成）

| 项 | 实际值 |
|---|---|
| GPU | GPU 1（CUDA_VISIBLE_DEVICES=1）✅ |
| 端口 | 10332（未复用 10331）✅ |
| 环境 | triposg_env（py3.10 + torch 2.6.0+cu118）✅ |
| 模型路径 | `/mnt/data/ai_workspace/models/TripoSG`（7.5G）✅ |
| 服务形态 | 常驻 FastAPI（`inference/triposg/server.py`）✅ |
| 输出目录 | `/mnt/data/ai_workspace/outputs3d/` ✅ |
| 实测显存 | 常驻 4.15GB / 峰值 9.6GB（24GB 卡余 14GB）✅ |
| 实测耗时 | 50 步推理 7.5min + 网格提取 18min ≈ 26min/图 |

### 端口分配总表

| 端口 | 服务 | GPU |
|---|---|---|
| 10303 | vllm（gemma AWQ） | 双卡 |
| 10331 | SDXL 生图 | 0 |
| **10332** | **Hunyuan3D-2.5（规划）** | **1** |

### 依赖清单（Hunyuan3D-2.5）

```
torch 2.7.1+cu118  diffusers(≥0.31)  transformers  accelerate
tome  dpm-solver  sentencepiece  fastapi  uvicorn
xformers（P40 无 flash-attn，官方降级路径）
```
