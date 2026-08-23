# CLand-LLM Inference 推理服务

本目录存放生产环境实际运行的模型推理服务，全部基于本机硬件（2× Tesla P40 24GB，Pascal sm_61）调优。

## 目录结构

```
inference/
├── sdxl/                    # SDXL 文本生图（diffusers）
│   ├── server.py            # 常驻 API 服务（FastAPI，端口 10331）
│   ├── gen.py               # 命令行单次生成
│   └── README.md
├── embedding/              # 向量检索（bge-m3/nomic 11435，reranker 10304，bench + 归档）
└── gemma/                   # Gemma-4 系列（llama.cpp / vllm）
    ├── gemma4.py            # 26B UD-Q4_K_M OpenAI 兼容服务（0.3.20 自动模板 + DSL 转换层，端口 10303）
    ├── gemma4_1.py          # 31B IQ2_M DSL 解析服务
    ├── gemma4_hf.py         # HuggingFace 加载版
    ├── my_api.py            # 流式 API（131K 上下文）
    ├── openai_api.py        # OpenAI 兼容流式 API（80K 上下文）
    ├── vllm_run.sh          # vllm serve 启动脚本（AWQ-8bit 双卡，端口 10303）
    ├── litellm_config.yaml  # litellm 网关配置（转发 10303）
    └── ARCHIVE-2026-08-23-gemma4-ollama.md  # 诊断归档：ollama 不认 gemma4 架构、迁移计划
```

## 环境说明（重要）

| 环境 | torch | P40 (sm_61) | 用途 |
|---|---|---|---|
| base | 2.7.1+cu118 | ✅ 支持 | SDXL（diffusers） |
| gemma_env | 2.10.0+cu128 | ❌ 不支持 | Gemma（llama.cpp 自带 CUDA 内核，不走 torch） |

- **P40 无 Tensor Core**：BF16 降速、无 FlashAttention；torch≥2.9 (cu128) 已砍掉 sm_70 以下支持
- Gemma 走 `llama_cpp`（自带 sm_50+ 内核），gemma_env 的 torch 版本不影响
- 模型权重统一存放 `/mnt/data/ai_workspace/models/`（不入库）

## 服务端口

| 服务 | 端口 | 启动 |
|---|---|---|
| SDXL 生图 | 10331 | 手动 |
| gemma4 转换服务（llama.cpp 26B） | 10303 | 手动 nohup（见 gemma/ARCHIVE） |
| reranker 精排（bge-reranker-v2-m3） | 10304 | 手动 `start_reranker.sh` |
| ollama 主实例（对话） | 11434 | systemd 自启 |
| ollama embedding（bge-m3/nomic） | 11435 | systemd 自启 |
