# Gemma-4 服务归档：2026-08-23 诊断与简化

> 归档内容：gemma4 在 P40 上的完整诊断链、gemma4.py 简化改造、ollama 后续迁移计划。
> 关联踩坑记录：PITFAILLOG.md 2026-08-23 三条（[模型] 前缀）。

## 一、诊断结论（为什么 gemma4 "指令不兼容 OpenAI API"）

1. **Gemma-4 原生 DSL 与 OpenAI API 是两套协议**，引擎必须做双向翻译：
   - 渲染层（OpenAI messages → `<|turn>` 模板）：llama-cpp-python ≥0.3.20 **自动支持**（Jinja2ChatFormatter 读 GGUF 内嵌官方模板，实测 `chat_template.default`）
   - 解析层（`<|tool_call>call:name{}` → OpenAI `tool_calls` JSON）：**引擎未实现，必须手写转换层**
2. **ollama 0.18.2 报 `unknown model architecture: 'gemma4'`**：版本太老（内核无 gemma4 架构），**不是 P40 不兼容**：
   - cuda_v12 runner（CUDA 12.8）在 535 驱动上 `cudaGetDeviceCount=2` ✅，libggml-cuda.so 含 sm_61
   - cuda_v13（CUDA 13）才需要 r580+ 驱动
   - 最新 ollama 内核 llama.cpp b10488 源码保留 `61-virtual`（Pascal）
3. **P40 上 torch 不可用**（cu128 无 sm_61 内核），gemma 全链路走 llama.cpp 自带内核，不依赖 torch。

## 二、当前方案（2026-08-23 起，启动脚本方式）

| 项 | 值 |
|---|---|
| 服务 | `gemma4.py`（简化版） |
| 端口 | 10303，OpenAI 兼容 API（/v1/chat/completions，非流式 + SSE 流式） |
| 模型 | `models/gemma-4-26B-A4B-it-UD-Q4_K_M/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`（16GB Q4） |
| 环境 | gemma_env（llama-cpp-python 0.3.20） |
| 启动 | `nohup /home/alice/miniconda3/envs/gemma_env/bin/python gemma4.py > /tmp/gemma4-server.log 2>&1 &` |
| 加载耗时 | ~180s（26B Q4，n_ctx=16384，flash_attn） |

**gemma4.py 简化版要点**（原版备份：`gemma4.py.bak`）：
- 删除手写 `GEMMA_4_JINJA`（引擎自动用 GGUF 官方模板）
- 保留转换层：thought → `reasoning_content`；DSL → `tool_calls` JSON；有 tool_calls 时 content 置 None
- 非流式：stop 序列防自导自演（`<|tool_response>`）
- 流式：缓冲状态机（跨 token 标签 + `<|"|>` 转义陷阱），匹配完整 tool_call 后物理截断
- 已修正原脚本模型路径 bug

**已验证**：基础对话 / 工具调用（非流式+流式）/ 多轮工具闭环 / 中文输出，全通过。

## 三、原有 systemd 服务（ollama 双实例，勿动）

两个服务均为 root 运行、enabled + active，重启自动拉起：

### 1. `ollama.service` — 主实例（GPU0 / 11434）

| 配置 | 值 | 说明 |
|---|---|---|
| ExecStart | `/usr/local/bin/ollama serve` | 版本 0.18.2（2026-03） |
| OLLAMA_HOST | `0.0.0.0` | 全端口监听 |
| CUDA_VISIBLE_DEVICES | `0` | 只用 0 号 P40 |
| OLLAMA_CUDA_FORCE_ALLOW_OLD_ARCH | `1` | **P40 关键**：允许 Pascal 老架构 |
| OLLAMA_LLM_LIBRARY | `ggml-cuda` | 强制 CUDA 后端 |
| OLLAMA_FLASH_ATTENTION | `1` | |
| OLLAMA_KV_CACHE_TYPE | `q8_0` | KV 量化省显存 |
| OLLAMA_NUM_PARALLEL | `1` | 串行推理（大模型） |
| PATH / LD_LIBRARY_PATH | CUDA 12.4 路径 | |

已装模型（`/root/.ollama`）：`command-r:latest`（18.7GB）、`llama3.1:8b`（4.9GB）、`gemma4-e4b-test`（8.2GB，**2026-08-23 测试导入的残留，可 `ollama rm` 清理**）。

### 2. `ollama-embedding.service` — 嵌入实例（GPU1 / 11435）

| 配置 | 值 | 说明 |
|---|---|---|
| ExecStart | `/usr/local/bin/ollama serve` | 第二实例 |
| OLLAMA_HOST | `0.0.0.0:11435` | 端口隔离 |
| OLLAMA_MODELS | `/var/lib/ollama-embedding` | 独立存储，避免文件锁冲突 |
| CUDA_VISIBLE_DEVICES | `1` | 只用 1 号 P40 |
| OLLAMA_FLASH_ATTENTION | `1` | |
| OLLAMA_NUM_PARALLEL | `4` | Embedding 批量并发优化（fp16，无需 KV 量化） |

已装模型：`nomic-embed-text:latest`（0.3GB）、`bge-m3:latest`（1.2GB）。

> 注意：ollama.service 注释把 P40 写成 "Kepler" 是笔误，P40 是 **Pascal (sm_61)**。

## 四、ollama 迁移计划（待执行）

- ollama v0.32.15（1.3GB）后台下载中 → `/tmp/ollama-new.tar.zst`（GitHub 直连，速度慢）
- 下载完成后升级：`tar --zstd -xf` 替换 `/usr/local/bin/ollama` 与 `/usr/local/bin/lib/`（两个 systemd 服务会自动重启）
- 升级后 `ollama pull gemma4:26b-a4b-it-q4_`（或 `ollama create` 导入本地 GGUF）
- 预期：基础对话 + OpenAI API 直接可用；**工具调用 DSL 解析大概率仍缺失** → 复用本目录转换层思路或继续走 10303 自建服务
- 升级风险点：确认新版的 cuda_v12 runner 仍含 sm_61（llama.cpp b10488 源码保留 `61-virtual`）；若默认选了 cuda_v13（需 r580+ 驱动）需显式指定 cuda_v12

## 五、文件清单

| 文件 | 说明 |
|---|---|
| `gemma4.py` | 当前生产版本（简化 + 流式） |
| `gemma4.py.bak` | 原版（手写 Jinja 模板版） |
| `gemma4_1.py` / `openai_api.py` / `my_api.py` / `gemma4_hf.py` | 历史版本（31B IQ2 / 流式 / 131K 上下文 / HF 加载） |
| `litellm_config.yaml` | litellm 网关配置（转发 10303，drop_params） |
| `vllm_run.sh` | vllm 双卡 AWQ 启动脚本（备用） |

> **实验/demo 归属地约定（2026-08-23）**：gemma-4 工具调用实验代码（`gemma_4_client.py`、`example_tool_*.py`、`benchmark_tool_calling.py`）在 `PythonTest` 仓库（`git@github.com:hack0303/PythonTest.git`，`~/work/pyproj/PythonTest/src/llm/client/`），后续相关 demo 放那边；本目录只放生产服务与归档。
