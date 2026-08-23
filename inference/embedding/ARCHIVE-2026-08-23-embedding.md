# Embedding 向量模型研究归档：2026-08-23

> 归档内容：ollama embedding 双模型现状、P40 实测性能、中文语义质量对比（重要结论）、研究资产索引。
> 关联服务：`ollama-embedding.service`（GPU1 / 11435，详见 `../gemma/ARCHIVE-2026-08-23-gemma4-ollama.md` 第三节）。

## 一、服务与模型现状

**服务**：`ollama-embedding.service`（root，enabled + active，独立存储 `/var/lib/ollama-embedding`）
- 端口 **11435**（`OLLAMA_HOST=0.0.0.0:11435`，局域网可访问，PythonTest 里用 `http://192.168.1.14:11435`）
- GPU 隔离：`CUDA_VISIBLE_DEVICES=1`（1 号 P40，与主实例 0 号卡物理隔离）
- 调优：`OLLAMA_FLASH_ATTENTION=1`、`OLLAMA_NUM_PARALLEL=4`（批量并发）

**Reranker 服务（2026-08-23 新增）**：`reranker_server.py`（FastAPI + FlagReranker，bge-reranker-v2-m3）
- 端口 **10304**，接口 `POST /rerank`（`{query, documents[], top_k?}` → 分数降序）、`GET /health`
- GPU1（`CUDA_VISIBLE_DEVICES=1`），启动：`./start_reranker.sh`（手动，无自启，幂等），日志 `/tmp/reranker-server.log`
- 模型启动时加载一次常驻；评分特性见 alice-research-hub §7（v2 无 sigmoid，分数范围大，不可跨模型比较）

| 模型 | 架构 | 参数量 | 维度 | 上下文 | 量化 | 大小 |
|---|---|---|---|---|---|---|
| `nomic-embed-text:latest` | nomic-bert | 137M | **768**（可截断 512/256/128/64） | 2048 | F16 | 274MB |
| `bge-m3:latest` | bert | 566.7M | **1024**（可截断 512/256） | 8192 | F16 | 1.2GB |

**API**（OpenAI 不兼容，用 ollama 原生）：`POST /api/embed`，返回 `{"embeddings": [[...]]}`。

## 二、P40 实测性能（GPU1，2026-08-23）

| 模型 | 单条（热） | 10 条 | 50 条 | 吞吐 |
|---|---|---|---|---|
| nomic-embed-text | 0.40s | 1.80s | 8.47s | **5.9 条/s** |
| bge-m3 | 0.76s | 5.22s | 24.36s | **2.1 条/s** |

- 冷启动（模型加载）额外约 3s / 15s
- 复测脚本：`bench_embedding.py`（本目录，改模型名即可）

## 三、中文语义质量对比（重要结论）

测试样本：P40 硬件文档相似句 vs 天气无关句（复测见 `bench_embedding.py`）：

| 模型 | 相似句 cos | 无关句 cos | 区分度 | 检索排序 |
|---|---|---|---|---|
| **bge-m3** | 0.769 | 0.409 | ✅ 差距 0.36 | ✅ query→相关 0.61 > 无关 0.41 |
| nomic-embed-text（裸调） | 0.566 | 0.700 | ❌ **倒挂** | ❌ |
| nomic-embed-text（+search_document: 前缀） | 0.719 | 0.788 | ❌ **仍倒挂** | ❌ 无关 0.58 > 相似 0.52 |

**结论**：
1. **中文语义检索必须用 bge-m3**；nomic-embed-text 是英文为主训练的模型，中文区分度倒挂，加官方 task 前缀也无法拯救
2. PythonTest 里的"漏斗检索"方案（nomic 256 维粗排 + bge-m3 1024 维精排）**粗排层建议换成 bge-m3 截断 256 维**（bge-m3 原生支持 Matryoshka 截断，同模型一致性更好），而不是 nomic
3. 截断后必须 L2 重归一化（已有代码实现，见研究资产）

## 四、研究资产索引（已有代码，勿重复造）

| 位置 | 内容 |
|---|---|
| `~/work/pyproj/PythonTest/src/embedding/` | ① 01: bge-m3 + Chroma 基础 RAG；② 02: nomic + Matryoshka 截断 256 维 + 重归一化；③ 03 + `local/`: 漏斗检索工程化（256 粗排 + 1024 精排）、GPU 余弦相似度 |
| `~/miniconda3/envs/dev_bge` | 本地 bge 研究环境：FlagEmbedding 1.2.1、sentence-transformers 2.6.1、chromadb 0.5.23、langchain 全家桶 |
| `models/bge-reranker-v2-m3/` | Reranker 模型（本地），FlagEmbedding 生态，可做精排第二层 |
| 本目录 | `bench_embedding.py` 实测脚本 + 本文档 |

> **实验/demo 归属地约定（2026-08-23）**：`PythonTest` 仓库（`git@github.com:hack0303/PythonTest.git`，本地 `~/work/pyproj/PythonTest/`）是实验与 demo 的正式归属地，后续新 demo 一律放那边。本目录只存生产化代码与实测归档。embedding 三阶段检索 demo 若升级（bge-m3 粗排 + reranker 精排），代码放 `PythonTest/src/embedding/`。

## 五、后续建议

1. **统一用 bge-m3**：粗排（截断 256）+ 精排（1024），删掉 nomic 依赖（或仅留英文场景）
2. 精排用 **10304 reranker 服务**（已部署常驻）或 dev_bge 直连 FlagEmbedding；demo 里精排步骤可改调 `POST /rerank` 替代进程内加载
3. 批量灌库用 `OLLAMA_NUM_PARALLEL=4` 已开，实测 50 条 24s 可接受；若需更高吞吐可直连 FlagEmbedding（dev_bge）走 GPU 批处理
4. Chroma/HNSW 索引层选型已跑通（chromadb 0.5.23），保持

## 六、本目录文件清单（2026-08-23 更新）

| 文件 | 说明 |
|---|---|
| `ARCHIVE-2026-08-23-embedding.md` | 本文档（调研归档） |
| `bench_embedding.py` | embedding 吞吐/质量复测脚本 |
| `reranker_server.py` | **reranker 常驻服务（10304）** |
| `start_reranker.sh` | reranker 手动启动脚本（无自启） |
