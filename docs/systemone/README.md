# System One 开源复现（NanoJev / Von）本机部署与基准

> 2026-09-20。工单 base/cland-crawler#11。部署全文见 `inference/nanojev/README.md`，结果为 P40 实测。

## 一、结论

| 项 | 结论 | 置信度 |
|---|---|---|
| **NanoJev 本机可跑且与官方记录一致** | 50×50 maze：244 attempts / 36 collisions / goal（官方 244/36，**逐项一致**）；Snake 8 局：4 trapped / 4 survived、mean food 21.0（官方同，**逐 case 8/8 一致**） | 高（实测 + 官方记录） |
| **未调 Qwen3-0.6B 对照成立** | maze 需 4134 attempts / 1768 collisions（≈17–19× / 49–57×）；Snake mean food 23.75 但 showcase case 先 trapped（25 food/211 步 vs NanoJev 27 food/256 步存活），**逐 case 与官方一致** | 高 |
| **Von 可作为 TypeSafe 协议 drop-in** | `POST /v1/systemone`（noul/choice/score）：单题 ~33ms、3 题 fan-out ~93ms（fp32），显存 ~1.8GB；自带测试套件 CUDA 23 passed（需 fp32 补丁，见下） | 高（自测 / 官方测试套件）+ 中（质量口径：经典 NLI sanity 正常，长语义先验题偏中性） |
| **P40 上应使用 fp32** | fp32 比 bf16（torch 模拟）快 ~1.5×，且 maze 数字与官方记录完全一致；默认 fp32 | 高 |
| **Jev 本体仍不可自部署** | 官方闭源仅 API；本次复现为开源 System One 近似，不等于 Jev | 高（见 cland-research 报告 #8） |

## 二、服务（常驻，手动启动）

| 服务 | 端口 | 启动 | 协议 |
|---|---|---|---|
| NanoJev（maze 变体 `local_atomic_seed17`） | 10338 | `inference/nanojev/start_server.sh` | `POST /api/evaluate` + 官方回放 UI |
| Von-1.0（ModernBERT-NLI + T=1.0367） | 10339 | `inference/nanojev/start_von.sh` | `POST /v1/systemone`（TypeSafe 对等） |

```bash
./start_server.sh local_atomic_seed17 10338 fp32   # 或 games_gold_seed17 起 Snake 变体
./start_von.sh 10339 cuda
```

调用示例与环境/权重/网络踩坑详见 `inference/nanojev/README.md`。

## 三、实测结果（P40，2026-09-20）

### 50×50 maze showcase（`maze:ood:50:24310922`）

| 系统 | attempts | collisions | 结果 | 耗时 | 原子准确率 |
|---|---:|---:|---|---:|---:|
| NanoJev fp32 | **244** | **36** | ✅ goal | 27.5s | 85.93% |
| NanoJev bf16 | 243 | 35 | ✅ goal | 42.5s | 86.06% |
| 官方记录（A100 bf16） | 244 | 36 | ✅ goal | — | — |
| 未调 Qwen fp32 | 4134 | 1768 | ✅ goal | 167.2s | 43.34% |
| 未调 Qwen bf16 | 4265 | 1843 | ✅ goal | 266.3s | 44.74% |
| 官方记录的 Qwen 基线 | 4726 | 2044 | ✅ goal | — | — |

### Snake 8 局 cohort（greedy，seed 17，horizon 256）

| 系统 | trapped/survived | 总/平均 food | showcase `12:61005` |
|---|---|---|---|
| NanoJev fp32 | 4 / 4 | 168 / 21.0 | 27 food，256 步存活（vs 官方 8/8 一致） |
| 未调 Qwen fp32 | 5 / 3 | 190 / 23.75 | 25 food，211 步 trapped（vs 官方 8/8 一致） |

### 服务性能

| 指标 | NanoJev fp32 | 备注 |
|---|---:|---|
| 加载 / 显存 | 46.2s / 2.39GB allocated（峰值 2.55GB） | 596,250,498 参数 |
| 单 state 4 题 | 136.7ms p50，29.3 q/s | HTTP 端到端 134.4ms，7.44 req/s |
| 4 states 16 题 | 452.8ms，35.3 q/s | 一次 backbone 前向 |
| Von 单 noul / 3 题 fan-out | 32.6ms / 93.1ms（fp32；bf16 对照 61/150ms） | 显存 ~1.8GB(fp32)，首请求懒加载 ~45s |

## 四、复现命令（摘要）

```bash
cd /mnt/data/ai_workspace/cland-llm/inference/nanojev
BENCH_GPU=1 ./run_bench_maze.sh both fp32     # maze：NanoJev + Qwen 对照
BENCH_GPU=1 ./run_bench_snake.sh both fp32    # snake：同样两组
venv/bin/python collect_results.py            # 汇总 runs/benchmark_summary.json
```

原始轨迹 JSON 在 `inference/nanojev/runs/`（未入库）；汇总表 `benchmark_summary.json` 入库。

## 五、局限 / 待验证

1. **Qwen 基线差 ~10%**（4134 vs 官方 4726）：P40 fp32 × transformers 5.14 vs A100 bf16 × 5.17 的数值差异在顺序探索中放大；量级结论稳定，单点数字口径需注明环境。
2. **原子题准确率口径**：实测 85.93% 是单 episode 的 796 题；README 的 77.84%（test）/76.56%（50×50 OOD）是全 split 口径，不可直接对比。
3. **Von 质量未经其 authored144 基准复验**：仅做经典 NLI sanity 与其自带测试套件（CUDA fp32 23 passed）；自评 91.23% 未在本机复现（基准集未随仓发布）。另，von 自动 dtype 在 P40 会选 bf16 模拟，导致决策边界精度回退（test_fanout noul 0.498 vs fp32 0.509）——已用 `patches/von-p40-fp32.patch` + `VON_DTYPE=fp32` 修复。
4. **中文场景未测**：两模型训练语料均为英文为主；CJK 表现待我方语料自测。
5. **`serve_decisions.py` 为单线程 HTTPServer**、限制 32 states/96 questions/256 paths；生产化需换 FastAPI/并发方案。
6. **torch 2.7 vs 上游 2.14**：功能实测等效，但上游 `torch._native` API 依赖本机 shim（`sitecustomize.py`），后续升级需复核。

## 六、出处

- NanoJev（MIT，★803）：https://github.com/TianyuCodings/NanoJev · 权重 https://huggingface.co/C-Tianyu/NanoJev（release `4a19595e`）· 数据集 `C-Tianyu/NanoJev-Data`
- Von（Apache-2.0，★17）：https://github.com/wfzyx/von · 权重 https://huggingface.co/wfzyx/von-1.0
- 未调 Qwen3-0.6B：revision `c1899de289a04d12100db370d81485cdf75e47ca`
- Jev 本体调研（#8）：`cland-research/docs/jev-决策模型调研.md`
