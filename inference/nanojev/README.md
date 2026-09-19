# NanoJev / Von —— 开源 System One 决策模型本机部署

本目录是 **Jev（TypeSafe System One）开源复现** 的本地部署与基准复现工程（工单 base/cland-crawler#11）。
对象仓：`TianyuCodings/NanoJev`（MIT，Qwen3-0.6B + decision heads）与 `wfzyx/von`（Apache-2.0，ModernBERT-NLI，TypeSafe `/v1/systemone` 协议对等）。

> 本机硬件：2× Tesla P40 24GB（sm_61，无 Tensor Core / 无原生 BF16）。
> PyTorch：2.7.1+cu118（P40 最后一个可用档位；torch≥2.9 cu128 已删 sm_70 以下内核）。
> Transformers：5.14.1（上游记录 torch 2.14 / transformers 5.17，见「兼容性」节）。

## 一、目录结构

```
inference/nanojev/
├── NanoJev/                 # 上游源码（git clone --depth 1，未改动）
├── von/                     # 上游源码 + checkpoints/von-modernbert-rlcd → 本地权重软链
├── venv/                    # 独立 venv（--system-site-packages 复用 base 的 cu118 torch）
├── sitecustomize.py         # torch 2.7 兼容 shim（torch._native.triton_utils no-op）
├── patches/von-p40-fp32.patch  # von dtype 显式覆盖（VON_DTYPE；P40 默认 fp32，setup.sh 自动应用）
├── download_variant.py      # NanoJev variant 权重下载（HF 镜像）
├── download_von.py          # von-1.0 权重下载（HF 镜像）
├── download_qwen.py         # 未调 Qwen3-0.6B 基线（固定 revision，默认 HF 缓存）
├── download_dataset.py      # NanoJev-Data games_v4/arcade 小包（官方 cohort + 回放）
├── start_server.sh          # NanoJev 常驻决策服务（默认 :10338）
├── start_von.sh             # Von 常驻服务（默认 :10339，/v1/systemone）
├── bench_service.py         # in-process 延迟/显存/吞吐基准
├── bench_http.py            # NanoJev HTTP 端到端基准
├── bench_von.py             # Von HTTP 基准
├── run_bench_maze.sh        # 50×50 maze showcase 复现（NanoJev / Qwen / both）
├── run_bench_snake.sh       # Snake 8 局 cohort 复现（NanoJev / Qwen / both）
├── collect_results.py       # runs/*.json → runs/benchmark_summary.json（紧凑结果表）
├── runs/                    # 实测输出（原始 JSON 含完整轨迹，不入库；summary 入库）
└── logs/                    # 服务与基准日志（不入库）
```

权重（不入库）：

| 模型 | 路径 | 大小 | 备注 |
|---|---|---|---|
| NanoJev local_atomic_seed17 | `/mnt/data/ai_workspace/models/nanojev/NanoJev/variants/local_atomic_seed17` | 2.39 GB | 50×50 maze showcase |
| NanoJev games_gold_seed17 | `.../variants/games_gold_seed17` | 2.39 GB | Snake showcase |
| 未调 Qwen3-0.6B | `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B` | 1.2 GB | 基线，固定 revision `c1899de2` |
| Von-1.0 | `/mnt/data/ai_workspace/models/von/von-1.0` | 1.58 GB | 底座 tasksource/ModernBERT-large-nli，T=1.0367 |
| NanoJev-Data games_v4/arcade | `/mnt/data/ai_workspace/models/nanojev/NanoJev-Data/games_v4/arcade` | <1 MB | 官方 8 局 cohort + 6 条回放 |

## 二、环境与安装（可复现）

```bash
cd /mnt/data/ai_workspace/cland-llm/inference/nanojev

# 1) venv：复用 base conda 的 torch 2.7.1+cu118（P40 可用），不污染 SDXL 环境
/home/alice/miniconda3/bin/python -m venv --system-site-packages venv

# 2) von 可编辑安装（其余依赖 base 已有：transformers/safetensors/numpy/fastapi/uvicorn/pydantic/httpx）
venv/bin/pip install -e ./von
```

### 网络实测（2026-09-20）

- GitHub **https 克隆不通 → 走 SSH**：`git clone git@github.com:TianyuCodings/NanoJev.git`
- HuggingFace 直连不通 → **`HF_ENDPOINT=https://hf-mirror.com`**（脚本内已默认设置）

### 权重下载

```bash
HF_ENDPOINT=https://hf-mirror.com venv/bin/python download_variant.py local_atomic_seed17   # maze
HF_ENDPOINT=https://hf-mirror.com venv/bin/python download_variant.py games_gold_seed17      # snake
HF_ENDPOINT=https://hf-mirror.com venv/bin/python download_von.py                            # von-1.0
HF_ENDPOINT=https://hf-mirror.com venv/bin/python download_qwen.py                           # Qwen3-0.6B 基线
HF_ENDPOINT=https://hf-mirror.com venv/bin/python download_dataset.py                        # 官方 cohort/回放
```

> 长下载可能被镜像断连（`httpx.RemoteProtocolError`）；huggingface_hub 支持断点续传，**重跑同命令即可**。

## 三、常驻服务

| 服务 | 端口 | 启动脚本 | 协议 |
|---|---|---|---|
| NanoJev 决策服务（maze 默认） | **10338** | `./start_server.sh [variant] [port] [precision]` | `POST /api/evaluate`（boolean/choice/score，返回完整分布） |
| Von 决策服务 | **10339** | `./start_von.sh [port] [device]` | `POST /v1/systemone`（TypeSafe drop-in：noul/choice/score） |

两者均默认 **fp32**（P40 无原生 BF16；模拟 bf16 慢 1.5–1.9× 且影响决策边界精度，见「兼容性」节）。

```bash
./start_server.sh local_atomic_seed17 10338 fp32     # NanoJev（服务 + 官方 web 回放 UI 同端口）
./start_von.sh 10339 cuda                            # Von（首次请求懒加载，约 45s）
```

### NanoJev 调用示例

```bash
curl http://127.0.0.1:10338/api/health
# {"ready": true, "model_loaded_once": true, "provider_calls": 0}

curl http://127.0.0.1:10338/api/evaluate -H 'Content-Type: application/json' -d '{
  "states": [{
    "id": "room1",
    "state": "The living room is 29 degrees. The target is 24 degrees. The window is closed and someone is home.",
    "questions": {
      "action": {"type": "choice",
                 "instructions": "Choose the action that most directly lowers the room temperature.",
                 "criteria": {"cool": "Turn on air conditioning", "light": "Turn on the lights", "wait": "Keep the current settings"}},
      "occupied": {"type": "boolean", "instructions": "Someone is home."},
      "heat": {"type": "score",
               "instructions": "Classify how far the room temperature exceeds the target.",
               "criteria": ["At or below target", "Above target by at most 3 degrees", "Above target by more than 3 degrees"]}
    }
  }]
}'
# → answers.*.probabilities（完整分布）+ p_true / value / score，execution.forward_passes=1（零输出解码）
```

浏览器打开 `http://127.0.0.1:10338/` 可看官方三系统回放 UI（maze/snake/side-by-side）。

### Von 调用示例（TypeSafe 协议对等）

```bash
curl http://127.0.0.1:10339/health

curl -X POST http://127.0.0.1:10339/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "von-1.0.0",
  "state": {"error": "Disk volume /var/log at 98% capacity."},
  "questions": {"requires_intervention": {"type": "noul",
      "instructions": "Does this disk space condition require operational intervention?"}}
}'
# → {"answers": {"requires_intervention": {"type": "noul", "noul": 0.5078}}, "usage": {...}}

# 多原语 fan-out（choice + noul + score 单次请求）
curl -X POST http://127.0.0.1:10339/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "von-1.0.0",
  "state": {"ticket_id": "INC-4091", "customer_tier": "enterprise",
            "message": "Payment gateway reports timeout on charge authorizations. Urgent."},
  "questions": {
    "intent": {"type": "choice", "instructions": "What is the operational nature of this ticket?",
               "criteria": {"payment_failure": "Failures processing charges, gateway timeouts, credit card declines",
                            "access_issue": "Login, SSO, authentication, or permission errors"}},
    "is_urgent": {"type": "noul", "instructions": "Does the request require immediate SLA intervention?"},
    "severity": {"type": "score", "instructions": "Rate the incident severity.",
                 "criteria": ["Low", "Medium", "High", "Critical"]}
  }
}'
```

## 四、基准复现（实测，2026-09-20，P40 fp32）

### 4.1 50×50 maze showcase（`maze:ood:50:24310922`）

```bash
BENCH_GPU=1 ./run_bench_maze.sh nanojev fp32   # NanoJev，27.5s
BENCH_GPU=1 ./run_bench_maze.sh qwen fp32      # 未调 Qwen3-0.6B 对照，167.2s
BENCH_GPU=1 ./run_bench_maze.sh nanojev bf16   # 精度对照
BENCH_GPU=1 ./run_bench_maze.sh qwen bf16
```

| 系统 | attempts | collisions | 结果 | 耗时 | 前向次数 | 原子题准确率 / Brier |
|---|---:|---:|---|---:|---:|---:|
| **NanoJev fp32（本次）** | **244** | **36** | ✅ goal | 27.5s | 199 | 85.93% / 0.107 |
| NanoJev bf16（本次） | 243 | 35 | ✅ goal | 42.5s | 199 | 86.06% / 0.107 |
| NanoJev 官方记录（A100 bf16） | 244 | 36 | ✅ goal | — | — | — |
| 未调 Qwen3-0.6B fp32（本次） | 4134 | 1768 | ✅ goal | 167.2s | 1137 | 43.34% / 0.298 |
| 未调 Qwen3-0.6B bf16（本次） | 4265 | 1843 | ✅ goal | 266.3s | 1212 | 44.74% / 0.295 |
| 官方记录的 Qwen 基线 | 4726 | 2044 | ✅ goal | — | — | — |

- NanoJev **fp32 与官方记录逐项一致（244/36）**；bf16 差 1 次碰撞，属数值路径微分歧，结论不变。
- Qwen 基线与官方记录差 ~10%（4134 vs 4726）：同 episode、同控制器、同算法；差异来自 P40（fp32/模拟 bf16）× transformers 5.14 与 A100 bf16 × 5.17 的数值差异，且探索是顺序决策，一次分歧会放大轨迹差。**量级结论稳定：Qwen 需要 ~17–19× 尝试、~49–57× 碰撞才能到达目标**。

### 4.2 Snake 8 局 cohort（`games_v4/arcade/cohort.jsonl`，greedy，seed 17，horizon 256）

```bash
BENCH_GPU=1 ./run_bench_snake.sh nanojev fp32  # 40.0s
BENCH_GPU=1 ./run_bench_snake.sh qwen fp32     # 未调 Qwen 对照
```

| 系统 | trapped / survived | 总 food | 平均 food | showcase `12:61005` | 结果 vs 官方记录 |
|---|---:|---:|---:|---|---|
| **NanoJev fp32（本次）** | 4 / 4 | 168 | 21.0 | 27 food / 256 步 survival | **逐 case 8/8 完全一致** |
| NanoJev bf16（本次） | 4 / 4 | 168 | 21.0 | 27 food / 256 步 survival | 8/8 一致 |
| 未调 Qwen3-0.6B fp32（本次） | 5 / 3 | 190 | 23.75 | 25 food / 211 步 trapped | **逐 case 8/8 完全一致** |

（Jev 官方 API 为付费资源，按纪律未调用；README 记录的 Jev 30 food 仅作对照，不参与本次复现。）

### 4.3 服务延迟 / 显存 / 吞吐（P40）

| 指标 | NanoJev fp32 | NanoJev bf16 | Von fp32 | Von bf16 |
|---|---:|---:|---:|---:|
| 权重加载 | 46.2s | 32.5s | 首次请求 ~45s（懒加载） |
| 参数显存（实测） | 2.39 GB allocated / 2.55 GB 峰值 | 2.39 / 3.60 GB 峰值 | 约 1.8 GB | 约 1.0 GB |
| 单 state 4 题（maze 决策） | **136.7 ms**（29.3 q/s） | 200.5 ms（20.0 q/s） | — |
| 4 states 16 题批处理 | 452.8 ms（35.3 q/s） | 651.6 ms（24.6 q/s） | — |
| 混合 3 题（choice+bool+score） | 82.4 ms（36.4 q/s） | 118.3 ms（25.4 q/s） | — | — |
| Von 单 noul / 3 题 fan-out | — | — | **32.6 / 93.1 ms** | 61.1 / 150.2 ms |
| HTTP 端到端（4 题） | 134.4 ms p50 / 7.44 req/s | — | — |

**结论：P40 上 fp32 比 bf16（模拟）快 1.5–1.9×，且与官方数值口径更接近，默认用 fp32。**
（P40 无原生 BF16；torch 2.7 的 `is_bf16_supported()` 对本卡返回 True 是「模拟」语义，实际 bf16 走慢路径。）

### 4.4 生成紧凑结果表

```bash
venv/bin/python collect_results.py   # → runs/benchmark_summary.json
```

## 五、兼容性说明（必读）

1. **torch 版本**：上游记录 `torch==2.14.0`；P40 无法使用（torch≥2.9 砍 sm_70 以下）。本机用 2.7.1+cu118，功能实测等效。
2. **`torch._native.triton_utils` shim**：上游脚本（`evaluate_model_edges_maze.py` 等）设置 `disable_native_triton=True` 时导入该模块。torch 2.7 无此模块 → `sitecustomize.py` 提供 no-op 等价实现（2.7 没有需要关闭的 native Triton overrides），运行时经 `PYTHONPATH` 自动加载（`start_server.sh` / `run_bench_*.sh` 已内置）。
3. **数值敏感性**：NanoJev 的 maze/snake 结果对精度不敏感（fp32/bf16 与原记录一致或差 1）；未调 Qwen 对精度敏感（fp32/bf16 轨迹不同），复现对照时需固定 `--precision`。
4. **服务限制**（上游 `serve_decisions.py`，未改动）：单请求 ≤32 states / 96 questions / 256 candidate paths；HTTPServer 单线程。
5. **Von 服务注意**：`/health` 恒返回 ok（不做模型状态检查）；模型在首个推理请求时加载；noul 判定为 entailment 两路 softmax，属模型行为，长语义先验题可能偏中性（经典 NLI 判定 sanity 正常）。
6. **Von dtype 补丁（重要）**：von 自动 dtype 在 CUDA 上优先 bf16（`is_bf16_supported()`）；P40 模拟 bf16 会使决策边界精度回退——官方测试 `test_fanout.py` 的 `is_blocking` noul 为 0.498（期望 >0.5），CPU/fp32 通过而 CUDA/bf16 失败。`patches/von-p40-fp32.patch` 增加 `VON_DTYPE` 覆盖，`setup.sh` 自动应用，`start_von.sh` 默认 `VON_DTYPE=fp32`。
   - 补丁后本机 CUDA 跑 von 自带测试套件：**23 passed / 1 deselected**（deselected 为可选依赖 trio 用例）。
   - 补丁后服务端复测：`is_blocking` noul **0.5092**、severity **1.11**、category=storage（用例期望 >0.5 / >1.0 / storage，全部满足）。

## 六、许可与出处

- NanoJev：MIT — https://github.com/TianyuCodings/NanoJev · 权重 https://huggingface.co/C-Tianyu/NanoJev · 固定 release `4a19595e`
- Von：Apache-2.0 — https://github.com/wfzyx/von · 权重 https://huggingface.co/wfzyx/von-1.0
- 未调 Qwen3-0.6B：Apache-2.0 — revision `c1899de289a04d12100db370d81485cdf75e47ca`
- 本部署未调用任何付费 API；全部推理在本机 P40 完成。
