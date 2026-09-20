# TODO 跟踪看板（cland-llm 主待办）

> 遵循 GFM 任务列表规范（todoprompt skill）：`- [ ]` 待办 | `- [x]` 已完成 | `- [/]` 执行中 | `- [!]` 阻塞
> 各子模块专项看板：`TODO-hand-pipe.md`（生图管线）· `TODO-image2video.md`（图生视频）

---

# 目标：本机推理服务（LLM/生图/视频）并发能力与稳定性提升

## 一、gemma4 服务端并发支持（n_parallel / 显式多 slot）

- **状态**：open · 优先级：medium · 来源：cland-reverser 性别推断任务（2026-08-30）
- **现象**：
  - 客户端并发 4 请求的干净受控测试（短 prompt，max_tokens=64）相对串行 **2.52x 提速**
  - 但生产长请求（20 条/批，prompt ~2500 tokens，max_tokens=320）下，客户端 4 worker 并发吞吐与串行**基本持平**（0.24s/条 vs 0.25s/条，仅 +8%）
  - 结论：服务端对长请求实为**排队串行**（疑似 n_parallel=1 内部互斥），短请求的"提速"为假象（可能是 batch decode 或请求交错调度的边界效应）
- **根因猜测**：
  - `gemma4.py` 中 `Llama(...)` 未传 `n_parallel`（llama-cpp-python 默认 1）
  - `chat_endpoint` 为 `async def`，内部直接同步调用 `llm.create_chat_completion`（阻塞事件循环；仅 C 层释放 GIL 时才部分并行）
- **方案**：
  - A `Llama(..., n_parallel=4)`：真并行需每 slot 独立 KV cache，需评估显存（当前 2×P40 各 ~10GB/24GB，16384 ctx）；llama-cpp-python 0.3.20 支持
  - B 同步调用移入 `asyncio.to_thread` + 请求级互斥，避免阻塞事件循环（不提升吞吐，仅保事件循环健康）
  - C 先做 A 的显存评估（n_parallel=2/4 对比 KV 占用与吞吐），再定
- **验收**：客户端并发 4 请求（生产参数：20 条/批、max_tokens=320）吞吐 ≥ 串行 × 3；`/v1/models` 顺带实现（健康检查别再 404）
- **注意**：改动需重启服务（加载 ~200s）；cland-reverser 的 `gender_llm.py` 已支持 `--workers` 参数，服务端就绪后可直接受益
