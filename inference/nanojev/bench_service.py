#!/usr/bin/env python3
"""NanoJev checkpoint 推理基准：延迟 / 显存 / 吞吐（in-process，不经 HTTP）。

用法:
  venv/bin/python bench_service.py --checkpoint-dir /path/to/variant [--precision fp32]
      [--iterations 20] [--batch-states 4] [--output runs/bench_xxx.json]

说明:
  - 输入使用 maze controller 的真实模板（evaluate_composed_maze.render_local_request）：
    50×50 maze 上的 5×5 局部窗口 + 4 个 boolean 方向安全题。
  - 另加 runbook 混合题（choice/boolean/score）与 N-states 批处理。
  - 延迟为单次 predict()（可能含多个 batch，取决于 batch_questions）。
  - 显存取 torch.cuda.memory_allocated / max_memory_allocated（进程内精确）。
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "NanoJev" / "scripts"))

import torch  # noqa: E402
from predict_toy_decisions import DecisionPredictor  # noqa: E402
from evaluate_composed_maze import render_local_request  # noqa: E402


def synthetic_maze_state():
    """从 50×50 showcase episode 取几何，构造一个局部窗口请求。"""
    episode = json.loads(
        (HERE / "NanoJev" / "results" / "side_by_side_maze_episode.jsonl").read_text().splitlines()[0]
    )
    state = episode["initial_state"]
    return state


def walkable_near_center(maze, count):
    """取中心附近 count 个可走格子（避开墙与目标点冲突）。"""
    size = maze["size"]
    walls = {tuple(w) for w in maze["walls"]}
    c = size // 2
    cells = []
    for r in range(c - 3, c + 4):
        for col in range(c - 3, c + 4):
            if (r, col) not in walls and 0 <= r < size and 0 <= col < size:
                cells.append([r, col])
            if len(cells) >= count:
                return cells
    raise RuntimeError("near-center walkable cells not found")


def build_payloads():
    maze = synthetic_maze_state()
    cells = walkable_near_center(maze, 5)
    center = dict(maze)
    center["position"] = cells[0]
    r1 = render_local_request(center, 5)
    single = {"states": [{"id": "bench:single", **r1}]}

    batch_states = []
    for i, pos in enumerate(cells[1:5]):
        row = dict(maze)
        row["position"] = pos
        req = render_local_request(row, 5)
        batch_states.append({"id": f"bench:batch{i}", **req})
    batch = {"states": batch_states}

    mixed = {
        "states": [{
            "id": "bench:mixed",
            "state": "The living room is 29 degrees. The target is 24 degrees. The window is closed and someone is home.",
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": "Choose the action that most directly lowers the room temperature.",
                    "criteria": {
                        "cool": "Turn on air conditioning",
                        "light": "Turn on the lights",
                        "wait": "Keep the current settings",
                    },
                },
                "occupied": {"type": "boolean", "instructions": "Someone is home."},
                "heat": {
                    "type": "score",
                    "instructions": "Classify how far the room temperature exceeds the target.",
                    "criteria": ["At or below target", "Above target by at most 3 degrees",
                                 "Above target by more than 3 degrees"],
                },
            },
        }]
    }
    return {"single_maze_4q": single, "batch_maze_4states_16q": batch, "mixed_3q": mixed}


def time_payload(engine, payload, iterations, batch_questions=0):
    lat = []
    questions = sum(len(s["questions"]) for s in payload["states"])
    paths = 0
    for s in payload["states"]:
        for q in s["questions"].values():
            if q["type"] == "boolean":
                paths += 1
            else:
                paths += len(q["criteria"])
    result = None
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = engine.predict(payload, batch_questions=batch_questions, temperature=1.0)
        lat.append(time.perf_counter() - t0)
    ex = result["execution"]
    return {
        "iterations": iterations,
        "states": len(payload["states"]),
        "questions": questions,
        "candidate_paths": paths,
        "latency_ms": {
            "min": min(lat) * 1000,
            "median": statistics.median(lat) * 1000,
            "mean": statistics.mean(lat) * 1000,
            "p95": sorted(lat)[max(0, int(len(lat) * 0.95) - 1)] * 1000,
            "max": max(lat) * 1000,
        },
        "throughput": {
            "states_per_s": len(payload["states"]) / statistics.median(lat),
            "questions_per_s": questions / statistics.median(lat),
            "candidate_paths_per_s": paths / statistics.median(lat),
        },
        "forward_passes_per_call": ex.get("forward_passes"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True)
    a = p.parse_args()

    t0 = time.perf_counter()
    engine = DecisionPredictor(a.checkpoint_dir, precision=a.precision, device_name=a.device)
    load_s = time.perf_counter() - t0

    torch.cuda.reset_peak_memory_stats()
    params = sum(p.numel() for p in engine.model.parameters())
    payloads = build_payloads()

    # warmup
    engine.predict(payloads["single_maze_4q"], temperature=1.0)

    report = {
        "checkpoint_dir": a.checkpoint_dir,
        "precision": a.precision,
        "device": torch.cuda.get_device_name(0),
        "parameter_count": params,
        "load_seconds": load_s,
        "memory_after_load_gb": {
            "allocated": torch.cuda.memory_allocated() / 1e9,
            "reserved": torch.cuda.memory_reserved() / 1e9,
        },
        "benchmarks": {},
    }
    for name, payload in payloads.items():
        report["benchmarks"][name] = time_payload(engine, payload, a.iterations)
    report["memory_peak_gb"] = {
        "max_allocated": torch.cuda.max_memory_allocated() / 1e9,
        "max_reserved": torch.cuda.max_memory_reserved() / 1e9,
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
