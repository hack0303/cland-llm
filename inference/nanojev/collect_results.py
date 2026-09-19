#!/usr/bin/env python3
"""汇总 inference/nanojev/runs/ 下的基准结果 → runs/benchmark_summary.json（紧凑表）。

只保留关键指标（attempts/collisions/food/latency/memory），避免把完整轨迹 JSON 入库。
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def load(name):
    path = RUNS / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def maze_summary(d):
    s = d["summary"]
    return {
        "elapsed_seconds": round(d["elapsed_seconds"], 1),
        "goal_completed": s["goal_completed"],
        "attempts": s["attempts"],
        "collisions": s["collisions"],
        "successful_moves": s["successful_moves"],
        "predict_calls": d["execution"]["predict_calls"],
        "atomic_questions": s["atomic_questions"],
        "atomic_accuracy": round(s["atomic_accuracy"], 4),
        "atomic_brier": round(s["atomic_brier"], 4),
        "episode_ids": d.get("selected_episode_ids") or [e["id"] for e in d["episodes"]],
    }


def snake_summary(d):
    s = d["summary"]
    return {
        "elapsed_seconds": round(d["elapsed_seconds"], 1),
        "episodes": s["episodes"],
        "outcomes": s["outcomes"],
        "survival_steps": s["survival_steps"],
        "food_collected": s["food_collected"],
        "mean_food_score": s["mean_food_score"],
        "model_decisions": s["model_decisions"],
        "predict_calls": d["execution"]["predict_calls"],
        "showcase_case": next({k: e[k] for k in ("id", "outcome", "food_score", "survival_steps")}
                              for e in d["episodes"] if e["id"] == "snake:showcase:12:61005"),
    }


def main():
    out = {
        "generated_by": "collect_results.py",
        "maze_50x50": {},
        "snake_cohort8": {},
        "service_inproc": {},
        "service_http": {},
        "von_http": {},
    }
    for key, name in (("nanojev_fp32", "maze_nanojev_local_atomic_fp32.json"),
                      ("nanojev_bf16", "maze_nanojev_local_atomic_bf16.json"),
                      ("qwen_fp32", "maze_qwen_native_fp32.json"),
                      ("qwen_bf16", "maze_qwen_native_bf16.json")):
        d = load(name)
        if d:
            out["maze_50x50"][key] = maze_summary(d)
    for key, name in (("nanojev_fp32", "snake_nanojev_games_gold_fp32.json"),
                      ("nanojev_bf16", "snake_nanojev_games_gold_bf16.json"),
                      ("qwen_fp32", "snake_qwen_native_fp32.json")):
        d = load(name)
        if d:
            out["snake_cohort8"][key] = snake_summary(d)
    for key, name in (("nanojev_fp32", "bench_inproc_fp32.json"),
                      ("nanojev_bf16", "bench_inproc_bf16.json")):
        d = load(name)
        if d:
            out["service_inproc"][key] = {
                "load_seconds": round(d["load_seconds"], 1),
                "parameter_count": d["parameter_count"],
                "memory_after_load_gb": {k: round(v, 3) for k, v in d["memory_after_load_gb"].items()},
                "memory_peak_gb": {k: round(v, 3) for k, v in d["memory_peak_gb"].items()},
                "benchmarks": {k: {
                    "median_ms": round(v["latency_ms"]["median"], 1),
                    "p95_ms": round(v["latency_ms"]["p95"], 1),
                    "questions_per_s": round(v["throughput"]["questions_per_s"], 1),
                } for k, v in d["benchmarks"].items()},
            }
    d = load("bench_http_10338_fp32.json")
    if d:
        out["service_http"]["nanojev_fp32"] = {
            "median_ms": round(d["latency_ms"]["median"], 1),
            "p95_ms": round(d["latency_ms"]["p95"], 1),
            "requests_per_s": round(d["throughput"]["requests_per_s"], 2),
        }
    d = load("bench_von_10339_fp32.json")
    if d:
        out["von_http"]["cuda_fp32"] = {k: {
            "median_ms": round(v["latency_ms"]["median"], 1),
            "min_ms": round(v["latency_ms"]["min"], 1),
        } for k, v in d["benchmarks"].items()}
    d = load("bench_von_10339_bf16.json")
    if d:
        out["von_http"]["cuda_bf16"] = {k: {
            "median_ms": round(v["latency_ms"]["median"], 1),
            "min_ms": round(v["latency_ms"]["min"], 1),
        } for k, v in d["benchmarks"].items()}
    (RUNS / "benchmark_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
