#!/usr/bin/env python3
"""NanoJev HTTP 服务端到端基准：对常驻服务 /api/evaluate 发 N 次请求。

用法:
  venv/bin/python bench_http.py --port 10338 [--iterations 20] [--output runs/bench_http.json]
"""
import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

PAYLOAD = {
    "states": [{
        "id": "bench:http",
        "state": ("Agent coordinate: (25,25); zero-based row and column. Rows increase south; "
                  "columns increase east. The local window is centered on A. '#': wall; '.': open; "
                  "'A': agent; 'X': outside.\nLocal map:\n.....\n..#..\n..A..\n.....\n....."),
        "questions": {
            "clear_north": {"type": "boolean",
                            "instructions": "If the agent attempts one cell north (row minus one, same column), will the destination be inside the maze and not a wall? Use the local map: '.' and 'A' are traversable, '#' is a wall, and 'X' is outside the maze.",
                            "criteria": {"true": "The destination of the one-cell north attempt is inside the maze and traversable.",
                                         "false": "The destination of the one-cell north attempt is a wall or outside the maze."}},
            "clear_east": {"type": "boolean",
                           "instructions": "If the agent attempts one cell east (same row, column plus one), will the destination be inside the maze and not a wall? Use the local map: '.' and 'A' are traversable, '#' is a wall, and 'X' is outside the maze.",
                           "criteria": {"true": "The destination of the one-cell east attempt is inside the maze and traversable.",
                                        "false": "The destination of the one-cell east attempt is a wall or outside the maze."}},
            "clear_south": {"type": "boolean",
                            "instructions": "If the agent attempts one cell south (row plus one, same column), will the destination be inside the maze and not a wall? Use the local map: '.' and 'A' are traversable, '#' is a wall, and 'X' is outside the maze.",
                            "criteria": {"true": "The destination of the one-cell south attempt is inside the maze and traversable.",
                                         "false": "The destination of the one-cell south attempt is a wall or outside the maze."}},
            "clear_west": {"type": "boolean",
                           "instructions": "If the agent attempts one cell west (same row, column minus one), will the destination be inside the maze and not a wall? Use the local map: '.' and 'A' are traversable, '#' is a wall, and 'X' is outside the maze.",
                           "criteria": {"true": "The destination of the one-cell west attempt is inside the maze and traversable.",
                                        "false": "The destination of the one-cell west attempt is a wall or outside the maze."}},
        },
    }]
}


def post(port, payload, timeout=120):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/evaluate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=10338)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    health = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/api/health", timeout=10).read())
    post(a.port, PAYLOAD)  # warmup
    lat = []
    first = None
    for _ in range(a.iterations):
        t0 = time.perf_counter()
        r = post(a.port, PAYLOAD)
        lat.append(time.perf_counter() - t0)
        first = first or r["execution"]
    report = {
        "port": a.port,
        "health": health,
        "iterations": a.iterations,
        "states": len(PAYLOAD["states"]),
        "questions": sum(len(s["questions"]) for s in PAYLOAD["states"]),
        "latency_ms": {
            "min": min(lat) * 1000,
            "median": statistics.median(lat) * 1000,
            "mean": statistics.mean(lat) * 1000,
            "p95": sorted(lat)[max(0, int(len(lat) * 0.95) - 1)] * 1000,
            "max": max(lat) * 1000,
        },
        "throughput": {"requests_per_s": 1 / statistics.median(lat)},
        "server_execution": first,
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
