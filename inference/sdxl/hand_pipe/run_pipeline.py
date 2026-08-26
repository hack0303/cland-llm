#!/usr/bin/env python3
"""工业管线端到端验证 — 4 个手脚崩坏典型场景
用法: python3 run_pipeline.py [--seed 42] [--skip 4]
"""
import argparse
import json
import time
import urllib.request

API = "http://127.0.0.1:10335"

SCENES = [
    {
        "name": "peace_sign",
        "prompt": "a man showing a peace sign with his hand toward camera, "
                  "shallow depth of field, photorealistic",
    },
    {
        "name": "holding_cup",
        "prompt": "a woman holding a coffee cup with both hands, warm cafe "
                  "light, photorealistic close-up",
    },
    {
        "name": "chin_hands",
        "prompt": "close-up portrait of a young woman, hands resting on her "
                  "chin, studio lighting, photorealistic",
    },
    {
        "name": "dancer",
        "prompt": "a ballet dancer mid-pose, arms extended, elegant hand "
                  "positions, stage spotlight, photorealistic",
    },
]


def call(path, body, timeout=3600):
    req = urllib.request.Request(f"{API}{path}",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenes", default="0,1,2,3")
    ap.add_argument("--skip-inpaint", action="store_true")
    args = ap.parse_args()
    report = {}
    for i in args.scenes.split(","):
        s = SCENES[int(i)]
        print(f"\n=== {s['name']} (seed {args.seed}) ===", flush=True)
        t0 = time.time()
        r = call("/pipeline", {
            "prompt": s["prompt"],
            "seed": args.seed,
            "use_cn": True,
            "use_inpaint": not args.skip_inpaint,
            "use_upscale": True,
            "steps": 30,
        })
        for st in r["steps"]:
            print(f"  step{st['step']} {st['name']}: {st['image']} "
                  f"({st['t']}s)", flush=True)
        report[s["name"]] = r
        print(f"  TOTAL: {r['total_seconds']}s", flush=True)
    with open("/mnt/data/ai_workspace/outputs/hand_pipe/pipeline_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nreport -> /mnt/data/ai_workspace/outputs/hand_pipe/pipeline_report.json")


if __name__ == "__main__":
    main()
