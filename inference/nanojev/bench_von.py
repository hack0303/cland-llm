#!/usr/bin/env python3
"""Von HTTP 服务基准：单题延迟 + 多题 fan-out 延迟（无付费调用）。"""
import json, statistics, time, urllib.request, argparse
from pathlib import Path

P = argparse.ArgumentParser()
P.add_argument("--port", type=int, default=10339)
P.add_argument("--iterations", type=int, default=10)
P.add_argument("--output", required=True)
a = P.parse_args()

SINGLE = {
    "model": "von-1.0.0",
    "state": {"error": "Disk volume /var/log at 98% capacity."},
    "questions": {"requires_intervention": {"type": "noul",
        "instructions": "Does this disk space condition require operational intervention?"}},
}
FANOUT = {
    "model": "von-1.0.0",
    "state": {"ticket_id": "INC-4091", "customer_tier": "enterprise",
              "message": "Payment gateway reports timeout on charge authorizations. Urgent."},
    "questions": {
        "intent": {"type": "choice", "instructions": "What is the operational nature of this ticket?",
                   "criteria": {"payment_failure": "Failures processing charges, gateway timeouts, credit card declines",
                                "access_issue": "Login, SSO, authentication, or permission errors"}},
        "is_urgent": {"type": "noul", "instructions": "Does the request require immediate SLA intervention?"},
        "severity": {"type": "score", "instructions": "Rate the incident severity.",
                     "criteria": ["Low", "Medium", "High", "Critical"]},
    },
}

def call(payload, timeout=300):
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/systemone",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

report = {"port": a.port, "benchmarks": {}}
for name, payload in (("single_noul", SINGLE), ("fanout_3q", FANOUT)):
    call(payload)  # warmup
    lat = []
    last = None
    for _ in range(a.iterations):
        t0 = time.perf_counter(); last = call(payload); lat.append(time.perf_counter() - t0)
    report["benchmarks"][name] = {
        "iterations": a.iterations,
        "latency_ms": {"min": min(lat)*1000, "median": statistics.median(lat)*1000,
                       "mean": statistics.mean(lat)*1000, "max": max(lat)*1000},
        "sample_answer": last["answers"],
        "usage": last.get("usage"),
    }
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
Path(a.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(report, indent=2, ensure_ascii=False))
