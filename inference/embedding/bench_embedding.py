#!/usr/bin/env python3
"""Embedding 实测脚本（P40 / ollama-embedding 11435）
用法: python3 bench_embedding.py [--model bge-m3|nomic-embed-text] [--bench|--quality]
"""
import json
import sys
import time
import urllib.request

import numpy as np

BASE = "http://127.0.0.1:11435"
DEFAULT_MODEL = "bge-m3"


def embed(model: str, texts: list) -> np.ndarray:
    body = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(f"{BASE}/api/embed", body, {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=120))
    return np.array(r["embeddings"])


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def bench(model: str):
    print(f"=== {model} 吞吐测试 ===")
    for n in (1, 10, 50):
        texts = [f"第 {i} 条测试文本：Tesla P40 推理性能研究" for i in range(n)]
        t0 = time.time()
        e = embed(model, texts)
        dt = time.time() - t0
        print(f"  {n:3d} 条: {dt:6.2f}s ({n/dt:4.1f} 条/s), 维度 {e.shape[1]}")


def quality(model: str):
    print(f"=== {model} 中文语义区分度 ===")
    docs = [
        "Tesla P40 显卡有 24GB 显存，适合跑深度学习推理",
        "服务器上安装了两块 P40 显卡用于模型推理",
        "今天天气很好，适合出去散步",
    ]
    query = "P40 显卡的显存有多大"
    inputs = docs + [query]
    if model == "nomic-embed-text":
        # nomic 官方要求 task 前缀
        inputs = ["search_document: " + t for t in docs] + ["search_query: " + query]
    e = embed(model, inputs)
    d0, d1, d2, q = e[0], e[1], e[2], e[3]
    print(f"  相似句 doc0 vs doc1: {cos(d0, d1):.4f}")
    print(f"  无关句 doc0 vs doc2: {cos(d0, d2):.4f}")
    print(f"  检索 query vs doc0(相关): {cos(q, d0):.4f}")
    print(f"  检索 query vs doc1(相关): {cos(q, d1):.4f}")
    print(f"  检索 query vs doc2(无关): {cos(q, d2):.4f}")
    ok = cos(q, d0) > cos(q, d2) and cos(q, d1) > cos(q, d2)
    print(f"  => 检索排序{'✅ 正确' if ok else '❌ 错误（无关文档排到了相关文档前面）'}")


if __name__ == "__main__":
    args = sys.argv[1:]
    model = DEFAULT_MODEL
    mode = "both"
    if "--model" in args:
        model = args[args.index("--model") + 1]
    if "--bench" in args:
        mode = "bench"
    if "--quality" in args:
        mode = "quality"
    if mode in ("bench", "both"):
        bench(model)
    if mode in ("quality", "both"):
        quality(model)
