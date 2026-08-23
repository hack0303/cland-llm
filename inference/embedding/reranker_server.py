"""Reranker 常驻服务（bge-reranker-v2-m3，GPU1）

用法：
  ./start_reranker.sh                # 启动（手动，无自启）
  curl http://127.0.0.1:10304/health
  curl http://127.0.0.1:10304/rerank -H 'Content-Type: application/json' -d '{
    "query": "如何优化 P40 显存的利用率？",
    "documents": ["在 P40 上部署大模型建议 4-bit 量化", "今天天气很好"],
    "top_k": 1
  }'

运行环境：dev_bge（conda），CUDA_VISIBLE_DEVICES=1（与 embedding 同卡，模型小不冲突）
模型：/mnt/data/ai_workspace/models/bge-reranker-v2-m3（fp16，双卡自动，单卡 ~1.2GB）
"""
import logging
from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from FlagEmbedding import FlagReranker

RERANKER_PATH = "/mnt/data/ai_workspace/models/bge-reranker-v2-m3"
logging.getLogger("FlagEmbedding").setLevel(logging.WARNING)

# 启动时加载一次，常驻
reranker = FlagReranker(RERANKER_PATH, use_fp16=True)

app = FastAPI(title="Reranker Service")


class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_k: Optional[int] = None  # 不传则返回全部排序结果


@app.get("/health")
def health():
    return {"status": "ok", "model": "bge-reranker-v2-m3", "port": 10304}


@app.post("/rerank")
def rerank(req: RerankRequest):
    """(query, doc) 逐对打分，按分数降序返回"""
    if not req.documents:
        return {"query": req.query, "results": []}
    pairs = [[req.query, d] for d in req.documents]
    scores = reranker.compute_score(pairs)
    results = sorted(zip(req.documents, scores), key=lambda x: x[1], reverse=True)
    if req.top_k:
        results = results[: req.top_k]
    return {
        "query": req.query,
        "count": len(results),
        "results": [{"text": t, "score": float(s)} for t, s in results],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10304, log_level="info")
