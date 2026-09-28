import json
import os
import uuid
from typing import Dict, List, Optional

import redis as redis_lib
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

# --- Embeddings (OpenAI-compatible) ---

class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Dict[str, int]  # prompt_tokens, total_tokens


# --- Rerank ---

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_k: Optional[int] = None


class RerankResult(BaseModel):
    index: int
    score: float


class RerankResponse(BaseModel):
    results: List[RerankResult]


# --- Classify ---

class TextPair(BaseModel):
    id: str
    text_a: str
    text_b: str


class ClassifyRequest(BaseModel):
    text_pairs: List[TextPair]


class ClassifyResult(BaseModel):
    id: str
    relation_type: str
    confidence_score: float
    text_a: str
    text_b: str
    debug_scores: Dict[str, float]


class ClassifyResponse(BaseModel):
    results: List[ClassifyResult]


# ---------------------------------------------------------------------------
# FastAPI app + Redis connection
# ---------------------------------------------------------------------------

app = FastAPI(title="Aurora Model Service v2")

redis: redis_lib.Redis = None  # type: ignore[assignment]


@app.on_event("startup")
def startup_event() -> None:
    global redis
    redis_host = os.environ["REDIS_HOST"]
    redis = redis_lib.Redis(host=redis_host, port=6379, decode_responses=True)


# ---------------------------------------------------------------------------
# Job dispatch helper
# ---------------------------------------------------------------------------

def dispatch_job(queue: str, payload: dict) -> dict:
    job_id = str(uuid.uuid4())
    job = {"job_id": job_id, **payload}
    redis.rpush(queue, json.dumps(job))
    result = redis.blpop(f"result:{job_id}", timeout=60)
    if result is None:
        raise HTTPException(status_code=504, detail="Inference timeout")
    parsed = json.loads(result[1])
    redis.delete(f"result:{job_id}")
    if "error" in parsed:
        raise HTTPException(status_code=500, detail=parsed["error"])
    return parsed


# ---------------------------------------------------------------------------
# Inference endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
    result = dispatch_job("queue:embed", {"input": req.input, "model": req.model})
    return EmbeddingResponse(**result)


@app.post("/v1/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    result = dispatch_job(
        "queue:rerank",
        {"query": req.query, "documents": req.documents, "top_k": req.top_k},
    )
    return RerankResponse(**result)


@app.post("/v1/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    result = dispatch_job(
        "queue:classify",
        {"text_pairs": [p.dict() for p in req.text_pairs]},
    )
    return ClassifyResponse(**result)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> JSONResponse:
    redis_ok = False
    try:
        redis.ping()
        redis_ok = True
    except Exception:
        pass

    worker_ready = redis_ok and redis.get("worker:ready") == "1"

    if redis_ok and worker_ready:
        return JSONResponse(
            {"status": "ok", "redis": True, "worker_ready": True},
            status_code=200,
        )
    return JSONResponse(
        {"status": "error", "redis": redis_ok, "worker_ready": worker_ready},
        status_code=503,
    )
