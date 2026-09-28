"""
Aurora Model Service v2 — GPU Worker
Headless process: loads all three models, then processes inference jobs
from Redis queues one at a time (strictly sequential GPU execution).
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np
import onnxruntime
import redis as redis_lib
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classification schema (preserved exactly from legacy service)
# ---------------------------------------------------------------------------

CANDIDATE_LABELS: List[str] = [
    "The first text requires or depends on the concepts in the second text.",
    "The first text provides examples, details, or further elaboration on the second text.",
    "The first text contradicts or opposes the second text.",
    "The two texts discuss completely unrelated topics.",
]

LABEL_MAP: Dict[str, str] = {
    CANDIDATE_LABELS[0]: "dependency",
    CANDIDATE_LABELS[1]: "expansion",
    CANDIDATE_LABELS[2]: "contradiction",
    CANDIDATE_LABELS[3]: "unrelated",
}

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------

BGE_MODEL_PATH = (
    "/app/models/models--BAAI--bge-reranker-base"
    "/snapshots/2cfc18c9415c912f9d8155881c133215df768a70"
    "/onnx/model.onnx"
)
BGE_TOKENIZER_PATH = (
    "/app/models/models--BAAI--bge-reranker-base"
    "/snapshots/2cfc18c9415c912f9d8155881c133215df768a70"
)

MODERNBERT_MODEL_PATH = (
    "/app/models/models--MoritzLaurer--ModernBERT-large-zeroshot-v2.0"
    "/snapshots/a51e07b524299e309dd2b88d48b0cfa2bd9ec598"
    "/onnx/model_fp16.onnx"
)
MODERNBERT_TOKENIZER_PATH = (
    "/app/models/models--MoritzLaurer--ModernBERT-large-zeroshot-v2.0"
    "/snapshots/a51e07b524299e309dd2b88d48b0cfa2bd9ec598"
)

NOMIC_MODEL_PATH = (
    "/app/models/models--nomic-ai--nomic-embed-text-v2-moe"
    "/snapshots/1066b6599d099fbb93dfcb64f9c37a7c9e503e85"
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _assert_on_gpu(session: onnxruntime.InferenceSession, name: str) -> None:
    """Raise if the ONNX session fell back to CPU instead of running on CUDA."""
    active = session.get_providers()
    log.info("%s active providers: %s", name, active)
    if "CUDAExecutionProvider" not in active:
        raise RuntimeError(
            f"{name} failed to load on GPU — active providers: {active}. "
            "Check that onnxruntime-gpu is installed and CUDA libraries are visible."
        )


def _assert_st_on_gpu(model: SentenceTransformer) -> None:
    """Log and verify that the SentenceTransformer model is on CUDA."""
    import torch
    device = next(model.parameters()).device
    log.info("Nomic Embedder device: %s", device)
    if device.type != "cuda":
        raise RuntimeError(
            f"Nomic Embedder loaded on {device} instead of CUDA. "
            "Ensure a CUDA-capable GPU is available and CUDA_VISIBLE_DEVICES is set."
        )


def load_models():
    """Load all three models and return (bge_session, bge_tokenizer,
    deberta_session, deberta_tokenizer, embedding_model)."""
    try:
        # CUDA provider options — enable fp16 for both sessions
        cuda_provider_options = {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": 8 * 1024 * 1024 * 1024,  # 8 GB
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
            "enable_cuda_graph": False,
        }
        providers = [("CUDAExecutionProvider", cuda_provider_options)]

        log.info("Loading BGE Reranker...")
        bge_session = onnxruntime.InferenceSession(
            BGE_MODEL_PATH,
            providers=providers,
        )
        _assert_on_gpu(bge_session, "BGE Reranker")
        bge_tokenizer = AutoTokenizer.from_pretrained(BGE_TOKENIZER_PATH, use_fast=True)

        log.info("Loading ModernBERT NLI...")
        deberta_session = onnxruntime.InferenceSession(
            MODERNBERT_MODEL_PATH,
            providers=providers,
        )
        _assert_on_gpu(deberta_session, "ModernBERT NLI")
        deberta_tokenizer = AutoTokenizer.from_pretrained(
            MODERNBERT_TOKENIZER_PATH, use_fast=True
        )

        log.info("Loading Nomic Embedder...")
        embedding_model = SentenceTransformer(NOMIC_MODEL_PATH, device="cuda", trust_remote_code=True)
        _assert_st_on_gpu(embedding_model)
    except FileNotFoundError as exc:
        log.error("Model file not found: %s", exc)
        sys.exit(1)

    log.info("Worker Ready")
    return bge_session, bge_tokenizer, deberta_session, deberta_tokenizer, embedding_model


# ---------------------------------------------------------------------------
# Inference handlers
# ---------------------------------------------------------------------------

def handle_embed(job: Dict[str, Any], embedding_model: SentenceTransformer) -> dict:
    """Embed handler — returns OpenAI-compatible EmbeddingResponse dict."""
    texts: List[str] = job["input"]
    model_name: str = job.get("model", "nomic-ai/nomic-embed-text-v2-moe")

    embeddings = embedding_model.encode(texts, normalize_embeddings=True)

    data = [
        {"object": "embedding", "embedding": emb.tolist(), "index": i}
        for i, emb in enumerate(embeddings)
    ]
    token_count = sum(len(t.split()) for t in texts)
    return {
        "object": "list",
        "data": data,
        "model": model_name,
        "usage": {"prompt_tokens": token_count, "total_tokens": token_count},
    }


RERANK_BATCH_SIZE = 8  # Process pairs in small batches to avoid GPU OOM


def handle_rerank(
    job: Dict[str, Any],
    bge_session: onnxruntime.InferenceSession,
    bge_tokenizer: Any,
) -> dict:
    """Rerank handler — returns RerankResponse dict sorted by score descending."""
    query: str = job["query"]
    documents: List[str] = job["documents"]
    top_k: int = job.get("top_k") or len(documents)
    top_k = min(top_k, len(documents))  # clamp to available docs

    pairs = [[query, doc] for doc in documents]
    all_scores: List[float] = []

    # Process in small batches to avoid OOM on large candidate sets
    for i in range(0, len(pairs), RERANK_BATCH_SIZE):
        batch = pairs[i:i + RERANK_BATCH_SIZE]
        inputs = bge_tokenizer(
            batch, padding=True, truncation=True, return_tensors="np"
        )
        logits = bge_session.run(None, dict(inputs))[0]  # shape: (B, 1)
        scores = 1.0 / (1.0 + np.exp(-logits[:, 0]))    # sigmoid → [0, 1]
        all_scores.extend(scores.tolist())

    indexed = sorted(enumerate(all_scores), key=lambda x: x[1], reverse=True)
    results = [{"index": idx, "score": score} for idx, score in indexed[:top_k]]
    return {"results": results}


def _softmax(logits: List[float]) -> np.ndarray:
    arr = np.array(logits, dtype=np.float64)
    arr -= arr.max()  # numerical stability
    exp = np.exp(arr)
    return exp / exp.sum()


def handle_classify(
    job: Dict[str, Any],
    modernbert_session: onnxruntime.InferenceSession,
    modernbert_tokenizer: Any,
) -> dict:
    """Classify handler — returns ClassifyResponse dict."""
    text_pairs = job["text_pairs"]
    results = []

    for pair in text_pairs:
        pair_id: str = pair["id"]
        text_a: str = pair["text_a"]
        text_b: str = pair["text_b"]
        formatted = f"{text_a}\n\n{text_b}"

        label_logits: List[float] = []
        for label in CANDIDATE_LABELS:
            inputs = modernbert_tokenizer(
                formatted,
                label,
                padding=True,
                truncation=True,
                return_tensors="np",
            )
            output = modernbert_session.run(None, dict(inputs))[0]
            # Model outputs [entailment, not_entailment] (shape 1x2); use entailment at index 0
            label_logits.append(float(output[0][0]))

        probs = _softmax(label_logits)
        winning_idx = int(np.argmax(probs))
        relation_type = LABEL_MAP[CANDIDATE_LABELS[winning_idx]]
        debug_scores = {
            LABEL_MAP[CANDIDATE_LABELS[i]]: float(probs[i]) for i in range(4)
        }
        confidence_score = float(probs[winning_idx])

        results.append(
            {
                "id": pair_id,
                "relation_type": relation_type,
                "confidence_score": confidence_score,
                "text_a": text_a,
                "text_b": text_b,
                "debug_scores": debug_scores,
            }
        )

    return {"results": results}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_worker(
    r: redis_lib.Redis,
    bge_session: onnxruntime.InferenceSession,
    bge_tokenizer: Any,
    deberta_session: onnxruntime.InferenceSession,
    deberta_tokenizer: Any,
    embedding_model: SentenceTransformer,
) -> None:
    """Blocking brpop loop — processes one job at a time, never exits on error."""
    queues = ["queue:embed", "queue:rerank", "queue:classify"]
    log.info("Worker entering main loop, listening on %s", queues)

    while True:
        try:
            item = r.brpop(queues, timeout=0)
        except redis_lib.exceptions.TimeoutError:
            # Socket-level timeout with no job — not a real error, just retry
            continue
        if item is None:
            continue

        queue_name_bytes, job_json = item
        # brpop returns bytes when decode_responses=False, str otherwise
        queue_name = (
            queue_name_bytes.decode()
            if isinstance(queue_name_bytes, bytes)
            else queue_name_bytes
        )
        job_json_str = (
            job_json.decode() if isinstance(job_json, bytes) else job_json
        )

        job: Dict[str, Any] = {}
        job_id: str = ""
        try:
            job = json.loads(job_json_str)
            job_id = job["job_id"]

            if queue_name == "queue:embed":
                result = handle_embed(job, embedding_model)
            elif queue_name == "queue:rerank":
                result = handle_rerank(job, bge_session, bge_tokenizer)
            elif queue_name == "queue:classify":
                result = handle_classify(job, deberta_session, deberta_tokenizer)
            else:
                raise ValueError(f"Unknown queue: {queue_name}")

            r.rpush(f"result:{job_id}", json.dumps(result))
            r.expire(f"result:{job_id}", 60)

        except Exception as exc:  # noqa: BLE001
            log.exception("Error processing job %s from %s: %s", job_id, queue_name, exc)
            if job_id:
                error_result = {"error": str(exc)}
                r.rpush(f"result:{job_id}", json.dumps(error_result))
                r.expire(f"result:{job_id}", 60)


def main() -> None:
    redis_host = os.environ.get("REDIS_HOST", "localhost")

    # Wait for Redis to be reachable before loading models.
    # This handles startup ordering — Redis may not be ready yet when the
    # worker container starts, especially on first `docker compose up`.
    r = None
    for attempt in range(1, 31):  # retry for up to ~5 minutes
        try:
            client = redis_lib.Redis(
                host=redis_host,
                port=6379,
                decode_responses=True,
                socket_keepalive=True,
                socket_connect_timeout=5,
            )
            client.ping()
            r = client
            log.info("Connected to Redis at %s:6379", redis_host)
            break
        except (redis_lib.exceptions.ConnectionError, redis_lib.exceptions.TimeoutError) as exc:
            log.warning(
                "Redis not ready (attempt %d/30): %s — retrying in 10s", attempt, exc
            )
            import time
            time.sleep(10)

    if r is None:
        log.error("Could not connect to Redis after 30 attempts. Exiting.")
        sys.exit(1)

    bge_session, bge_tokenizer, deberta_session, deberta_tokenizer, embedding_model = (
        load_models()
    )

    r.set("worker:ready", "1")

    run_worker(r, bge_session, bge_tokenizer, deberta_session, deberta_tokenizer, embedding_model)


if __name__ == "__main__":
    main()
