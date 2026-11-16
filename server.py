"""BGE-M3 embedding + bge-reranker-v2-m3 reranking server (one process, one GPU).

Endpoints
  GET  /health          status, loaded models, GPU memory
  GET  /v1/models       OpenAI-style model list
  POST /v1/embeddings   OpenAI-compatible (encoding_format "float" or "base64")
  POST /v1/rerank       Cohere/Jina-style

Configured through env vars (scripts/serve.sh exports them from config.env).
"""

import base64
import logging
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer

EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
MAX_EMB_LEN = int(os.environ.get("MAX_EMB_LEN", 8192))
MAX_RR_LEN = int(os.environ.get("MAX_RR_LEN", 1024))
EMB_BATCH = int(os.environ.get("EMB_BATCH", 16))
RR_BATCH = int(os.environ.get("RR_BATCH", 32))
USE_FP16 = os.environ.get("USE_FP16", "1") == "1"
MAX_INPUTS = int(os.environ.get("MAX_INPUTS", 2048))

EMBED_ID = EMBED_MODEL.split("/")[-1]    # "bge-m3"
RERANK_ID = RERANK_MODEL.split("/")[-1]  # "bge-reranker-v2-m3"

log = logging.getLogger("uvicorn.error")


def _load_api_key() -> Optional[str]:
    path = os.environ.get("EMBED_API_KEY_FILE")
    if path and Path(path).is_file():
        return Path(path).read_text().strip() or None
    return os.environ.get("EMBED_API_KEY") or None


API_KEY = _load_api_key()
models: dict = {}
gpu_lock = threading.Lock()  # one GPU job at a time: predictable memory, no OOM from overlap


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; check CUDA_VISIBLE_DEVICES and the torch build")
    t0 = time.time()
    embedder = SentenceTransformer(EMBED_MODEL, device="cuda")
    embedder.max_seq_length = MAX_EMB_LEN
    reranker = CrossEncoder(RERANK_MODEL, device="cuda", max_length=MAX_RR_LEN)
    if USE_FP16:
        embedder.half()
        reranker.half()
    embedder.eval()
    reranker.eval()
    models["embed"] = embedder
    models["rerank"] = reranker
    log.info(
        "loaded %s + %s on %s in %.1fs (fp16=%s, %.2f GiB allocated)",
        EMBED_MODEL, RERANK_MODEL, torch.cuda.get_device_name(0), time.time() - t0,
        USE_FP16, torch.cuda.memory_allocated() / 2**30,
    )
    if API_KEY is None:
        log.warning("no API key configured: endpoints are open to anyone who can reach the port")
    yield
    models.clear()
    torch.cuda.empty_cache()


app = FastAPI(title="bge-m3 embedding + rerank", lifespan=lifespan)


def require_key(authorization: Optional[str] = Header(None)):
    if API_KEY is None:
        return
    if not authorization or not secrets.compare_digest(authorization, f"Bearer {API_KEY}"):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def _count_tokens(tokenizer, texts: List[str], max_len: int) -> int:
    ids = tokenizer(texts, truncation=True, max_length=max_len)["input_ids"]
    return sum(len(x) for x in ids)


# ---------------------------------------------------------------- schemas


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None          # accepted for client compatibility; one model is served
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class RerankDocument(BaseModel):
    text: str


class RerankRequest(BaseModel):
    query: str
    documents: List[Union[str, RerankDocument]]
    top_n: Optional[int] = Field(None, ge=1)
    return_documents: bool = True
    model: Optional[str] = None


# ---------------------------------------------------------------- routes


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {"embedding": EMBED_MODEL, "rerank": RERANK_MODEL},
        "device": torch.cuda.get_device_name(0),
        "fp16": USE_FP16,
        "max_len": {"embedding": MAX_EMB_LEN, "rerank": MAX_RR_LEN},
        "gpu_mem_gib": {
            "allocated": round(torch.cuda.memory_allocated() / 2**30, 2),
            "reserved": round(torch.cuda.memory_reserved() / 2**30, 2),
        },
    }


@app.get("/v1/models", dependencies=[Depends(require_key)])
def list_models():
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": EMBED_ID, "object": "model", "created": created, "owned_by": "BAAI"},
            {"id": RERANK_ID, "object": "model", "created": created, "owned_by": "BAAI"},
        ],
    }


@app.post("/v1/embeddings", dependencies=[Depends(require_key)])
def create_embeddings(req: EmbeddingRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    if not texts:
        raise HTTPException(400, "input must not be empty")
    if len(texts) > MAX_INPUTS:
        raise HTTPException(400, f"at most {MAX_INPUTS} inputs per request")

    embedder = models["embed"]
    with gpu_lock:
        vecs = embedder.encode(
            texts,
            batch_size=EMB_BATCH,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    if req.dimensions is not None and req.dimensions != vecs.shape[1]:
        raise HTTPException(400, f"{EMBED_ID} only produces {vecs.shape[1]}-dim embeddings")

    if req.encoding_format == "base64":
        data = [base64.b64encode(v.astype("<f4").tobytes()).decode() for v in vecs]
    else:
        data = [v.tolist() for v in vecs]

    n_tokens = _count_tokens(embedder.tokenizer, texts, MAX_EMB_LEN)
    return {
        "object": "list",
        "model": EMBED_ID,
        "data": [{"object": "embedding", "index": i, "embedding": e} for i, e in enumerate(data)],
        "usage": {"prompt_tokens": n_tokens, "total_tokens": n_tokens},
    }


@app.post("/v1/rerank", dependencies=[Depends(require_key)])
def rerank(req: RerankRequest):
    docs = [d if isinstance(d, str) else d.text for d in req.documents]
    if len(docs) > MAX_INPUTS:
        raise HTTPException(400, f"at most {MAX_INPUTS} documents per request")
    if not docs:
        return {"model": RERANK_ID, "results": [], "usage": {"total_tokens": 0}}

    reranker = models["rerank"]
    pairs = [(req.query, d) for d in docs]
    with gpu_lock:
        scores = reranker.predict(
            pairs, batch_size=RR_BATCH, convert_to_numpy=True, show_progress_bar=False
        )  # sigmoid-activated, 0..1

    order = sorted(range(len(docs)), key=lambda i: -float(scores[i]))
    if req.top_n is not None:
        order = order[: req.top_n]

    results = []
    for i in order:
        item = {"index": i, "relevance_score": float(scores[i])}
        if req.return_documents:
            item["document"] = {"text": docs[i]}
        results.append(item)

    n_tokens = _count_tokens(reranker.tokenizer, [f"{req.query} {d}" for d in docs], MAX_RR_LEN)
    return {"model": RERANK_ID, "results": results, "usage": {"total_tokens": n_tokens}}
