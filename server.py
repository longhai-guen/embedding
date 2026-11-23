"""BGE-M3 embedding + bge-reranker-v2-m3 reranking server (one process, one GPU).

``WORKERS`` worker threads share the GPU (``GPU_INDEX`` in config.env), each with its own copy of both models and its
own CUDA stream, so that many jobs run on the GPU at once and one job's CPU work (tokenizing, copying results) overlaps
another's GPU work. Requests go into one queue. A free worker takes the oldest job and, with it, the other waiting
jobs of the same kind (up to ``MERGE_ITEMS`` texts or pairs), so requests that arrive together run as one batch.
A job whose client has gone away before a worker starts it is dropped.

Endpoints
  GET  /health          status, loaded models, GPUs, queue
  GET  /v1/models       OpenAI-style model list
  POST /v1/embeddings   OpenAI-compatible (encoding_format "float" or "base64")
  POST /v1/rerank       Cohere/Jina-style

Configured through env vars (scripts/serve.sh exports them from config.env).
"""

import asyncio
import base64
import json
import logging
import os
import secrets
import threading
import time
from collections import deque
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
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
MERGE_ITEMS = int(os.environ.get("MERGE_ITEMS", 256))  # texts / pairs per merged GPU job
WORKERS = int(os.environ.get("WORKERS", 2))  # jobs on the GPU at once; each holds a copy of both models (~2.3 GB)
DISCONNECT_POLL_S = 0.5

EMBED_ID = EMBED_MODEL.split("/")[-1]    # "bge-m3"
RERANK_ID = RERANK_MODEL.split("/")[-1]  # "bge-reranker-v2-m3"

log = logging.getLogger("uvicorn.error")


def _load_api_key() -> Optional[str]:
    path = os.environ.get("EMBED_API_KEY_FILE")
    if path and Path(path).is_file():
        return Path(path).read_text().strip() or None
    return os.environ.get("EMBED_API_KEY") or None


API_KEY = _load_api_key()


# ---------------------------------------------------------------- GPUs and the job queue


def _token_counts(tokenizer, texts: List[str], max_len: int) -> List[int]:
    return [len(ids) for ids in tokenizer(texts, truncation=True, max_length=max_len)["input_ids"]]


class Replica:
    """Both models on one GPU, for one worker thread only (an HF tokenizer must not be shared between threads)."""

    def __init__(self, device: int):
        self.device = device
        self.stream = None
        self.name = torch.cuda.get_device_name(device)
        self.embedder = SentenceTransformer(EMBED_MODEL, device=f"cuda:{device}")
        self.embedder.max_seq_length = MAX_EMB_LEN
        self.reranker = CrossEncoder(RERANK_MODEL, device=f"cuda:{device}", max_length=MAX_RR_LEN)
        if USE_FP16:
            self.embedder.half()
            self.reranker.half()
        self.embedder.eval()
        self.reranker.eval()
        self.dim = self.embedder.get_sentence_embedding_dimension()

    def bind(self) -> None:
        """In the worker thread: its own stream, so its kernels can run next to the other workers'."""
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.Stream(self.device)

    def run(self, kind: str, items: list) -> tuple[np.ndarray, List[int]]:
        """One output and one token count per item: an embedding per text, or a score per (query, document)."""
        with torch.cuda.stream(self.stream):
            return self._run(kind, items)

    def _run(self, kind: str, items: list) -> tuple[np.ndarray, List[int]]:
        if kind == "embed":
            vecs = self.embedder.encode(
                items, batch_size=EMB_BATCH, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            ).astype(np.float32)
            return vecs, _token_counts(self.embedder.tokenizer, items, MAX_EMB_LEN)
        # longest first, so each batch pads to a similar length
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]) + len(items[i][1]), reverse=True)
        scores = np.empty(len(items), dtype=np.float32)
        scores[order] = self.reranker.predict(
            [items[i] for i in order], batch_size=RR_BATCH, convert_to_numpy=True, show_progress_bar=False
        )  # sigmoid-activated, 0..1
        return scores, _token_counts(self.reranker.tokenizer, [f"{q} {d}" for q, d in items], MAX_RR_LEN)


@dataclass
class Job:
    kind: str     # "embed" | "rerank"
    items: list   # texts, or (query, document) pairs
    future: Future = field(default_factory=Future)


class Scheduler:
    """One job queue, one worker thread per replica."""

    def __init__(self, replicas: list, merge_items: int):
        self.replicas = replicas
        self.busy = 0
        self._merge_items = merge_items
        self._jobs: deque[Job] = deque()
        self._cond = threading.Condition()
        self._stopping = False
        self._threads = [threading.Thread(target=self._work, args=(r,), daemon=True) for r in replicas]
        for t in self._threads:
            t.start()

    @property
    def queued(self) -> int:
        return len(self._jobs)

    def submit(self, kind: str, items: list) -> Future:
        """The future's result is (outputs, total tokens) of this job alone."""
        job = Job(kind, items)
        with self._cond:
            self._jobs.append(job)
            self._cond.notify()
        return job.future

    def stop(self) -> None:
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        for t in self._threads:
            t.join()

    def _take(self) -> Optional[List[Job]]:
        """The oldest job plus the waiting jobs of its kind that fit in ``merge_items``; None when stopping."""
        with self._cond:
            while True:
                while not self._jobs and not self._stopping:
                    self._cond.wait()
                if self._stopping:
                    return None
                first = self._jobs.popleft()
                if not first.future.set_running_or_notify_cancel():
                    continue  # the client went away before any GPU started it
                batch, size = [first], len(first.items)
                for job in list(self._jobs):
                    if job.kind == first.kind and size + len(job.items) <= self._merge_items:
                        self._jobs.remove(job)
                        if job.future.set_running_or_notify_cancel():
                            batch.append(job)
                            size += len(job.items)
                self.busy += 1
                return batch

    def _work(self, replica) -> None:
        replica.bind()
        while (batch := self._take()) is not None:
            try:
                outputs, tokens = replica.run(batch[0].kind, [x for job in batch for x in job.items])
            except Exception as exc:  # a failed batch fails its requests, not the worker
                log.exception("%s batch of %d jobs failed on GPU %s", batch[0].kind, len(batch), replica.device)
                if "out of memory" in str(exc):
                    log.error("CUDA out of memory: lower WORKERS, RR_BATCH / EMB_BATCH or MAX_EMB_LEN")
                for job in batch:
                    job.future.set_exception(exc)
            else:
                start = 0
                for job in batch:
                    end = start + len(job.items)
                    job.future.set_result((outputs[start:end], sum(tokens[start:end])))
                    start = end
            finally:
                with self._cond:
                    self.busy -= 1


def load_replicas() -> list:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; check CUDA_VISIBLE_DEVICES and the torch build")
    return [Replica(device) for device in range(torch.cuda.device_count()) for _ in range(WORKERS)]


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    replicas = load_replicas()
    state["scheduler"] = Scheduler(replicas, MERGE_ITEMS)
    state["dim"] = replicas[0].dim
    log.info(
        "loaded %s + %s on %s, %d workers, in %.1fs (fp16=%s, merge up to %d items, %.2f GiB allocated)",
        EMBED_MODEL, RERANK_MODEL, replicas[0].name, len(replicas), time.time() - t0, USE_FP16, MERGE_ITEMS,
        sum(torch.cuda.memory_allocated(d) for d in {r.device for r in replicas}) / 2**30,
    )
    if API_KEY is None:
        log.warning("no API key configured: endpoints are open to anyone who can reach the port")
    yield
    state.pop("scheduler").stop()
    state.clear()
    torch.cuda.empty_cache()


app = FastAPI(title="bge-m3 embedding + rerank", lifespan=lifespan)


def require_key(authorization: Optional[str] = Header(None)):
    if API_KEY is None:
        return
    if not authorization or not secrets.compare_digest(authorization, f"Bearer {API_KEY}"):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


async def _run(request: Request, kind: str, items: list):
    """Queue the job and wait for it; cancel it if the client disconnects first."""
    future = state["scheduler"].submit(kind, items)
    waiting = asyncio.wrap_future(future)
    while True:
        done, _ = await asyncio.wait({waiting}, timeout=DISCONNECT_POLL_S)
        if done:
            return waiting.result()
        if await request.is_disconnected():
            future.cancel()  # only works while no GPU has started it
            waiting.add_done_callback(lambda f: f.cancelled() or f.exception())  # no "never retrieved" warning
            raise HTTPException(499, "client disconnected")


async def _json(build, *args) -> Response:
    """Build and serialize the answer off the event loop: thousands of vectors take a while."""
    body = await asyncio.to_thread(lambda: json.dumps(build(*args)))
    return Response(body, media_type="application/json")


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
    scheduler = state["scheduler"]
    devices = sorted({r.device for r in scheduler.replicas})
    return {
        "status": "ok",
        "models": {"embedding": EMBED_MODEL, "rerank": RERANK_MODEL},
        "device": scheduler.replicas[0].name,
        "fp16": USE_FP16,
        "max_len": {"embedding": MAX_EMB_LEN, "rerank": MAX_RR_LEN},
        "gpu_mem_gib": {
            "allocated": round(sum(torch.cuda.memory_allocated(d) for d in devices) / 2**30, 2),
            "reserved": round(sum(torch.cuda.memory_reserved(d) for d in devices) / 2**30, 2),
        },
        "jobs": {
            "workers": len(scheduler.replicas),
            "running": scheduler.busy,
            "queued": scheduler.queued,
            "merge_items": MERGE_ITEMS,
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


def _embedding_answer(vecs: np.ndarray, n_tokens: int, encoding_format: str) -> dict:
    if encoding_format == "base64":
        data = [base64.b64encode(v.astype("<f4").tobytes()).decode() for v in vecs]
    else:
        data = [v.tolist() for v in vecs]
    return {
        "object": "list",
        "model": EMBED_ID,
        "data": [{"object": "embedding", "index": i, "embedding": e} for i, e in enumerate(data)],
        "usage": {"prompt_tokens": n_tokens, "total_tokens": n_tokens},
    }


@app.post("/v1/embeddings", dependencies=[Depends(require_key)])
async def create_embeddings(req: EmbeddingRequest, request: Request):
    texts = [req.input] if isinstance(req.input, str) else req.input
    if not texts:
        raise HTTPException(400, "input must not be empty")
    if len(texts) > MAX_INPUTS:
        raise HTTPException(400, f"at most {MAX_INPUTS} inputs per request")
    if req.dimensions is not None and req.dimensions != state["dim"]:
        raise HTTPException(400, f"{EMBED_ID} only produces {state['dim']}-dim embeddings")

    vecs, n_tokens = await _run(request, "embed", texts)
    return await _json(_embedding_answer, vecs, n_tokens, req.encoding_format)


@app.post("/v1/rerank", dependencies=[Depends(require_key)])
async def rerank(req: RerankRequest, request: Request):
    docs = [d if isinstance(d, str) else d.text for d in req.documents]
    if len(docs) > MAX_INPUTS:
        raise HTTPException(400, f"at most {MAX_INPUTS} documents per request")
    if not docs:
        return {"model": RERANK_ID, "results": [], "usage": {"total_tokens": 0}}

    scores, n_tokens = await _run(request, "rerank", [(req.query, d) for d in docs])

    order = sorted(range(len(docs)), key=lambda i: -float(scores[i]))
    if req.top_n is not None:
        order = order[: req.top_n]
    results = []
    for i in order:
        item = {"index": i, "relevance_score": float(scores[i])}
        if req.return_documents:
            item["document"] = {"text": docs[i]}
        results.append(item)
    return {"model": RERANK_ID, "results": results, "usage": {"total_tokens": n_tokens}}
