"""BGE-M3 embedding + bge-reranker-v2-m3 reranking server (one process, one GPU).

Each visible GPU has one inference thread. The embedder is ready at startup, and the
reranker loads on first use by default. Workers combine queued requests of the same
kind and split them into length-sorted batches constrained by padded token count.

Endpoints
  GET  /health          status, loaded models, GPUs, queue
  GET  /v1/models       OpenAI-style model list
  POST /v1/embeddings   OpenAI-compatible (encoding_format "float" or "base64")
  POST /v1/rerank       Cohere/Jina-style

Configured through env vars (scripts/serve.sh exports them from config.env).
"""

import asyncio
import base64
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
from pydantic_core import to_json
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
BATCH_TOKENS = int(os.environ.get("BATCH_TOKENS", 8192))
PRELOAD_RERANK = os.environ.get("PRELOAD_RERANK", "0") == "1"
if min(BATCH_TOKENS, EMB_BATCH, RR_BATCH, MAX_EMB_LEN, MAX_RR_LEN) < 1:
    raise ValueError("batch sizes, token budget and maximum lengths must be positive")
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


def _batches(lengths: List[int], max_items: int):
    """Group similar lengths. A single long input can use its full context."""
    order = sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True)
    start = 0
    while start < len(order):
        longest = max(1, lengths[order[start]])
        count = min(max_items, max(1, BATCH_TOKENS // longest))
        yield order[start:start + count]
        start += count


class ModelRunner:
    """Owned by one worker, so model loading and inference on this GPU are serialized."""

    def __init__(self, device: str):
        self.device = device
        self.model_kwargs = {"torch_dtype": torch.float16 if USE_FP16 else torch.float32}
        self.embedder = SentenceTransformer(EMBED_MODEL, device=device, model_kwargs=self.model_kwargs.copy())
        self.embedder.max_seq_length = MAX_EMB_LEN
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.reranker = None
        if PRELOAD_RERANK:
            self._load_reranker()

    def _load_reranker(self):
        if self.reranker is None:
            log.info("loading %s on %s", RERANK_MODEL, self.device)
            self.reranker = CrossEncoder(
                RERANK_MODEL, device=self.device, max_length=MAX_RR_LEN, model_kwargs=self.model_kwargs.copy()
            )
        return self.reranker

    def run(self, kind: str, items: list) -> tuple[np.ndarray, List[int]]:
        if kind == "embed":
            model, max_len, batch_size = self.embedder, MAX_EMB_LEN, EMB_BATCH
            outputs = np.empty((len(items), self.dim), dtype=np.float32)
        else:
            model, max_len, batch_size = self._load_reranker(), MAX_RR_LEN, RR_BATCH
            outputs = np.empty(len(items), dtype=np.float32)

        # Tokenize pairs as pairs so the length includes the separator tokens.
        lengths = model.tokenizer(items, truncation=True, max_length=max_len, return_length=True)["length"]
        for indices in _batches(lengths, batch_size):
            batch = [items[i] for i in indices]
            kwargs = dict(batch_size=len(batch), convert_to_numpy=True, show_progress_bar=False)
            if kind == "embed":
                outputs[indices] = model.encode(batch, normalize_embeddings=True, **kwargs)
            else:
                outputs[indices] = model.predict(batch, **kwargs).reshape(-1)
        return outputs, lengths


@dataclass
class Job:
    kind: str     # "embed" | "rerank"
    items: list   # texts, or (query, document) pairs
    future: Future = field(default_factory=Future)


class Scheduler:
    """One job queue, one worker thread per visible GPU."""

    def __init__(self, runners: list, merge_items: int):
        self.runners = runners
        self.busy = 0
        self._merge_items = merge_items
        self._jobs: deque[Job] = deque()
        self._cond = threading.Condition()
        self._stopping = False
        self._threads = [threading.Thread(target=self._work, args=(r,), daemon=True) for r in runners]
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

    def _work(self, runner) -> None:
        torch.cuda.set_device(runner.device)
        while (batch := self._take()) is not None:
            try:
                outputs, tokens = runner.run(batch[0].kind, [x for job in batch for x in job.items])
            except Exception as exc:  # a failed batch fails its requests, not the worker
                log.exception("%s batch of %d jobs failed on GPU %s", batch[0].kind, len(batch), runner.device)
                if "out of memory" in str(exc):
                    log.error("CUDA out of memory: lower BATCH_TOKENS, or the maximum input length for a single item")
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


def load_runners() -> list:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; check CUDA_VISIBLE_DEVICES and the torch build")
    return [ModelRunner(f"cuda:{device}") for device in range(torch.cuda.device_count())]


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    runners = load_runners()
    state["scheduler"] = Scheduler(runners, MERGE_ITEMS)
    state["dim"] = runners[0].dim
    log.info(
        "loaded %s on %d GPU(s) in %.1fs (preload_rerank=%s, fp16=%s, %.2f GiB allocated)",
        EMBED_MODEL, len(runners), time.time() - t0, PRELOAD_RERANK, USE_FP16,
        sum(torch.cuda.memory_allocated(d) for d in {r.device for r in runners}) / 2**30,
    )
    if API_KEY is None:
        log.warning("no API key configured: endpoints are open to anyone who can reach the port")
    try:
        yield
    finally:
        state.pop("scheduler").stop()
        state.clear()
        runners.clear()
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
    body = await asyncio.to_thread(lambda: to_json(build(*args)))
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
    devices = sorted({r.device for r in scheduler.runners})
    return {
        "status": "ok",
        "models": {"embedding": EMBED_MODEL, "rerank": RERANK_MODEL},
        "device": torch.cuda.get_device_name(devices[0]),
        "reranker_loaded": {r.device: r.reranker is not None for r in scheduler.runners},
        "fp16": USE_FP16,
        "max_len": {"embedding": MAX_EMB_LEN, "rerank": MAX_RR_LEN},
        "gpu_mem_gib": {
            "allocated": round(sum(torch.cuda.memory_allocated(d) for d in devices) / 2**30, 2),
            "reserved": round(sum(torch.cuda.memory_reserved(d) for d in devices) / 2**30, 2),
        },
        "jobs": {
            "workers": len(scheduler.runners),
            "running": scheduler.busy,
            "queued": scheduler.queued,
            "merge_items": MERGE_ITEMS,
            "batch_tokens": BATCH_TOKENS,
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
        data = [base64.b64encode(v.astype("<f4", copy=False)).decode() for v in vecs]
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
