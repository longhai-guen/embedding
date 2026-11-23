#!/usr/bin/env python3
"""Load test: latency and throughput at several concurrency levels. Standard library only.

  python scripts/bench.py                                  # rerank 40 docs + embed 1 query, at 1 / 4 / 8 / 16 clients
  python scripts/bench.py --kind rerank --docs 40 --concurrency 1 8 32 --requests 64

Run it before and after a change (GPU_INDEX, USE_FP16, MERGE_ITEMS, RR_BATCH) with the same arguments.
"""

import argparse
import json
import os
import random
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

WORDS = ("máy bơm nước ly tâm điện công suất hàng hóa thép không gỉ nhựa linh kiện dùng cho xe có động cơ "
         "pump steel plastic parts machine electric motor food preparation household").split()


def text(n_words, rng):
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def post(url, key, path, payload):
    req = urllib.request.Request(url.rstrip("/") + path, json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        r.read()
    return time.perf_counter() - t0


def run(args, key, kind, clients):
    rng = random.Random(0)
    if kind == "rerank":
        path = "/v1/rerank"
        payloads = [{"query": text(15, rng), "documents": [text(args.doc_words, rng) for _ in range(args.docs)],
                     "return_documents": False} for _ in range(args.requests)]
    else:
        path = "/v1/embeddings"
        payloads = [{"input": [text(15, rng)]} for _ in range(args.requests)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(clients) as pool:
        lat = sorted(pool.map(lambda p: post(args.url, key, path, p), payloads))
    wall = time.perf_counter() - t0
    p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
    print(f"{kind:6} clients={clients:3}  p50 {statistics.median(lat):6.2f}s  p95 {p95:6.2f}s  "
          f"max {lat[-1]:6.2f}s  {len(lat) / wall:6.1f} req/s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8001")
    p.add_argument("--key", default=os.environ.get("EMBED_API_KEY"))
    p.add_argument("--key-file", default="~/.embed_api_key")
    p.add_argument("--kind", choices=("rerank", "embed", "both"), default="both")
    p.add_argument("--docs", type=int, default=40, help="documents per rerank request")
    p.add_argument("--doc-words", type=int, default=90, help="words per document (~500 characters)")
    p.add_argument("--requests", type=int, default=32, help="requests per concurrency level")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    args = p.parse_args()

    key = args.key
    key_file = os.path.expanduser(args.key_file)
    if not key and os.path.isfile(key_file):
        key = open(key_file).read().strip()

    kinds = ("rerank", "embed") if args.kind == "both" else (args.kind,)
    for kind in kinds:
        post(args.url, key, "/v1/rerank" if kind == "rerank" else "/v1/embeddings",
             {"query": "warm up", "documents": ["x"]} if kind == "rerank" else {"input": "warm up"})
        for clients in args.concurrency:
            run(args, key, kind, clients)


if __name__ == "__main__":
    main()
