#!/usr/bin/env python3
"""Smoke test for the running server. Standard library only, so it works on the
server (no curl there) and on a laptop through an SSH tunnel.

  python scripts/test_client.py                         # http://127.0.0.1:8001, key from ~/.embed_api_key
  python scripts/test_client.py --url http://127.0.0.1:8001 --key-file ~/.embed_api_key
"""

import argparse
import array
import base64
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request


def call(url, key, path, payload=None):
    req = urllib.request.Request(url.rstrip("/") + path, method="POST" if payload is not None else "GET")
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    body = json.dumps(payload).encode() if payload is not None else None
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, body, timeout=120) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"{path}: HTTP {e.code}: {e.read().decode()}")
    return out, (time.time() - t0) * 1000


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8001")
    p.add_argument("--key", default=os.environ.get("EMBED_API_KEY"))
    p.add_argument("--key-file", default="~/.embed_api_key")
    args = p.parse_args()

    key = args.key
    key_file = os.path.expanduser(args.key_file)
    if not key and os.path.isfile(key_file):
        key = open(key_file).read().strip()

    results = []

    h, ms = call(args.url, None, "/health")
    results.append(check("health", h.get("status") == "ok", f"({ms:.0f} ms) {h.get('device')} {h.get('gpu_mem_gib')}"))

    texts = [
        "The giant panda is a bear species native to China.",
        "Gấu trúc lớn là một loài gấu sống ở Trung Quốc.",
        "Paris is the capital of France.",
    ]
    e, ms = call(args.url, key, "/v1/embeddings", {"input": texts, "model": "bge-m3"})
    vecs = [d["embedding"] for d in e["data"]]
    dim = len(vecs[0])
    norm = math.sqrt(sum(x * x for x in vecs[0]))
    results.append(check("embeddings shape", len(vecs) == 3 and dim == 1024, f"({ms:.0f} ms) n={len(vecs)} dim={dim}"))
    results.append(check("embeddings normalized", abs(norm - 1) < 1e-3, f"|v|={norm:.4f}"))
    s_same, s_diff = cos(vecs[0], vecs[1]), cos(vecs[0], vecs[2])
    results.append(check("cross-lingual similarity", s_same > s_diff, f"en~vi={s_same:.3f} en~paris={s_diff:.3f}"))

    b, _ = call(args.url, key, "/v1/embeddings", {"input": texts[0], "encoding_format": "base64"})
    decoded = array.array("f")
    decoded.frombytes(base64.b64decode(b["data"][0]["embedding"]))
    results.append(check("base64 matches float", max(abs(x - y) for x, y in zip(decoded, vecs[0])) < 1e-5))

    docs = ["Paris is the capital of France.", "The giant panda is a bear native to China.", "Pandas is a Python library."]
    r, ms = call(args.url, key, "/v1/rerank", {"query": "what is a panda bear?", "documents": docs, "top_n": 2})
    top = r["results"][0]
    results.append(check(
        "rerank order", top["index"] == 1 and len(r["results"]) == 2,
        f"({ms:.0f} ms) " + ", ".join(f"#{x['index']}={x['relevance_score']:.3f}" for x in r["results"]),
    ))

    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
