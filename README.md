# embedding

Self-hosted **BGE-M3** embeddings and **bge-reranker-v2-m3** reranking on a single GPU.
One FastAPI process loads BGE-M3 at startup and the reranker on first use. It serves an OpenAI-compatible
`/v1/embeddings` endpoint and a Cohere/Jina-style `/v1/rerank` endpoint.

```
config.env.example      template for settings (storage paths, GPU index, port, models, lengths)
config.env              your real settings, created on the server (gitignored)
server.py               FastAPI app
requirements.txt        Python deps (torch is installed separately, per CUDA driver)
scripts/setup_env.sh    one-time: conda env + caches on shared storage + torch + deps
scripts/download_models.sh  download models (or link them from a shared hub), verify offline load
scripts/serve.sh        start | stop | restart | status | logs
scripts/test_client.py  smoke test (stdlib only)
scripts/bench.py        load test: latency and req/s at several concurrency levels (stdlib only)
logs/                   server.log and server.pid (gitignored)
```

## Storage layout

Heavy files stay out of `$HOME`. All paths come from `config.env`:

| What | Where |
| --- | --- |
| conda env `embed` | `$STORAGE_ROOT/conda/envs` (via conda `envs_dirs`) |
| `HF_HOME` (token, xet cache, `hub/`) | `$STORAGE_ROOT/cache/huggingface` |
| pip / torch caches | `$STORAGE_ROOT/cache/{pip,torch}` |
| model weights | `$SHARED_HUB/models--BAAI--*` if `SHARED_HUB` is set (symlinked into `$HF_HOME/hub`), else `$HF_HOME/hub` |

`HF_HOME` stays personal, never point it at a shared folder. Only model folders are shared.

## Setup on a GPU server

**0. Conda (once per server).** If Miniconda isn't installed yet, install `Miniconda3-latest-Linux-x86_64.sh` into your home directory, then point envs and pkgs at shared storage *before* creating envs:

```bash
storage_root="/path/to/your/storage"    # same as STORAGE_ROOT in config.env
mkdir -p "$storage_root/conda/envs" "$storage_root/conda/pkgs"
conda config --prepend envs_dirs "$storage_root/conda/envs"
conda config --prepend pkgs_dirs "$storage_root/conda/pkgs"
conda config --show envs_dirs pkgs_dirs   # shared paths must come first
```

Log out and back in.

**1. Clone and configure.**

```bash
git clone https://github.com/longhai-guen/embedding.git ~/embedding && cd ~/embedding
cp config.env.example config.env
nano config.env      # set STORAGE_ROOT, SHARED_HUB (optional), GPU_INDEX from nvidia-smi
```

`config.env` is gitignored, so your server paths never reach the repo. The scripts stop if it is missing
or still has the placeholder path.

**2. Create the env.** This picks the torch CUDA wheel from the driver version (`cu129` for 12.9, and so on). To force one, set `TORCH_CUDA="cu126"` in `config.env`.

```bash
scripts/setup_env.sh
```

**3. Get the models.** With `SHARED_HUB` set, models already in the shared hub are only symlinked, and missing ones are downloaded there so others can reuse them. With it empty, they download into `$HF_HOME/hub`.

```bash
scripts/download_models.sh
```

**4. Run.**

```bash
scripts/serve.sh start      # waits until /health answers
scripts/serve.sh status     # pid, health JSON, GPU memory
scripts/serve.sh logs       # tail -f logs/server.log
scripts/serve.sh restart    # after changing config or server.py
scripts/serve.sh stop
```

On first start, an API key is generated in `~/.embed_api_key` (mode 600). The server runs with
`HF_HUB_OFFLINE=1`, so it never contacts the Hub at serve time.

**5. Test.**

```bash
python scripts/test_client.py
python scripts/bench.py        # before and after changing BATCH_TOKENS, USE_FP16, EMB_BATCH, RR_BATCH
```

## Using it from your laptop

The server listens on `127.0.0.1:8001` only. Tunnel to it:

```bash
ssh -N -L 8001:127.0.0.1:8001 <server>
```

Then copy the key (`ssh <server> cat ~/.embed_api_key`) and use it like any OpenAI-compatible embedding API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key=KEY)
vecs = client.embeddings.create(model="bge-m3", input=["hello", "xin chào"])
```

```python
import requests

r = requests.post(
    "http://127.0.0.1:8001/v1/rerank",
    headers={"Authorization": f"Bearer {KEY}"},
    json={"query": "what is a panda?", "documents": ["Pandas are bears.", "Paris is in France."], "top_n": 1},
)
print(r.json()["results"])   # [{"index": 0, "relevance_score": 0.99, "document": {"text": ...}}]
```

## API

All endpoints except `/health` need `Authorization: Bearer <key>`.

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/health` | | status, models, device, GPU memory, jobs running / queued |
| GET | `/v1/models` | | `bge-m3`, `bge-reranker-v2-m3` |
| POST | `/v1/embeddings` | `input` (str or list), `encoding_format` (`float`/`base64`), `model` (ignored) | OpenAI embedding list, 1024-dim, L2-normalized |
| POST | `/v1/rerank` | `query`, `documents` (str or `{"text"}`), `top_n`, `return_documents` | results sorted by `relevance_score` (sigmoid, 0–1) |

Embeddings are normalized, so cosine similarity equals the dot product. Inputs longer than
`MAX_EMB_LEN` / `MAX_RR_LEN` tokens are truncated.

## Tuning (`config.env`)

| Variable | Default | Notes |
| --- | --- | --- |
| `GPU_INDEX` | `0` | `nvidia-smi` index (PCI bus order) |
| `PORT` / `HOST` | `8001` / `127.0.0.1` | keep localhost and use SSH tunnels |
| `MAX_EMB_LEN` | `8192` | bge-m3's full context |
| `MAX_RR_LEN` | `1024` | per query+passage pair; up to 8192 |
| `EMB_BATCH` / `RR_BATCH` | `16` / `32` | lower these if long inputs hit CUDA OOM |
| `USE_FP16` | `1` | Pascal cards (GTX 10xx) have no tensor cores: measure `0` with `scripts/bench.py` |
| `MAX_INPUTS` | `2048` | texts/documents per request |
| `BATCH_TOKENS` | `8192` | padded tokens per inference batch, shared setting for both models |
| `PRELOAD_RERANK` | `0` | set to `1` to load both models before the server reports ready |
| `MERGE_ITEMS` | `256` | waiting requests of the same kind are merged into one job up to this many texts/pairs |

### How requests run

One thread owns each visible GPU. Existing `config.env` files can keep the old `WORKERS` setting.
It no longer controls model copies. With `USE_FP16=1`, weights load directly in fp16. The first rerank
request on a GPU pays the cost of loading the reranker from the local cache. Subsequent requests reuse it.
During that load, embedding requests for the same GPU wait. Check `reranker_loaded` in `/health` for its state.

A worker takes the oldest request and merges waiting requests of the same kind up to `MERGE_ITEMS`.
Within that group it sorts texts or pairs by token length. Each inference batch fits both the item limit
and the padded-token budget (longest sequence length × batch size). A single input longer than the budget
runs alone, up to the configured model context length. The server restores the original item indices and
counts usage per request. Queued requests are dropped when their client disconnects.

## Not covered

- **Sparse / ColBERT vectors from bge-m3.** This server returns dense vectors only. For hybrid search, install
  `FlagEmbedding` in the env and swap the embedder for `BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)` with
  `encode(..., return_sparse=True, return_colbert_vecs=True)`.
- **vLLM instead of this server.** `vllm serve BAAI/bge-m3 --runner pooling` and
  `vllm serve BAAI/bge-reranker-v2-m3 --runner pooling` give more throughput for heavy batch jobs, at the cost of two
  processes on the GPU and dense-only output.

## Troubleshooting

- `conda envs_dirs[0] is ... not under ...` → do step 0, then log out and back in.
- `torch cannot see a GPU` → wrong `GPU_INDEX`, or the torch CUDA tag is newer than the driver supports. Check
  `nvidia-smi` and set `TORCH_CUDA`.
- `server exited during startup` → the last 40 lines of `logs/server.log` are printed. A common cause is the models
  not being in the cache yet (`scripts/download_models.sh`), because the server runs offline.
- CUDA OOM → lower `BATCH_TOKENS`. If a single input still fails, lower `MAX_EMB_LEN` or `MAX_RR_LEN` for that model, then run `scripts/serve.sh restart`.
- Slow answers under load → check `jobs.queued` in `scripts/serve.sh status`. Compare throughput and p95 latency with `scripts/bench.py` when tuning batch sizes.
