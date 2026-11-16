#!/usr/bin/env bash
# One-time setup per GPU server: conda env "embed" with all caches on shared storage,
# torch for the server's CUDA driver, and the server's Python deps.
source "$(dirname "$0")/common.sh"

[[ -d "$STORAGE_ROOT" ]] || die "shared storage $STORAGE_ROOT does not exist (an admin must create it)"
mkdir -p "$HF_HOME/hub" "$PIP_CACHE_DIR" "$TORCH_HOME" "$LOG_DIR"

conda_init

# The env must land on shared storage, not in ~/miniconda3/envs.
first_envs_dir="$(conda config --show envs_dirs | awk '/^ *- /{print $2; exit}')"
case "$first_envs_dir" in
  "$STORAGE_ROOT"/*) ;;
  *) die "conda envs_dirs[0] is '$first_envs_dir', not under $STORAGE_ROOT.
Point conda at your storage first:
  mkdir -p $STORAGE_ROOT/conda/envs $STORAGE_ROOT/conda/pkgs
  conda config --prepend envs_dirs $STORAGE_ROOT/conda/envs
  conda config --prepend pkgs_dirs $STORAGE_ROOT/conda/pkgs
then log out and back in." ;;
esac

if env_exists; then
  log "conda env '$CONDA_ENV' already exists, reusing it"
else
  log "creating conda env '$CONDA_ENV' (python $PYTHON_VERSION)"
  conda create -y -n "$CONDA_ENV" -c conda-forge --override-channels "python=$PYTHON_VERSION"
fi

log "setting cache env vars on '$CONDA_ENV'"
conda env config vars set -n "$CONDA_ENV" \
  HF_HOME="$HF_HOME" \
  PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  TORCH_HOME="$TORCH_HOME" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID >/dev/null

activate_env

detect_cuda_tag() {
  local v n t
  v="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
  [[ -n "$v" ]] || die "could not read CUDA version from nvidia-smi; set TORCH_CUDA in config.env"
  n=$(( ${v%%.*} * 10 + ${v#*.} ))   # 12.9 -> 129
  for t in 130 129 128 126 124 121 118; do
    if (( n >= t )); then echo "cu$t"; return; fi
  done
  die "driver CUDA $v is too old for current torch wheels"
}

cuda_tag="${TORCH_CUDA:-$(detect_cuda_tag)}"
log "installing torch from the $cuda_tag wheel index"
python -m pip install --upgrade pip
python -m pip install torch --index-url "https://download.pytorch.org/whl/$cuda_tag"

# Pin the torch just installed: otherwise a dependency asking for a newer torch makes pip
# replace it with a PyPI build for a newer CUDA than this driver supports.
torch_ver="$(python -c 'import torch; print(torch.__version__)')"
log "installing server requirements (keeping torch $torch_ver)"
python -m pip install -r "$ROOT/requirements.txt" "torch==$torch_ver"

log "checking torch can see GPU $GPU_INDEX"
CUDA_VISIBLE_DEVICES="$GPU_INDEX" python - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch cannot see a GPU"
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) -> {p.name}, {p.total_memory / 2**30:.1f} GiB")
EOF

log "done. Next: scripts/download_models.sh"
