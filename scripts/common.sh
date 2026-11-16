#!/usr/bin/env bash
# Shared helpers for the scripts in this folder: load config, activate conda.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "error: $*" >&2; exit 1; }

[[ -f "$ROOT/config.env" ]] || die "config.env not found. Create it on this server:
  cp $ROOT/config.env.example $ROOT/config.env   # then edit the paths"
# shellcheck disable=SC1091
source "$ROOT/config.env"
[[ "$STORAGE_ROOT" != /path/to/* ]] || die "edit STORAGE_ROOT in $ROOT/config.env (still the placeholder)"
# Export so Python (hf, pip, torch, the server) sees them even if the conda env lacks these vars.
export HF_HOME PIP_CACHE_DIR TORCH_HOME

conda_init() {
  local base
  if command -v conda >/dev/null 2>&1; then
    base="$(conda info --base)"
  elif [[ -d "$HOME/miniconda3" ]]; then
    base="$HOME/miniconda3"
  else
    die "conda not found; install Miniconda first (see README)"
  fi
  # shellcheck disable=SC1091
  source "$base/etc/profile.d/conda.sh"
}

env_exists() {
  conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"
}

activate_env() {
  conda_init
  env_exists || die "conda env '$CONDA_ENV' not found; run scripts/setup_env.sh first"
  # conda's activate hooks reference unset variables
  set +u
  conda activate "$CONDA_ENV"
  set -u
  # config.env wins over any cache vars stored on the conda env
  # shellcheck disable=SC1091
  source "$ROOT/config.env"
  export HF_HOME PIP_CACHE_DIR TORCH_HOME
}
