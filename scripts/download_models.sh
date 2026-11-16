#!/usr/bin/env bash
# Get both models into the HF cache and verify they load offline.
# With SHARED_HUB set: reuse/download them in the team-shared hub and symlink into $HF_HOME/hub.
# With SHARED_HUB empty: download straight into $HF_HOME/hub.
source "$(dirname "$0")/common.sh"
activate_env

HUB="$HF_HOME/hub"
mkdir -p "$HUB"
if [[ -n "$SHARED_HUB" ]]; then
  [[ -d "$SHARED_HUB" ]] || die "SHARED_HUB $SHARED_HUB not found"
fi

# onnx/ is ~2.3 GB for bge-m3 and unused here
EXCLUDE="onnx/*"

if command -v hf >/dev/null 2>&1; then
  HF_DL=(hf download)
else
  HF_DL=(huggingface-cli download)
fi

for model in "$EMBED_MODEL" "$RERANK_MODEL"; do
  name="models--${model//\//--}"
  link="$HUB/$name"

  if [[ -z "$SHARED_HUB" ]]; then
    log "$model: downloading into $HUB"
    HF_HUB_CACHE="$HUB" "${HF_DL[@]}" "$model" --exclude "$EXCLUDE"
    target="$link"
  else
    shared="$SHARED_HUB/$name"
    if [[ -d "$shared" ]]; then
      log "$model: already in shared hub, skipping download"
    else
      log "$model: downloading into shared hub"
      HF_HUB_CACHE="$SHARED_HUB" "${HF_DL[@]}" "$model" --exclude "$EXCLUDE"
    fi

    if [[ -L "$link" ]]; then
      log "$model: symlink exists -> $(readlink "$link")"
    elif [[ -e "$link" ]]; then
      die "$link is a real directory, not a symlink. Move it to the shared hub first:
  mv $link $SHARED_HUB/ && ln -s $shared $link"
    else
      ln -s "$shared" "$link"
      log "$model: linked $link -> $shared"
    fi
    target="$shared"
  fi

  broken="$(find "$target/snapshots" -xtype l | wc -l | tr -d ' ')"
  [[ "$broken" == 0 ]] || die "$model: $broken broken symlinks under $target/snapshots"
  log "$model: size $(du -shL "$target" | cut -f1), no broken links"

  # same exclude as the download, or newer huggingface_hub reports the skipped files as missing
  HF_HOME="$HF_HOME" python - "$model" "$EXCLUDE" <<'EOF'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(sys.argv[1], local_files_only=True, ignore_patterns=[sys.argv[2]])
print("  offline snapshot:", path)
EOF
done

log "done. Next: scripts/serve.sh start"
