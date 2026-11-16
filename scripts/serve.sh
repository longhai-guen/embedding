#!/usr/bin/env bash
# Control the embedding/rerank server.
#   scripts/serve.sh start | stop | restart | status | logs
source "$(dirname "$0")/common.sh"

PID_FILE="$LOG_DIR/server.pid"
LOG_FILE="$LOG_DIR/server.log"
mkdir -p "$LOG_DIR"

is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

check_host() {
  # 0.0.0.0 / :: listen everywhere; probe via loopback
  case "$HOST" in 0.0.0.0|::) echo 127.0.0.1 ;; *) echo "$HOST" ;; esac
}

health() {
  python - "$(check_host)" "$PORT" <<'EOF'
import sys, urllib.request
try:
    with urllib.request.urlopen(f"http://{sys.argv[1]}:{sys.argv[2]}/health", timeout=3) as r:
        print(r.read().decode())
except Exception:
    sys.exit(1)
EOF
}

ensure_key() {
  if [[ ! -s "$API_KEY_FILE" ]]; then
    (umask 077; python -c 'import secrets; print(secrets.token_urlsafe(32))' > "$API_KEY_FILE")
    log "generated API key in $API_KEY_FILE"
  fi
}

start() {
  if is_running; then
    log "already running (pid $(cat "$PID_FILE"))"
    return
  fi
  activate_env
  ensure_key

  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_INDEX"
  export HF_HUB_OFFLINE=1   # models are already in the cache; never hit the network at serve time
  export EMBED_MODEL RERANK_MODEL MAX_EMB_LEN MAX_RR_LEN EMB_BATCH RR_BATCH USE_FP16 MAX_INPUTS
  export EMBED_API_KEY_FILE="$API_KEY_FILE"

  cd "$ROOT"
  echo "===== start $(date '+%F %T') gpu=$GPU_INDEX $HOST:$PORT =====" >> "$LOG_FILE"
  nohup python -m uvicorn server:app --host "$HOST" --port "$PORT" --workers 1 >> "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  log "started pid $!, loading models (log: $LOG_FILE)"

  for _ in $(seq 1 150); do
    if ! is_running; then
      tail -n 40 "$LOG_FILE" >&2
      rm -f "$PID_FILE"
      die "server exited during startup"
    fi
    if out="$(health)"; then
      log "healthy: $out"
      return
    fi
    sleep 2
  done
  die "server not healthy after 300s; check $LOG_FILE"
}

stop() {
  if ! is_running; then
    log "not running"
    rm -f "$PID_FILE"
    return
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  kill "$pid"
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    log "pid $pid still alive after 30s, sending SIGKILL"
    kill -9 "$pid"
  fi
  rm -f "$PID_FILE"
  log "stopped"
}

status() {
  if is_running; then
    log "running (pid $(cat "$PID_FILE")) on $HOST:$PORT, GPU $GPU_INDEX"
    activate_env
    health || log "process alive but /health not answering (still loading?)"
  else
    log "not running"
  fi
  nvidia-smi -i "$GPU_INDEX" --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null || true
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  logs) tail -n 100 -f "$LOG_FILE" ;;
  *) echo "usage: $0 {start|stop|restart|status|logs}" >&2; exit 2 ;;
esac
