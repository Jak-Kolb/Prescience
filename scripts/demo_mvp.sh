#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:-0}"
MODEL="${2:-yolov8n.pt}"
LINE_ID="${3:-line-1}"
DEVICE_ID="${4:-device-1}"
CONFIG="${5:-configs/default.yaml}"

cleanup() {
  if [[ -n "${CLOUD_PID:-}" ]] && kill -0 "$CLOUD_PID" 2>/dev/null; then
    kill "$CLOUD_PID" || true
  fi
}
trap cleanup EXIT

prescience cloud serve --host 127.0.0.1 --port 8000 --config "$CONFIG" &
CLOUD_PID=$!

echo "Cloud running at http://127.0.0.1:8000 (pid=$CLOUD_PID)"
sleep 1.5

prescience run \
  --source "$SOURCE" \
  --model "$MODEL" \
  --line-id "$LINE_ID" \
  --device-id "$DEVICE_ID" \
  --config "$CONFIG" \
  --event-endpoint "http://127.0.0.1:8000/events"
