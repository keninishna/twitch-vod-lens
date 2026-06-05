#!/usr/bin/env bash
set -euo pipefail

BEE_ROOT="/home/john/beellama-prebuilt/v0.3.1"
BEE_BIN="$BEE_ROOT/llama-server"
MODEL_DIR="/home/john/models/bee-qwen36-27b"
LOG_PATH="/home/john/bee_prebuilt_v031.log"
PID_PATH="/home/john/bee_prebuilt_v031.pid"

pkill -f "$BEE_BIN" 2>/dev/null || true
sleep 2

export LD_LIBRARY_PATH="/home/john/.local/lib/python3.12/site-packages/nvidia/cu13/lib:$BEE_ROOT:${LD_LIBRARY_PATH:-}"

nohup "$BEE_BIN" \
  -m "$MODEL_DIR/Qwen3.6-27B-Q5_K_S.gguf" \
  --mmproj "$MODEL_DIR/mmproj-BF16.gguf" \
  --spec-draft-model "$MODEL_DIR/dflash-draft-3.6-q4_k_m.gguf" \
  --spec-type dflash \
  --spec-dflash-cross-ctx 1024 \
  --host 0.0.0.0 \
  --port 8082 \
  -np 1 \
  --kv-unified \
  -ngl all \
  --spec-draft-ngl all \
  -b 2048 \
  -ub 512 \
  --ctx-size 102400 \
  --cache-type-k q5_0 \
  --cache-type-v q4_1 \
  --flash-attn on \
  --jinja \
  --no-mmap \
  --mlock \
  --no-host \
  --reasoning on \
  --chat-template-kwargs '{"preserve_thinking":true}' \
  --temp 0.6 \
  --top-k 20 \
  --top-p 1.0 \
  --min-p 0.0 \
  > "$LOG_PATH" 2>&1 &

echo $! > "$PID_PATH"
echo "PID=$(cat "$PID_PATH")"
echo "LOG=$LOG_PATH"
