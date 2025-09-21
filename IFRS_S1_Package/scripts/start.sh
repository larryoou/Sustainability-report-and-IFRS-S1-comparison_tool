#!/usr/bin/env bash
set -euo pipefail

# IFRS S1 Package - Start all services and frontend
# Ports:
#   8004 - Accelerated backend (required)
#   8000 - Local semantic proxy (optional)
#   9000 - Frontend HTTP server

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PKG_DIR/logs"
FRONTEND_DIR="$PKG_DIR/frontend"
BACKEND_DIR="$PKG_DIR/backend"

mkdir -p "$LOG_DIR" "$PKG_DIR/vector_cache"

echo "🌱 IFRS S1 Package"
echo "📁 Root: $PKG_DIR"

# Clean ports if occupied
for port in 8004 8000 9000; do
  if lsof -ti:$port >/dev/null 2>&1; then
    echo "⚠️  Port $port in use. Killing..."
    lsof -ti:$port | xargs kill -9 2>/dev/null || true
  fi
done

# Start accelerated backend (8004)
echo "🚀 Starting Accelerated Backend (8004) ..."
(
  cd "$BACKEND_DIR"
  nohup python3 accelerated_dual_model_service.py > "$LOG_DIR/accelerated_service.log" 2>&1 &
  ACCEL_PID=$!
  echo $ACCEL_PID > "$PKG_DIR/.accel.pid"
)
ACCEL_PID=$(cat "$PKG_DIR/.accel.pid")
sleep 2

# Start local semantic proxy (8000)
echo "🚀 Starting Local Semantic Proxy (8000) ..."
# Use uvicorn directly to avoid reload spawning; run from backend dir for reliable imports
(
  cd "$BACKEND_DIR"
  nohup python3 -m uvicorn local_semantic_service:app --host 0.0.0.0 --port 8000 --log-level info > "$LOG_DIR/local_service.log" 2>&1 &
  echo $! > "$PKG_DIR/.local.pid"
)
LOCAL_PID=$(cat "$PKG_DIR/.local.pid")
sleep 2

# Pick frontend port 9000-9005
HTTP_PORT=""
for p in 9000 9001 9002 9003 9004 9005; do
  if ! lsof -ti:$p >/dev/null 2>&1; then
    HTTP_PORT=$p
    break
  fi
done
HTTP_PORT=${HTTP_PORT:-9000}

# Start static HTTP server for frontend
if [ -d "$FRONTEND_DIR" ]; then
  echo "🌐 Starting Frontend HTTP server on $HTTP_PORT ..."
  (
    cd "$FRONTEND_DIR"
    nohup python3 -m http.server "$HTTP_PORT" > "$LOG_DIR/http_server.log" 2>&1 &
    HTTP_PID=$!
    echo $HTTP_PID > "$PKG_DIR/.http.pid"
  )
  HTTP_PID=$(cat "$PKG_DIR/.http.pid")
else
  echo "❌ Frontend directory not found: $FRONTEND_DIR"
  HTTP_PID=""
fi

# Save PIDs
{
  echo "$ACCEL_PID"
  echo "$LOCAL_PID"
  if [ -n "${HTTP_PID:-}" ]; then echo "$HTTP_PID"; fi
} > "$PKG_DIR/.pids"

# Health summary
HEALTHY=0
for port in 8004 8000 "$HTTP_PORT"; do
  if [ -n "$port" ]; then
    if lsof -i:$port >/dev/null 2>&1; then
      echo "✅ Port $port is active"
      ((HEALTHY++))
    else
      echo "❌ Port $port is not responding"
    fi
  fi
done

echo "📊 Services up: $HEALTHY/3"
FRONTEND_URL="http://localhost:$HTTP_PORT/ifrs_s1_auto_keywords_tool.html"
[ -n "$HTTP_PORT" ] && echo "➡️  Open: $FRONTEND_URL"

echo "⏳ Sleeping 3s for readiness..." && sleep 3

# 自動打開瀏覽器（由主啟動腳本處理）
echo "🌐 瀏覽器將由主啟動腳本自動打開..."

# Trap for quick stop
trap 'echo; echo "🛑 Stopping..."; [ -f "$PKG_DIR/.pids" ] && xargs kill -9 < "$PKG_DIR/.pids" 2>/dev/null || true; rm -f "$PKG_DIR/.pids" "$PKG_DIR/.accel.pid" "$PKG_DIR/.local.pid" "$PKG_DIR/.http.pid"; echo "✅ Stopped"; exit 0' INT

echo "⚡ Running. Press Ctrl+C to stop. Logs: $LOG_DIR"
while true; do sleep 10; done
