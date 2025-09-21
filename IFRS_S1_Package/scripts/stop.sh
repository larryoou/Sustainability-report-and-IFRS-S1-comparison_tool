#!/usr/bin/env bash
set -euo pipefail

# IFRS S1 Package - Stop all services and clean up
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PKG_DIR/logs"

echo "🛑 Stopping IFRS S1 Package"
echo "📁 Root: $PKG_DIR"

kill_from_file() {
  local file="$1"
  if [ -f "$file" ]; then
    while read -r pid; do
      if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
        echo "• Killing PID $pid"
        kill "$pid" 2>/dev/null || true
        sleep 0.5
        if kill -0 "$pid" >/dev/null 2>&1; then
          echo "  ↳ Force killing PID $pid"
          kill -9 "$pid" 2>/dev/null || true
        fi
      fi
    done < "$file"
    rm -f "$file"
  fi
}

# Kill processes from aggregated PID list
if [ -f "$PKG_DIR/.pids" ]; then
  echo "🔎 Killing processes from $PKG_DIR/.pids"
  kill_from_file "$PKG_DIR/.pids"
fi

# Kill individual PID files if present
for f in "$PKG_DIR/.accel.pid" "$PKG_DIR/.local.pid" "$PKG_DIR/.http.pid"; do
  if [ -f "$f" ]; then
    echo "🔎 Killing process from $f"
    kill_from_file "$f"
  fi
done

# Ensure ports are free (cover possible frontend port range)
for port in 8004 8000 9000 9001 9002 9003 9004 9005; do
  if lsof -ti:$port >/dev/null 2>&1; then
    echo "🧹 Port $port still in use — killing listeners"
    lsof -ti:$port | xargs kill -9 2>/dev/null || true
  fi
done

# Cleanup residual PID files
rm -f "$PKG_DIR/.pids" "$PKG_DIR/.accel.pid" "$PKG_DIR/.local.pid" "$PKG_DIR/.http.pid"

# Summary
for port in 8004 8000 9000 9001 9002 9003 9004 9005; do
  if lsof -ti:$port >/dev/null 2>&1; then
    echo "❌ Port $port still occupied"
  else
    echo "✅ Port $port is free"
  fi
done

echo "✅ Stop complete. Logs (if any): $LOG_DIR"
