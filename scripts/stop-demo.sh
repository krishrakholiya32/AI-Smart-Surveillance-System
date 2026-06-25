#!/bin/sh
# Tears down everything scripts/start-demo.sh brought up.
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

docker compose down

# Kill the webcam bridge if it's running. It's a native Windows python.exe
# process started detached from this shell, so `ps`/`kill` can't see or
# signal it reliably — find it by the port it's listening on instead.
PID=$(netstat -ano 2>/dev/null | grep ":8765 .*LISTENING" | awk '{print $NF}' | head -1)
if [ -n "$PID" ]; then
    echo "Stopping webcam bridge (PID $PID)..."
    taskkill //F //PID "$PID" >/dev/null 2>&1 || true
fi

echo "Stopped."
