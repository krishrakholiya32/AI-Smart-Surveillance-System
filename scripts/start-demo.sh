#!/bin/sh
# Brings the app up for a local, in-person demo: Docker Desktop, the
# webcam MJPEG bridge (so the container can see your laptop's webcam),
# and the docker-compose stack. Run scripts/stop-demo.sh when you're done.
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ── Docker Desktop ───────────────────────────────────────────────────
if ! docker info >/dev/null 2>&1; then
    echo "Starting Docker Desktop..."
    "/c/Program Files/Docker/Docker/Docker Desktop.exe" &
    disown
    until docker info >/dev/null 2>&1; do sleep 2; done
    echo "Docker is ready."
fi

# ── Webcam bridge (skip if you're using DroidCam/RTSP instead) ──────
if ! curl -s -o /dev/null http://localhost:8765/health 2>/dev/null; then
    echo "Starting webcam bridge on :8765..."
    python "$PROJECT_DIR/scripts/webcam_server.py" >/tmp/webcam_server.log 2>&1 &
    disown
    sleep 2
fi

# ── App stack ─────────────────────────────────────────────────────────
docker compose up -d

echo
echo "Ready: http://localhost"
echo "Stop everything with: scripts/stop-demo.sh"
