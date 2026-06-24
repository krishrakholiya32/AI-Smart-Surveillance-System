#!/bin/sh
set -e

if [ ! -f "$MODELS_PATH/yolo11s.onnx" ]; then
    echo "ONNX exports not found in $MODELS_PATH — exporting now (one-time, ~1-2 min)..."
    python /app/scripts/export_onnx.py
fi

exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
