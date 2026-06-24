"""
Export YOLO .pt models to ONNX for 2-3x faster CPU inference.
Runs automatically on first container boot (see backend/entrypoint.sh).
Can also be run manually: MODELS_PATH=./models python scripts/export_onnx.py
"""
from ultralytics import YOLO
import os

MODELS_DIR = os.environ.get("MODELS_PATH", os.path.join(os.path.dirname(__file__), "..", "models"))
MODELS = [
    ("yolo11s.pt",      "yolo11s.onnx"),
    ("yolo11n-pose.pt", "yolo11n-pose.onnx"),
    ("weapon.pt",       "weapon.onnx"),
]

for pt_name, _ in MODELS:
    pt_path = os.path.join(MODELS_DIR, pt_name)
    if not os.path.exists(pt_path):
        print(f"  SKIP {pt_name} (not found in {MODELS_DIR})")
        continue
    print(f"  Exporting {pt_name} → ONNX …")
    m = YOLO(pt_path)
    m.export(format="onnx", imgsz=320, simplify=True, opset=17)
    print(f"  Done: {pt_path.replace('.pt', '.onnx')}")

print("\nAll exports complete. The backend prefers ONNX automatically.")
