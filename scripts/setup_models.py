"""
Copy or symlink the existing .pt model files into the models/ directory
so Docker can mount them. Run from the project root.
"""
import os
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_FILES = ["yolo11s.pt", "yolo11n-pose.pt", "weapon.pt"]

for name in MODEL_FILES:
    src = os.path.join(PROJECT_ROOT, name)
    dst = os.path.join(MODELS_DIR, name)
    if os.path.exists(dst):
        print(f"  Already exists: {name}")
        continue
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copied: {name}")
    else:
        print(f"  NOT FOUND: {name} — place it in project root or models/")

print("\nDone. Model files are in models/")
print("Next: run  python scripts/export_onnx.py  to create faster ONNX versions")
