"""
Loads YOLO models once and caches them for all camera workers.
Prefers ONNX exports for faster CPU inference.
"""
import os
from functools import lru_cache

import torch
from ultralytics import YOLO

from app.core.config import settings

# Docker's CPU quota isn't visible to nproc()/sched_getaffinity inside the
# container, so PyTorch defaults its thread pool to the host's full core
# count and oversubscribes against the actual quota, which is slower than
# capping it correctly. Match this to the `cpus` limit in docker-compose.yml.
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))


def _load(name: str, task: str) -> YOLO:
    # ONNX exports don't embed task metadata like .pt checkpoints do,
    # so the task must be passed explicitly or Ultralytics silently
    # mis-detects pose models as plain detection models.
    onnx_path = os.path.join(settings.MODELS_PATH, name.replace(".pt", ".onnx"))
    pt_path   = os.path.join(settings.MODELS_PATH, name)
    if os.path.exists(onnx_path):
        return YOLO(onnx_path, task=task)
    return YOLO(pt_path, task=task)


@lru_cache(maxsize=1)
def get_models():
    import numpy as np
    person_m = _load("yolo11s.pt",      task="detect")
    pose_m   = _load("yolo11n-pose.pt", task="pose")
    weapon_m = _load("weapon.pt",       task="detect")
    # warm-up (matches the 320x240 inference size used by the pipeline)
    dummy = np.zeros((240, 320, 3), dtype=np.uint8)
    person_m(dummy, imgsz=320, verbose=False)
    pose_m(dummy,   imgsz=320, verbose=False)
    weapon_m(dummy, imgsz=320, verbose=False)
    return person_m, pose_m, weapon_m
