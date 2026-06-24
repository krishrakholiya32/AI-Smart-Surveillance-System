import os

from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.core.config import settings

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("")
async def get_settings(_=Depends(get_current_user)):
    onnx_models = {
        "person": "yolo11s.onnx",
        "pose": "yolo11n-pose.onnx",
        "weapon": "weapon.onnx",
    }
    onnx_active = {
        name: os.path.exists(os.path.join(settings.MODELS_PATH, fname))
        for name, fname in onnx_models.items()
    }
    return {
        "thresholds": {
            "CONF_PERSON": settings.CONF_PERSON,
            "CONF_WEAPON": settings.CONF_WEAPON,
            "RUN_THRESH_NORM": settings.RUN_THRESH_NORM,
            "LOITER_SECS": settings.LOITER_SECS,
            "CROWD_LIMIT": settings.CROWD_LIMIT,
        },
        "onnx_active": onnx_active,
    }
