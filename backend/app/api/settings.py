import os

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.core.config import settings, DEFAULT_THRESHOLDS

router = APIRouter(prefix="/api/settings", tags=["settings"])


def _thresholds_dict():
    return {
        "CONF_PERSON": settings.CONF_PERSON,
        "CONF_WEAPON": settings.CONF_WEAPON,
        "RUN_THRESH_NORM": settings.RUN_THRESH_NORM,
        "LOITER_SECS": settings.LOITER_SECS,
        "CROWD_LIMIT": settings.CROWD_LIMIT,
    }


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
        "thresholds": _thresholds_dict(),
        "onnx_active": onnx_active,
    }


class ThresholdsUpdate(BaseModel):
    CONF_PERSON: float | None = Field(None, ge=0.0, le=1.0)
    CONF_WEAPON: float | None = Field(None, ge=0.0, le=1.0)
    RUN_THRESH_NORM: float | None = Field(None, gt=0.0)
    LOITER_SECS: int | None = Field(None, ge=0)
    CROWD_LIMIT: int | None = Field(None, ge=1)


@router.patch("/thresholds")
async def update_thresholds(payload: ThresholdsUpdate, _=Depends(get_current_user)):
    # Mutates the shared settings singleton — every running camera worker
    # reads `self.settings` (the same object) each frame, so this takes
    # effect immediately with no restart needed.
    for key, value in payload.model_dump(exclude_none=True).items():
        setattr(settings, key, value)
    return {"thresholds": _thresholds_dict()}


@router.post("/thresholds/reset")
async def reset_thresholds(_=Depends(get_current_user)):
    for key, value in DEFAULT_THRESHOLDS.items():
        setattr(settings, key, value)
    return {"thresholds": _thresholds_dict()}
