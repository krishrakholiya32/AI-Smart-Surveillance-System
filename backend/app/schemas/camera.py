from pydantic import BaseModel
from typing import Optional


class CameraCreate(BaseModel):
    name: str
    source: str
    width: int = 640
    height: int = 480


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    source: Optional[str] = None
    is_active: Optional[bool] = None
    width: Optional[int] = None
    height: Optional[int] = None


class CameraOut(BaseModel):
    id: int
    name: str
    source: str
    is_active: bool
    width: int
    height: int

    class Config:
        from_attributes = True
