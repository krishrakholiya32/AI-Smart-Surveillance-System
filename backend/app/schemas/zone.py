from pydantic import BaseModel
from typing import List, Optional


class ZoneCreate(BaseModel):
    camera_id: int
    name: str = "Zone 1"
    points: List[List[float]]   # [[x,y], [x,y], ...]
    color: str = "#0050dc"


class ZoneUpdate(BaseModel):
    name: Optional[str] = None
    points: Optional[List[List[float]]] = None
    color: Optional[str] = None
    is_active: Optional[bool] = None


class ZoneOut(BaseModel):
    id: int
    camera_id: int
    name: str
    points: List[List[float]]
    color: str
    is_active: bool

    class Config:
        from_attributes = True
