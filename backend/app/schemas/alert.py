from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any


class AlertOut(BaseModel):
    id: int
    camera_id: int
    alert_type: str
    message: str
    person_id: Optional[int]
    snapshot_path: Optional[str]
    meta: Optional[Dict[str, Any]]
    created_at: datetime

    class Config:
        from_attributes = True
