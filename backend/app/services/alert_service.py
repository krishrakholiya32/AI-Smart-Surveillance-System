import os
import time
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.alert import Alert
from app.models.event import Event
from app.core.config import settings


async def save_alert(
    db: AsyncSession,
    camera_id: int,
    alert_type: str,
    message: str,
    person_id: Optional[int] = None,
    snapshot_bytes: Optional[bytes] = None,
    meta: Optional[dict] = None,
    status: str = "confirmed",
):
    snapshot_path = None
    if snapshot_bytes:
        os.makedirs(settings.STORAGE_PATH, exist_ok=True)
        fname = f"{alert_type}_{camera_id}_{int(time.time())}.jpg"
        fpath = os.path.join(settings.STORAGE_PATH, fname)
        with open(fpath, "wb") as f:
            f.write(snapshot_bytes)
        snapshot_path = fpath

    alert = Alert(
        camera_id=camera_id,
        alert_type=alert_type,
        message=message,
        person_id=person_id,
        snapshot_path=snapshot_path,
        meta=meta,
        status=status,
    )
    db.add(alert)

    event = Event(camera_id=camera_id, level="alert", message=message)
    db.add(event)

    await db.commit()
    return alert


async def log_event(db: AsyncSession, message: str, level: str = "info", camera_id: Optional[int] = None):
    event = Event(camera_id=camera_id, level=level, message=message)
    db.add(event)
    await db.commit()
