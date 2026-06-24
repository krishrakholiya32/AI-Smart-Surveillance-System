import os
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.alert import Alert
from app.schemas.alert import AlertOut, AlertStatusUpdate

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("", response_model=List[AlertOut])
async def list_alerts(
    camera_id: Optional[int] = None,
    alert_type: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    _=Depends(get_current_user),
):
    q = select(Alert).order_by(desc(Alert.created_at)).offset(offset).limit(limit)
    if camera_id is not None:
        q = q.where(Alert.camera_id == camera_id)
    if alert_type:
        q = q.where(Alert.alert_type == alert_type)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/{alert_id}", response_model=AlertOut)
async def get_alert(alert_id: int, db: AsyncSession = Depends(get_db), _=Depends(get_current_user)):
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(404, "Alert not found")
    return alert


@router.get("/{alert_id}/snapshot")
async def get_snapshot(alert_id: int, db: AsyncSession = Depends(get_db), _=Depends(get_current_user)):
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert or not alert.snapshot_path or not os.path.exists(alert.snapshot_path):
        raise HTTPException(404, "Snapshot not found")
    return FileResponse(alert.snapshot_path, media_type="image/jpeg")


@router.patch("/{alert_id}", response_model=AlertOut)
async def update_alert_status(
    alert_id: int,
    payload: AlertStatusUpdate,
    db: AsyncSession = Depends(get_db),
    _=Depends(get_current_user),
):
    if payload.status not in ("confirmed", "dismissed"):
        raise HTTPException(400, "status must be 'confirmed' or 'dismissed'")
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(404, "Alert not found")
    alert.status = payload.status
    await db.commit()
    await db.refresh(alert)
    return alert


@router.delete("/{alert_id}", status_code=204)
async def delete_alert(alert_id: int, db: AsyncSession = Depends(get_db), _=Depends(get_current_user)):
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(404, "Alert not found")
    if alert.snapshot_path and os.path.exists(alert.snapshot_path):
        os.remove(alert.snapshot_path)
    await db.delete(alert)
    await db.commit()
