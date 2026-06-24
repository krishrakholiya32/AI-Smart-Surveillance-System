import asyncio
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.camera import Camera
from app.models.zone import Zone
from app.schemas.camera import CameraCreate, CameraOut, CameraUpdate
from app.services import camera_manager

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@router.get("", response_model=List[CameraOut])
async def list_cameras(db: AsyncSession = Depends(get_db), _=Depends(get_current_user)):
    result = await db.execute(select(Camera))
    return result.scalars().all()


@router.post("", response_model=CameraOut, status_code=201)
async def create_camera(payload: CameraCreate, db: AsyncSession = Depends(get_db), _=Depends(get_current_user)):
    cam = Camera(**payload.model_dump())
    db.add(cam)
    await db.commit()
    await db.refresh(cam)
    return cam


@router.get("/{camera_id}", response_model=CameraOut)
async def get_camera(camera_id: int, db: AsyncSession = Depends(get_db), _=Depends(get_current_user)):
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = result.scalar_one_or_none()
    if not cam:
        raise HTTPException(404, "Camera not found")
    return cam


@router.patch("/{camera_id}", response_model=CameraOut)
async def update_camera(camera_id: int, payload: CameraUpdate, db: AsyncSession = Depends(get_db), _=Depends(get_current_user)):
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = result.scalar_one_or_none()
    if not cam:
        raise HTTPException(404, "Camera not found")
    for k, v in payload.model_dump(exclude_none=True).items():
        setattr(cam, k, v)
    await db.commit()
    await db.refresh(cam)
    return cam


@router.delete("/{camera_id}", status_code=204)
async def delete_camera(camera_id: int, db: AsyncSession = Depends(get_db), _=Depends(get_current_user)):
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = result.scalar_one_or_none()
    if not cam:
        raise HTTPException(404, "Camera not found")
    camera_manager.stop_camera(camera_id)
    await db.delete(cam)
    await db.commit()


@router.post("/{camera_id}/start", status_code=200)
async def start_camera(camera_id: int, db: AsyncSession = Depends(get_db), _=Depends(get_current_user)):
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = result.scalar_one_or_none()
    if not cam:
        raise HTTPException(404, "Camera not found")

    zone_result = await db.execute(select(Zone).where(Zone.camera_id == camera_id, Zone.is_active == True))
    zones = [{"id": z.id, "name": z.name, "points": z.points, "color": z.color}
             for z in zone_result.scalars().all()]

    loop = asyncio.get_event_loop()

    from app.services.alert_service import save_alert
    from app.core.database import AsyncSessionLocal

    async def alert_cb(cam_id, alert_data, snapshot):
        async with AsyncSessionLocal() as session:
            await save_alert(
                session,
                camera_id=cam_id,
                alert_type=alert_data["type"],
                message=alert_data["message"],
                person_id=alert_data.get("person_id"),
                snapshot_bytes=snapshot,
                meta=alert_data.get("meta"),
            )

    camera_manager.start_camera(camera_id, cam.source, zones, loop, alert_cb)
    cam.is_active = True
    await db.commit()
    return {"status": "started", "camera_id": camera_id}


@router.post("/{camera_id}/stop", status_code=200)
async def stop_camera(camera_id: int, db: AsyncSession = Depends(get_db), _=Depends(get_current_user)):
    camera_manager.stop_camera(camera_id)
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = result.scalar_one_or_none()
    if cam:
        cam.is_active = False
        await db.commit()
    return {"status": "stopped", "camera_id": camera_id}


@router.get("/{camera_id}/status")
async def camera_status(camera_id: int, _=Depends(get_current_user)):
    return {"camera_id": camera_id, "running": camera_manager.is_running(camera_id)}
