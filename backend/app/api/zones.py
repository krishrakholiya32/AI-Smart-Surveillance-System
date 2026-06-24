from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.zone import Zone
from app.schemas.zone import ZoneCreate, ZoneOut, ZoneUpdate
from app.services import camera_manager

router = APIRouter(prefix="/api/zones", tags=["zones"])


async def _push_zones(db: AsyncSession, camera_id: int):
    result = await db.execute(
        select(Zone).where(Zone.camera_id == camera_id, Zone.is_active == True)
    )
    zones = [{"id": z.id, "name": z.name, "points": z.points, "color": z.color}
             for z in result.scalars().all()]
    camera_manager.update_zones(camera_id, zones)


@router.get("", response_model=List[ZoneOut])
async def list_zones(camera_id: int | None = None, db: AsyncSession = Depends(get_db), _=Depends(get_current_user)):
    q = select(Zone)
    if camera_id is not None:
        q = q.where(Zone.camera_id == camera_id)
    result = await db.execute(q)
    return result.scalars().all()


@router.post("", response_model=ZoneOut, status_code=201)
async def create_zone(payload: ZoneCreate, db: AsyncSession = Depends(get_db), _=Depends(get_current_user)):
    zone = Zone(**payload.model_dump())
    db.add(zone)
    await db.commit()
    await db.refresh(zone)
    await _push_zones(db, zone.camera_id)
    return zone


@router.patch("/{zone_id}", response_model=ZoneOut)
async def update_zone(zone_id: int, payload: ZoneUpdate, db: AsyncSession = Depends(get_db), _=Depends(get_current_user)):
    result = await db.execute(select(Zone).where(Zone.id == zone_id))
    zone = result.scalar_one_or_none()
    if not zone:
        raise HTTPException(404, "Zone not found")
    for k, v in payload.model_dump(exclude_none=True).items():
        setattr(zone, k, v)
    await db.commit()
    await db.refresh(zone)
    await _push_zones(db, zone.camera_id)
    return zone


@router.delete("/{zone_id}", status_code=204)
async def delete_zone(zone_id: int, db: AsyncSession = Depends(get_db), _=Depends(get_current_user)):
    result = await db.execute(select(Zone).where(Zone.id == zone_id))
    zone = result.scalar_one_or_none()
    if not zone:
        raise HTTPException(404, "Zone not found")
    camera_id = zone.camera_id
    await db.delete(zone)
    await db.commit()
    await _push_zones(db, camera_id)
