from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.event import Event
from app.schemas.event import EventOut

router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("", response_model=List[EventOut])
async def list_events(
    camera_id: Optional[int] = None,
    level: Optional[str] = None,
    limit: int = Query(100, le=500),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    _=Depends(get_current_user),
):
    q = select(Event).order_by(desc(Event.created_at)).offset(offset).limit(limit)
    if camera_id is not None:
        q = q.where(Event.camera_id == camera_id)
    if level:
        q = q.where(Event.level == level)
    result = await db.execute(q)
    return result.scalars().all()
