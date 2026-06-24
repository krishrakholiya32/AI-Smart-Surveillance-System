import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import create_tables
from app.core.security import hash_password
from app.models.user import User
from app.core.database import AsyncSessionLocal
from sqlalchemy import select

from app.api import auth, cameras, zones, alerts, events, stream, settings as settings_api
from app.models.camera import Camera
from app.models.zone import Zone
from app.services import camera_manager
from app.services.alert_service import save_alert

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await create_tables()
    await _seed_admin()
    await _autostart_cameras()
    log.info("AI Smart Surveillance System started")
    yield
    # Shutdown
    camera_manager.stop_all()
    log.info("All cameras stopped")


async def _autostart_cameras():
    from sqlalchemy import select as sa_select
    async with AsyncSessionLocal() as db:
        result = await db.execute(sa_select(Camera).where(Camera.is_active == True))
        active_cams = result.scalars().all()
    if not active_cams:
        return
    loop = asyncio.get_event_loop()
    for cam in active_cams:
        async with AsyncSessionLocal() as db:
            zr = await db.execute(
                sa_select(Zone).where(Zone.camera_id == cam.id, Zone.is_active == True)
            )
            zones = [{"id": z.id, "name": z.name, "points": z.points, "color": z.color}
                     for z in zr.scalars().all()]

        async def _alert_cb(cam_id, alert_data, snapshot, _cam_id=cam.id):
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

        camera_manager.start_camera(cam.id, cam.source, zones, loop, _alert_cb)
        log.info(f"Auto-started camera {cam.id} ({cam.name})")


async def _seed_admin():
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.username == settings.ADMIN_USERNAME))
        if not result.scalar_one_or_none():
            admin = User(
                username=settings.ADMIN_USERNAME,
                hashed_password=hash_password(settings.ADMIN_PASSWORD),
            )
            db.add(admin)
            await db.commit()
            log.info(f"Admin user '{settings.ADMIN_USERNAME}' created")


app = FastAPI(
    title="AI Smart Surveillance System",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(cameras.router)
app.include_router(zones.router)
app.include_router(alerts.router)
app.include_router(events.router)
app.include_router(stream.router)
app.include_router(settings_api.router)


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}
