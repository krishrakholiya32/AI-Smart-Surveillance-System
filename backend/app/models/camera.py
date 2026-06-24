from sqlalchemy import String, Boolean, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    source: Mapped[str] = mapped_column(String(512))  # "0", "1", or RTSP/HTTP URL
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    width: Mapped[int] = mapped_column(Integer, default=640)
    height: Mapped[int] = mapped_column(Integer, default=480)

    zones: Mapped[list["Zone"]] = relationship("Zone", back_populates="camera", cascade="all, delete-orphan")
    alerts: Mapped[list["Alert"]] = relationship("Alert", back_populates="camera", cascade="all, delete-orphan")
    events: Mapped[list["Event"]] = relationship("Event", back_populates="camera", cascade="all, delete-orphan")
