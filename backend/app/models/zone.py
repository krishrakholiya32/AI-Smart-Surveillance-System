from sqlalchemy import String, ForeignKey, JSON, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Zone(Base):
    __tablename__ = "zones"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"))
    name: Mapped[str] = mapped_column(String(128), default="Zone 1")
    # List of [x, y] pairs as JSON: [[x1,y1],[x2,y2],...]
    points: Mapped[list] = mapped_column(JSON)
    color: Mapped[str] = mapped_column(String(16), default="#0050dc")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    camera: Mapped["Camera"] = relationship("Camera", back_populates="zones")
