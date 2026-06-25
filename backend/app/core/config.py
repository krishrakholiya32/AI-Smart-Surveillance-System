from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "AI Smart Surveillance System"
    SECRET_KEY: str  # required — set in .env, no insecure default
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:80",
    ]

    DATABASE_URL: str = "postgresql+asyncpg://surveillance:surveillance@db:5432/surveillance"

    STORAGE_PATH: str = "/app/storage/alerts"
    MODELS_PATH: str = "/app/models"

    CONF_PERSON: float = 0.50
    CONF_WEAPON: float = 0.65
    RUN_THRESH_NORM: float = 0.18
    LOITER_SECS: int = 8
    CD_ZONE_BODY: float = 8.0
    CD_WEAPON: float = 10.0
    CD_RUN: float = 6.0
    CD_CROWD: float = 15.0
    CROWD_LIMIT: int = 5

    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str  # required — set in .env, no insecure default

    class Config:
        env_file = ".env"


settings = Settings()

# Hardcoded fallback values for the Settings UI's "reset to default" action.
# Intentionally independent of .env — on a fresh deploy (e.g. Oracle), .env
# may itself hold tuned values, and "reset" should mean "back to factory",
# not "back to whatever .env currently says".
DEFAULT_THRESHOLDS = {
    "CONF_PERSON": 0.50,
    "CONF_WEAPON": 0.65,
    "RUN_THRESH_NORM": 0.18,
    "LOITER_SECS": 8,
    "CROWD_LIMIT": 5,
}
