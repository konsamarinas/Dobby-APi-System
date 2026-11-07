"""Backend configuration and timezone helpers."""

from datetime import date, datetime
import os
from zoneinfo import ZoneInfo

BUSINESS_TIMEZONE = ZoneInfo("Europe/Athens")
ORDER_TYPES = {"dine_in", "delivery", "takeaway"}
CHANNELS = {"waiter_app", "e_menu", "kiosk"}


def cors_origins() -> list[str]:
    """Return configured browser origins, with safe local defaults."""
    configured = os.getenv("DOBBY_CORS_ORIGINS", "")
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    if not origins:
        origins = ["http://localhost:3000", "http://localhost:5173"]
    if "*" in origins:
        raise RuntimeError("DOBBY_CORS_ORIGINS must not contain '*' when credentials are enabled")
    return origins


def local_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(BUSINESS_TIMEZONE)


def athens_today() -> date:
    return datetime.now(BUSINESS_TIMEZONE).date()
