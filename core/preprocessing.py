"""Timezone-safe preprocessing for Dobby order data."""

from __future__ import annotations

from datetime import date
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

BUSINESS_TIMEZONE = ZoneInfo("Europe/Athens")


def local_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return pd.NaT
    return timestamp.tz_convert(BUSINESS_TIMEZONE).tz_localize(None)


def utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    return timestamp if not pd.isna(timestamp) else pd.NaT


def local_date(value: Any) -> date | None:
    timestamp = local_timestamp(value)
    return timestamp.date() if pd.notna(timestamp) else None


def item_rows(orders_by_business: dict[str, list[dict[str, Any]]],
              business_id: str, order_type: str | None = None) -> pd.DataFrame:
    rows = []
    for order in orders_by_business.get(business_id, []):
        if order_type and order.get("type") != order_type:
            continue
        timestamp = local_timestamp(order.get("created_at"))
        for item in order.get("items") or []:
            rows.append({"order_id": order.get("id"), "product_id": item.get("product_id"),
                         "category_id": (item.get("category") or {}).get("id"),
                         "quantity": int(item.get("quantity", 0)), "timestamp": timestamp})
    return pd.DataFrame(rows, columns=["order_id", "product_id", "category_id", "quantity", "timestamp"])
