"""Existing seasonal forecasting algorithm extracted from the API routes."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd


def forecast_series(rows: pd.DataFrame, level: str, identifier: str) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=float)
    key = "product_id" if level == "product" else "category_id"
    selected = rows[rows[key] == identifier].dropna(subset=["timestamp"])
    if selected.empty:
        return pd.Series(dtype=float)
    series = selected.groupby(selected["timestamp"].dt.floor("D"))["quantity"].sum().sort_index()
    index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(index, fill_value=0.0).astype(float)


def seasonal_forecast(series: pd.Series, target: date) -> tuple[float, float, float]:
    if series.empty:
        return 0.0, 0.0, 0.0
    values = series[series.index.day_of_week == target.weekday()]
    quantity = float(values.tail(8).mean()) if not values.empty else float(series.tail(28).mean())
    residuals = (values - quantity).abs()
    spread = float(residuals.tail(8).mean()) if not residuals.empty else math.sqrt(quantity)
    return max(0.0, quantity), max(0.0, quantity - 1.28 * spread), max(quantity, quantity + 1.28 * spread)


def forecast_points(series: pd.Series, start: date, horizon: int) -> list[tuple[date, float, float, float]]:
    return [(target, *seasonal_forecast(series, target))
            for target in (start + timedelta(days=offset) for offset in range(horizon))]
