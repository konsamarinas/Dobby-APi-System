"""Chronological forecasting evaluation helpers."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from .engine import calculate_metrics, evaluate_models, seasonal_average_forecast, seasonal_naive_forecast


def wape(errors: float, actuals: float) -> float | None:
    return errors / actuals if actuals else None


def previous_week_baseline(history: dict[date, float], target: date) -> float:
    return history.get(target - timedelta(days=7), 0.0)


def rolling_origin_evaluation(daily_totals: dict[tuple[str, date], float],
                              test_fraction: float = 0.2) -> dict[str, Any]:
    observed_dates = sorted({day for _, day in daily_totals})
    if len(observed_dates) < 2:
        return {"wape": None, "baseline_wape": None, "test_period": None, "n_test_days": 0}
    calendar_dates = [observed_dates[0] + timedelta(days=offset)
                      for offset in range((observed_dates[-1] - observed_dates[0]).days + 1)]
    split = min(len(calendar_dates) - 1, max(1, int(len(calendar_dates) * (1 - test_fraction))))
    test_dates = calendar_dates[split:]
    actual, predictions, baseline = [], [], []
    aggregate_pred, aggregate_base = [], []
    for target in test_dates:
        actual.append(sum(value for (item_id, day), value in daily_totals.items() if day == target))
        pred_total = base_total = 0.0
        for product_id in {key[0] for key in daily_totals}:
            values = {day: value for (item_id, day), value in daily_totals.items()
                      if item_id == product_id and day < target}
            if values:
                index = pd.date_range(min(values), target - timedelta(days=1), freq="D")
                series = pd.Series([values.get(day.date(), 0.0) for day in index], index=index)
                pred_total += seasonal_average_forecast(series, target)[0]
                base_total += seasonal_naive_forecast(series, target)
        aggregate_pred.append(pred_total)
        aggregate_base.append(base_total)
    seasonal_metrics = calculate_metrics(actual, aggregate_pred)
    baseline_metrics = calculate_metrics(actual, aggregate_base)
    return {
        "wape": seasonal_metrics.wape, "baseline_wape": baseline_metrics.wape,
        "test_period": {"from": test_dates[0].isoformat(), "to": test_dates[-1].isoformat()},
        "n_test_days": len(test_dates),
    }
