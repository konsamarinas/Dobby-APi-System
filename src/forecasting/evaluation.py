"""Leakage-safe rolling-origin forecasting evaluation."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from .engine import seasonal_forecast


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
    first_date, last_date = test_dates[0], test_dates[-1]
    forecast_errors = baseline_errors = actuals = 0.0
    for target in test_dates:
        # This history is the only information available at this origin.
        history_end = target - timedelta(days=1)
        for product_id in {key[0] for key in daily_totals}:
            history = {
                day: daily_totals[(product_id, day)]
                for item_id, day in daily_totals
                if item_id == product_id and day <= history_end
            }
            if not history:
                continue
            start = min(history)
            index = [start + timedelta(days=offset) for offset in range((history_end - start).days + 1)]
            series = pd.Series([history.get(day, 0.0) for day in index],
                               index=pd.date_range(start, history_end, freq="D"))
            prediction, _, _ = seasonal_forecast(series, target)
            previous_week = previous_week_baseline(history, target)
            actual = daily_totals.get((product_id, target), 0.0)
            forecast_errors += abs(actual - prediction)
            baseline_errors += abs(actual - previous_week)
            actuals += actual
    return {"wape": wape(forecast_errors, actuals), "baseline_wape": wape(baseline_errors, actuals),
            "test_period": {"from": first_date.isoformat(), "to": last_date.isoformat()},
            "n_test_days": len(test_dates)}
