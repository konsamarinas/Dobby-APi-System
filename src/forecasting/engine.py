"""Leakage-safe forecasting models and model selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:  # pragma: no cover - exercised when optional runtime dependency is absent
    RandomForestRegressor = None

MODEL_FEATURES = (
    "day_of_week", "day_of_month", "month", "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7", "rolling_std_14",
)
FORECAST_MODELS = ("seasonal_naive", "seasonal_average", "random_forest")


@dataclass(frozen=True)
class ModelMetrics:
    mae: float | None
    rmse: float | None
    wape: float | None
    bias: float | None
    n_observations: int

    def as_dict(self) -> dict[str, Any]:
        return {key: (round(value, 4) if isinstance(value, float) else value)
                for key, value in self.__dict__.items()}


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


def seasonal_naive_forecast(series: pd.Series, target: date) -> float:
    """Return the most recent observed value for the target weekday."""
    if series.empty:
        return 0.0
    values = series[series.index.day_of_week == target.weekday()]
    return max(0.0, float(values.iloc[-1] if not values.empty else series.iloc[-1]))


def seasonal_average_forecast(series: pd.Series, target: date) -> tuple[float, float, float]:
    if series.empty:
        return 0.0, 0.0, 0.0
    values = series[series.index.day_of_week == target.weekday()].tail(8)
    quantity = float(values.mean()) if not values.empty else float(series.tail(28).mean())
    residuals = (values - quantity).abs()
    spread = float(residuals.mean()) if not residuals.empty else math.sqrt(quantity)
    return max(0.0, quantity), max(0.0, quantity - 1.28 * spread), max(quantity, quantity + 1.28 * spread)


def _feature_row(history: pd.Series, target: date) -> dict[str, float] | None:
    target_timestamp = pd.Timestamp(target)
    values = history.copy()
    required = (1, 7, 14, 28)
    lags: dict[int, float] = {}
    for lag in required:
        timestamp = target_timestamp - pd.Timedelta(days=lag)
        if timestamp not in values.index:
            return None
        lags[lag] = float(values.loc[timestamp])
    window_7 = values.loc[target_timestamp - pd.Timedelta(days=7):].iloc[:-1].tail(7)
    window_14 = values.loc[target_timestamp - pd.Timedelta(days=14):].iloc[:-1].tail(14)
    return {
        "day_of_week": float(target_timestamp.day_of_week), "day_of_month": float(target_timestamp.day),
        "month": float(target_timestamp.month), "lag_1": lags[1], "lag_7": lags[7],
        "lag_14": lags[14], "lag_28": lags[28],
        "rolling_mean_7": float(window_7.mean()), "rolling_mean_14": float(window_14.mean()),
        "rolling_std_7": float(window_7.std(ddof=0)), "rolling_std_14": float(window_14.std(ddof=0)),
    }


def feature_frame(series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    features, targets = [], []
    for timestamp in series.index[28:]:
        row = _feature_row(series.loc[:timestamp].iloc[:-1], timestamp.date())
        if row is not None:
            features.append(row)
            targets.append(float(series.loc[timestamp]))
    return pd.DataFrame(features, columns=MODEL_FEATURES), pd.Series(targets, dtype=float)


def random_forest_forecast(series: pd.Series, target: date) -> float | None:
    model = _fit_random_forest(series)
    row = _feature_row(series, target)
    if model is None or row is None:
        return None
    return max(0.0, float(model.predict(pd.DataFrame([row], columns=MODEL_FEATURES))[0]))


def _fit_random_forest(series: pd.Series) -> Any | None:
    if RandomForestRegressor is None:
        return None
    train_x, train_y = feature_frame(series)
    if len(train_x) < 28:
        return None
    model = RandomForestRegressor(
        n_estimators=40, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=1,
    )
    model.fit(train_x, train_y)
    return model


def forecast_one(series: pd.Series, target: date, model: str) -> float:
    if model == "seasonal_naive":
        return seasonal_naive_forecast(series, target)
    if model == "random_forest":
        prediction = random_forest_forecast(series, target)
        return prediction if prediction is not None else seasonal_average_forecast(series, target)[0]
    return seasonal_average_forecast(series, target)[0]


def recursive_forecast(series: pd.Series, start: date, horizon: int, model: str) -> list[tuple[date, float, float, float]]:
    history = series.copy()
    fitted_model = _fit_random_forest(history) if model == "random_forest" else None
    points = []
    for offset in range(horizon):
        target = start + timedelta(days=offset)
        row = _feature_row(history, target) if fitted_model is not None else None
        prediction = (
            max(0.0, float(fitted_model.predict(pd.DataFrame([row], columns=MODEL_FEATURES))[0]))
            if fitted_model is not None and row is not None
            else forecast_one(history, target, model)
        )
        _, lower, upper = seasonal_average_forecast(history, target)
        points.append((target, prediction, min(lower, prediction), max(upper, prediction)))
        history.loc[pd.Timestamp(target)] = prediction
    return points


def calculate_metrics(actual: list[float], predicted: list[float]) -> ModelMetrics:
    if not actual:
        return ModelMetrics(None, None, None, None, 0)
    errors = [prediction - observed for observed, prediction in zip(actual, predicted)]
    absolute = [abs(error) for error in errors]
    return ModelMetrics(
        mae=sum(absolute) / len(actual),
        rmse=math.sqrt(sum(error * error for error in errors) / len(actual)),
        wape=sum(absolute) / sum(actual) if sum(actual) else None,
        bias=sum(errors) / len(actual),
        n_observations=len(actual),
    )


def evaluate_models(series: pd.Series, test_fraction: float = 0.2) -> dict[str, ModelMetrics]:
    """Evaluate models on the same chronological holdout without future leakage."""
    if len(series) < 14:
        return {model: ModelMetrics(None, None, None, None, 0) for model in FORECAST_MODELS}
    split = min(len(series) - 1, max(1, int(len(series) * (1 - test_fraction))))
    train = series.iloc[:split]
    test = series.iloc[split:]
    results: dict[str, ModelMetrics] = {}
    for model in FORECAST_MODELS:
        history = train.copy()
        predictions = []
        for timestamp, observed in test.items():
            prediction = forecast_one(history, timestamp.date(), model)
            predictions.append(prediction)
            history.loc[timestamp] = float(observed)
        results[model] = calculate_metrics(test.tolist(), predictions)
    return results


def select_model(series: pd.Series) -> tuple[str, dict[str, ModelMetrics]]:
    validation_end = max(1, int(len(series) * 0.8))
    validation_series = series.iloc[:validation_end]
    metrics = evaluate_models(validation_series)
    eligible = [(metrics[model].wape, model) for model in FORECAST_MODELS
                if metrics[model].wape is not None]
    if len(series) < 56 or not eligible:
        return "seasonal_average", metrics
    return min(eligible, key=lambda value: (value[0], FORECAST_MODELS.index(value[1])))[1], metrics


# Backward-compatible name used by existing callers.
seasonal_forecast = seasonal_average_forecast


def forecast_points(series: pd.Series, start: date, horizon: int) -> list[tuple[date, float, float, float]]:
    return recursive_forecast(series, start, horizon, "seasonal_average")
