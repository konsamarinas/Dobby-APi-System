"""Dobby API using the documented nested products and orders formats."""

from __future__ import annotations

import json
import logging
import math
from copy import deepcopy
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from .core.config import athens_today, cors_origins
from .core.data_loader import load_datasets
from .core.preprocessing import item_rows, local_date, local_timestamp, utc_timestamp
from .forecasting.evaluation import rolling_origin_evaluation

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REC_MODEL_VERSION = "rec-v1.3"
FORECAST_MODEL_VERSION = "fc-v1.0"
ORDER_TYPES = {"dine_in", "delivery", "takeaway"}
CHANNELS = {"waiter_app", "e_menu", "kiosk"}
BUSINESS_TIMEZONE = ZoneInfo("Europe/Athens")

app = FastAPI(title="Dobby Restaurant Management API", version=REC_MODEL_VERSION,
              description="API προτάσεων και πρόβλεψης ζήτησης για το dobby.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
logger = logging.getLogger(__name__)


class CartItem(BaseModel):
    product_id: str
    quantity: int = Field(ge=1)
    selection_option_ids: list[str] = Field(default_factory=list)
    extra_ids: list[str] = Field(default_factory=list)


class RecommendationRequest(BaseModel):
    business_id: str
    order_type: Literal["dine_in", "delivery", "takeaway"]
    channel: Literal["waiter_app", "e_menu", "kiosk"] = "e_menu"
    timestamp: datetime
    cart: list[CartItem] = Field(default_factory=list)
    limit: int = Field(default=3, ge=1, le=3)

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware ISO 8601")
        return value


class Recommendation(BaseModel):
    product_id: str
    name: str
    category_id: str
    type: Literal["upsell", "cross_sell"]
    replaces_product_id: str | None = None
    price: int
    score: float = Field(ge=0, le=1)
    reason: str


class RecommendationResponse(BaseModel):
    recommendations: list[Recommendation]
    model_version: str


class ForecastPoint(BaseModel):
    date: date
    quantity: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    business_id: str
    level: Literal["product", "category"]
    id: str
    name: str
    order_type: str | None
    unit: Literal["pieces"] = "pieces"
    interval: float = 0.8
    forecast: list[ForecastPoint]
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_versions: dict[str, str]


class EvaluationPeriod(BaseModel):
    from_: str = Field(alias="from")
    to: str

    model_config = {"populate_by_name": True}


class RecommendationMetrics(BaseModel):
    hit_rate_at_3: float
    baseline_hit_rate_at_3: float
    n_baskets: int
    test_period: EvaluationPeriod | None
    methodology: str = Field(
        description="Global chronological 80/20 split, or business-local chronological 80/20 split."
    )


class ForecastMetrics(BaseModel):
    wape: float | None
    baseline_wape: float | None
    level: Literal["product"]
    n_series: int
    test_period: EvaluationPeriod | None
    methodology: str = Field(description="Rolling-origin chronological evaluation.")


class BusinessImpact(BaseModel):
    estimated_aov_uplift_pct: float | None
    estimated_waste_reduction_pct: float | None
    assumptions: str


class MetricsResponse(BaseModel):
    recommendations: RecommendationMetrics
    forecast: ForecastMetrics
    business_impact: BusinessImpact
    model_versions: dict[str, str]


PRODUCTS, ORDERS, SALES_STATS = load_datasets(DATA_DIR)
PRODUCTS_DF = pd.json_normalize(PRODUCTS) if PRODUCTS else pd.DataFrame()
ORDERS_DF = pd.json_normalize(ORDERS) if ORDERS else pd.DataFrame()
SALES_STATS_DF = pd.json_normalize(SALES_STATS) if SALES_STATS else pd.DataFrame()
PRODUCTS_BY_BUSINESS = defaultdict(dict)
for _product in PRODUCTS:
    PRODUCTS_BY_BUSINESS[_product.get("business_id")][_product.get("id")] = _product
ORDERS_BY_BUSINESS = defaultdict(list)
for _order in ORDERS:
    ORDERS_BY_BUSINESS[_order.get("business_id")].append(_order)
for _orders in ORDERS_BY_BUSINESS.values():
    _orders.sort(key=lambda order: utc_timestamp(order.get("created_at")))


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    horizon_error = any(
        error.get("loc") == ("query", "horizon")
        for error in exc.errors()
    )
    status_code = 422 if horizon_error else 400
    code = "unprocessable" if status_code == 422 else "invalid_request"
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": str(exc.errors())}})


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
    code = (
        "not_found" if exc.status_code == 404
        else "unprocessable" if exc.status_code == 422
        else "internal_error" if exc.status_code >= 500
        else "invalid_request"
    )
    return JSONResponse(status_code=exc.status_code, content={"error": {"code": code, "message": str(exc.detail)}})


@app.exception_handler(Exception)
async def internal_error(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API error", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error", "message": "Internal server error"}},
    )


def business_products(business_id: str) -> list[dict[str, Any]]:
    return list(PRODUCTS_BY_BUSINESS.get(business_id, {}).values())


def product_price(product: dict[str, Any], order_type: str) -> int:
    return int(product[{"dine_in": "price_store", "delivery": "price_delivery", "takeaway": "price_takeaway"}[order_type]])


@lru_cache(maxsize=32)
def cached_item_rows(business_id: str, order_type: str | None = None) -> pd.DataFrame:
    return item_rows(ORDERS_BY_BUSINESS, business_id, order_type)


def recommendation_candidates(
    request: RecommendationRequest,
    excluded_order_id: str | None = None,
    before: datetime | None = None,
) -> list[Recommendation]:
    if not business_products(request.business_id):
        raise HTTPException(status_code=404, detail=f"Unknown business_id: {request.business_id}")
    known_product_ids = {product["id"] for product in business_products(request.business_id)}
    unknown_cart_ids = {item.product_id for item in request.cart} - known_product_ids
    if unknown_cart_ids:
        raise HTTPException(status_code=404, detail=f"Unknown product_id: {sorted(unknown_cart_ids)[0]}")
    products = [p for p in business_products(request.business_id)
                if p.get("available") and p.get("published") and request.order_type in (p.get("order_types") or [])]
    if not products:
        return []
    rows = cached_item_rows(request.business_id, request.order_type)
    if excluded_order_id is not None and not rows.empty:
        rows = rows[rows["order_id"] != excluded_order_id]
    if before is not None and not rows.empty:
        cutoff = local_timestamp(before)
        rows = rows[rows["timestamp"] < cutoff]
    cart_ids = {item.product_id for item in request.cart}
    popularity = Counter(rows["product_id"]) if not rows.empty else Counter()
    max_popularity = max(popularity.values(), default=1)
    candidates = []
    for product in products:
        product_id = product["id"]
        if product_id in cart_ids:
            continue
        product_rows = rows[rows["product_id"] == product_id] if not rows.empty else rows
        time_score = 0.0
        if not product_rows.empty:
            hours = product_rows["timestamp"].dt.hour
            request_hour = int(local_timestamp(request.timestamp).hour)
            time_score = min(float(((hours - request_hour).abs() <= 2).mean()), 1.0)
        co_occurrence = 0.0
        if cart_ids and not rows.empty:
            basket_ids = rows.loc[rows["product_id"].isin(cart_ids), "order_id"].unique()
            co_occurrence = min(float(rows[(rows["order_id"].isin(basket_ids)) & (rows["product_id"] == product_id)].shape[0]) /
                                max(len(basket_ids), 1), 1.0)
        score = min(1.0, 0.45 * popularity[product_id] / max_popularity + 0.30 * co_occurrence + 0.25 * time_score)
        category_id = product.get("category", {}).get("id", "")
        replacement = next(
            (cart_product for cart_product in products
             if cart_product.get("id") in cart_ids
             and cart_product.get("category", {}).get("id") == category_id
             and product_price(product, request.order_type) > product_price(cart_product, request.order_type)),
            None,
        )
        is_cross_sell = replacement is None
        if replacement is not None:
            score = min(1.0, score + 0.4)
        candidates.append((score or 0.15, product, is_cross_sell, replacement))
    candidates.sort(key=lambda value: (-value[0], value[1]["id"]))
    result = []
    for score, product, is_cross_sell, replacement in candidates[:request.limit]:
        kind = "cross_sell" if is_cross_sell else "upsell"
        replaced = replacement.get("id") if replacement else None
        product_name = replacement.get("name_gr", "την επιλογή σας") if replacement else "τις επιλογές σας"
        reason = (
            f"Ταιριάζει υπέροχα με {product_name} και παραγγέλνεται συχνά μαζί του."
            if is_cross_sell else f"Δημοφιλής αναβάθμιση του {product_name}."
        )
        result.append(Recommendation(product_id=product["id"], name=product.get("name_gr") or product.get("name_en", ""),
                                     category_id=product.get("category", {}).get("id", ""), type=kind,
                                     replaces_product_id=replaced, price=product_price(product, request.order_type),
                                     score=round(score, 4), reason=reason))
    return result


@app.post("/recommendations", response_model=RecommendationResponse, response_model_exclude_none=True)
def recommendations(request: RecommendationRequest) -> RecommendationResponse:
    return RecommendationResponse(recommendations=recommendation_candidates(request), model_version=REC_MODEL_VERSION)


def forecast_series(business_id: str, level: str, identifier: str, order_type: str | None) -> pd.Series:
    rows = cached_item_rows(business_id, order_type)
    if rows.empty:
        return pd.Series(dtype=float)
    key = "product_id" if level == "product" else "category_id"
    rows = rows[rows[key] == identifier].dropna(subset=["timestamp"])
    if rows.empty:
        return pd.Series(dtype=float)
    series = rows.groupby(rows["timestamp"].dt.floor("D"))["quantity"].sum().sort_index()
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


@app.get("/forecast", response_model=ForecastResponse)
def forecast(
    business_id: str,
    level: Literal["product", "category"],
    id: str,
    start_date: date | None = None,
    horizon: int = Query(default=7, ge=1, le=14),
    order_type: Literal["dine_in", "delivery", "takeaway"] | None = None,
) -> ForecastResponse:
    products = business_products(business_id)
    if not products:
        raise HTTPException(status_code=404, detail=f"Unknown business_id: {business_id}")
    product = next((p for p in products if p["id"] == id), None)
    category_product = next((p for p in products if p.get("category", {}).get("id") == id), None)
    if (level == "product" and product is None) or (level == "category" and category_product is None):
        raise HTTPException(status_code=404, detail=f"Unknown {level} id: {id}")
    name = product.get("name_gr", "") if product else category_product["category"].get("name_gr", "")
    series = forecast_series(business_id, level, id, order_type)
    forecast_start = start_date or (athens_today() + timedelta(days=1))
    points = []
    for offset in range(horizon):
        target = forecast_start + timedelta(days=offset)
        quantity, lower, upper = seasonal_forecast(series, target)
        points.append(ForecastPoint(date=target, quantity=round(quantity, 2),
                                    lower=round(min(lower, quantity), 2), upper=round(max(upper, quantity), 2)))
    return ForecastResponse(business_id=business_id, level=level, id=id, name=name,
                            order_type=order_type, forecast=points, model_version=FORECAST_MODEL_VERSION)


def _calculate_metrics(business_id: str | None = None) -> dict[str, Any]:
    if business_id is not None and not business_products(business_id):
        raise HTTPException(status_code=404, detail=f"Unknown business_id: {business_id}")
    orders = [o for o in ORDERS if business_id is None or o.get("business_id") == business_id]
    orders.sort(key=lambda order: order.get("created_at", ""))
    series_count = len({item.get("product_id") for order in orders for item in order.get("items", [])})
    test_orders = orders[max(1, int(len(orders) * 0.8)):]
    recommendation_tests = 0
    recommendation_hits = 0
    baseline_hits = 0
    recommendation_dates: list[date] = []
    for order in test_orders:
        distinct_items = list(dict.fromkeys(item.get("product_id") for item in order.get("items", [])))
        if len(distinct_items) < 2:
            continue
        hidden_product_id = distinct_items[-1]
        visible_items = [
            {"product_id": product_id, "quantity": 1}
            for product_id in distinct_items[:-1]
        ]
        request = RecommendationRequest(
            business_id=order["business_id"],
            order_type=order["type"],
            timestamp=local_timestamp(order["created_at"]).to_pydatetime().replace(tzinfo=BUSINESS_TIMEZONE),
            cart=visible_items,
        )
        recommendations_for_test = recommendation_candidates(
            request, excluded_order_id=order["id"], before=pd.to_datetime(order["created_at"]).to_pydatetime()
        )
        recommended_ids = {item.product_id for item in recommendations_for_test}
        product_rows = cached_item_rows(order["business_id"], order["type"])
        cutoff = local_timestamp(order["created_at"])
        product_rows = product_rows[(product_rows["timestamp"] < cutoff) & (product_rows["order_id"] != order["id"])]
        popular_ids = (
            product_rows.groupby("product_id")["quantity"].sum()
            .sort_values(ascending=False)
            .head(3)
            .index
            .tolist()
        )
        recommendation_tests += 1
        evaluated_date = local_date(order["created_at"])
        if evaluated_date is not None:
            recommendation_dates.append(evaluated_date)
        recommendation_hits += hidden_product_id in recommended_ids
        baseline_hits += hidden_product_id in popular_ids

    hit_rate = recommendation_hits / recommendation_tests if recommendation_tests else 0.0
    baseline_hit_rate = baseline_hits / recommendation_tests if recommendation_tests else 0.0
    daily_totals: dict[tuple[str, date], float] = {}
    for order in orders:
        order_date = local_date(order.get("created_at"))
        if order_date is None:
            continue
        for item in order.get("items", []):
            key = (item["product_id"], order_date)
            daily_totals[key] = daily_totals.get(key, 0.0) + float(item.get("quantity", 0))
    forecast_metrics = rolling_origin_evaluation(daily_totals)
    recommendation_period = (
        {"from": min(recommendation_dates), "to": max(recommendation_dates)}
        if recommendation_dates else None
    )
    methodology = (
        "business-local chronological 80/20 split"
        if business_id is not None else "global chronological 80/20 split"
    )
    return {
        "recommendations": {"hit_rate_at_3": round(hit_rate, 4), "baseline_hit_rate_at_3": round(baseline_hit_rate, 4),
                            "n_baskets": recommendation_tests,
                            "test_period": ({key: value.isoformat() for key, value in recommendation_period.items()}
                                            if recommendation_period else None),
                            "methodology": methodology},
        "forecast": {"wape": round(forecast_metrics["wape"], 4) if forecast_metrics["wape"] is not None else None,
                     "baseline_wape": round(forecast_metrics["baseline_wape"], 4)
                     if forecast_metrics["baseline_wape"] is not None else None,
                     "level": "product", "n_series": series_count,
                     "test_period": forecast_metrics["test_period"],
                     "methodology": "rolling-origin chronological evaluation"},
        "business_impact": {"estimated_aov_uplift_pct": None, "estimated_waste_reduction_pct": None,
                            "assumptions": "Δεν εκτιμάται χωρίς online experiment ή validated business assumptions."},
        "model_versions": {"recommendations": REC_MODEL_VERSION, "forecast": FORECAST_MODEL_VERSION},
    }


@lru_cache(maxsize=8)
def _cached_metrics(business_id: str | None = None) -> dict[str, Any]:
    """Cache metrics for the immutable datasets loaded during application startup."""
    return _calculate_metrics(business_id)


def clear_metrics_cache() -> None:
    """Invalidate metrics after a dataset reload or any in-process data replacement."""
    _cached_metrics.cache_clear()


@app.get("/metrics", response_model=MetricsResponse)
def metrics(business_id: str | None = None) -> dict[str, Any]:
    if business_id is not None and not business_products(business_id):
        raise HTTPException(status_code=404, detail=f"Unknown business_id: {business_id}")
    return deepcopy(_cached_metrics(business_id))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_versions={"recommendations": REC_MODEL_VERSION, "forecast": FORECAST_MODEL_VERSION},
    )
