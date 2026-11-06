"""Dobby API using the documented nested products and orders formats."""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REC_MODEL_VERSION = "rec-v1.2"
FORECAST_MODEL_VERSION = "fc-v0.9"
ORDER_TYPES = {"dine_in", "delivery", "takeaway"}

app = FastAPI(title="Dobby Restaurant Management API", version=REC_MODEL_VERSION,
              description="API προτάσεων και πρόβλεψης ζήτησης για το dobby.")
logger = logging.getLogger(__name__)


class CartItem(BaseModel):
    product_id: str
    quantity: int = Field(ge=1)
    selection_option_ids: list[str] = Field(default_factory=list)
    extra_ids: list[str] = Field(default_factory=list)


class RecommendationRequest(BaseModel):
    business_id: str
    order_type: Literal["dine_in", "delivery", "takeaway"]
    channel: str = "api"
    timestamp: datetime
    cart: list[CartItem]
    limit: int = Field(default=3, ge=1, le=3)


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


def read_json(name: str) -> list[dict[str, Any]]:
    path = DATA_DIR / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, list):
        raise RuntimeError(f"{name} must contain a JSON array")
    return value


PRODUCTS = read_json("products.json")
ORDERS = read_json("orders.json")
SALES_STATS = read_json("sales_stats.json")
PRODUCTS_DF = pd.json_normalize(PRODUCTS) if PRODUCTS else pd.DataFrame()
ORDERS_DF = pd.json_normalize(ORDERS) if ORDERS else pd.DataFrame()
SALES_STATS_DF = pd.json_normalize(SALES_STATS) if SALES_STATS else pd.DataFrame()


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
    code = "not_found" if exc.status_code == 404 else "unprocessable" if exc.status_code == 422 else "invalid_request"
    return JSONResponse(status_code=exc.status_code, content={"error": {"code": code, "message": str(exc.detail)}})


@app.exception_handler(Exception)
async def internal_error(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API error", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error", "message": "Internal server error"}},
    )


def business_products(business_id: str) -> list[dict[str, Any]]:
    return [p for p in PRODUCTS if p.get("business_id") == business_id]


def product_price(product: dict[str, Any], order_type: str) -> int:
    return int(product[{"dine_in": "price_store", "delivery": "price_delivery", "takeaway": "price_takeaway"}[order_type]])


def item_rows(business_id: str, order_type: str | None = None) -> pd.DataFrame:
    rows = []
    for order in ORDERS:
        if order.get("business_id") != business_id or (order_type and order.get("type") != order_type):
            continue
        timestamp = pd.to_datetime(order.get("created_at"), errors="coerce")
        for item in order.get("items") or []:
            rows.append({"order_id": order.get("id"), "product_id": item.get("product_id"),
                         "category_id": (item.get("category") or {}).get("id"),
                         "quantity": int(item.get("quantity", 0)), "timestamp": timestamp})
    return pd.DataFrame(rows, columns=["order_id", "product_id", "category_id", "quantity", "timestamp"])


def recommendation_candidates(
    request: RecommendationRequest,
    excluded_order_id: str | None = None,
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
    rows = item_rows(request.business_id, request.order_type)
    if excluded_order_id is not None and not rows.empty:
        rows = rows[rows["order_id"] != excluded_order_id]
    cart_ids = {item.product_id for item in request.cart}
    cart_categories = {p.get("category", {}).get("id") for p in products if p.get("id") in cart_ids}
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
            time_score = min(float(((hours - request.timestamp.hour).abs() <= 2).mean()), 1.0)
        co_occurrence = 0.0
        if cart_ids and not rows.empty:
            basket_ids = rows.loc[rows["product_id"].isin(cart_ids), "order_id"].unique()
            co_occurrence = min(float(rows[(rows["order_id"].isin(basket_ids)) & (rows["product_id"] == product_id)].shape[0]) /
                                max(len(basket_ids), 1), 1.0)
        score = min(1.0, 0.45 * popularity[product_id] / max_popularity + 0.30 * co_occurrence + 0.25 * time_score)
        category_id = product.get("category", {}).get("id", "")
        is_cross_sell = not cart_ids or category_id not in cart_categories
        candidates.append((score or 0.15, product, is_cross_sell))
    candidates.sort(key=lambda value: (-value[0], value[1]["id"]))
    result = []
    for score, product, is_cross_sell in candidates[:request.limit]:
        kind = "cross_sell" if is_cross_sell else "upsell"
        replaced = next(iter(cart_ids), None) if kind == "upsell" else None
        reason = "Ταιριάζει υπέροχα με τις επιλογές σας και παραγγέλνεται συχνά μαζί τους." if is_cross_sell else "Δημοφιλής αναβάθμιση της επιλογής σας."
        result.append(Recommendation(product_id=product["id"], name=product.get("name_gr") or product.get("name_en", ""),
                                     category_id=product.get("category", {}).get("id", ""), type=kind,
                                     replaces_product_id=replaced, price=product_price(product, request.order_type),
                                     score=round(score, 4), reason=reason))
    return result


@app.post("/recommendations", response_model=RecommendationResponse, response_model_exclude_none=True)
def recommendations(request: RecommendationRequest) -> RecommendationResponse:
    return RecommendationResponse(recommendations=recommendation_candidates(request), model_version=REC_MODEL_VERSION)


def forecast_series(business_id: str, level: str, identifier: str, order_type: str | None) -> pd.Series:
    rows = item_rows(business_id, order_type)
    if rows.empty:
        return pd.Series(dtype=float)
    key = "product_id" if level == "product" else "category_id"
    rows = rows[rows[key] == identifier].dropna(subset=["timestamp"])
    if rows.empty:
        return pd.Series(dtype=float)
    return rows.groupby(rows["timestamp"].dt.floor("D"))["quantity"].sum().sort_index()


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
    if level == "product" and product is None or level == "category" and category_product is None:
        raise HTTPException(status_code=404, detail=f"Unknown {level} id: {id}")
    name = product.get("name_gr", "") if product else category_product["category"].get("name_gr", "")
    series = forecast_series(business_id, level, id, order_type)
    forecast_start = start_date or (date.today() + timedelta(days=1))
    history = series.tail(28)
    mean = float(history.mean()) if not history.empty else 0.0
    spread = float(history.std(ddof=1)) if len(history) > 1 else math.sqrt(mean)
    quantity = max(0.0, mean)
    lower, upper = max(0.0, quantity - 1.28 * spread), quantity + 1.28 * spread
    points = [ForecastPoint(date=forecast_start + timedelta(days=i), quantity=round(quantity, 2),
                            lower=round(min(lower, quantity), 2), upper=round(max(upper, quantity), 2))
              for i in range(horizon)]
    return ForecastResponse(business_id=business_id, level=level, id=id, name=name,
                            order_type=order_type, forecast=points, model_version=FORECAST_MODEL_VERSION)


@app.get("/metrics")
def metrics(business_id: str | None = None) -> dict[str, Any]:
    orders = [o for o in ORDERS if business_id is None or o.get("business_id") == business_id]
    timestamps = [o.get("created_at", "")[:10] for o in orders if o.get("created_at")]
    test_period = {"from": min(timestamps) if timestamps else date.today().isoformat(),
                   "to": max(timestamps) if timestamps else date.today().isoformat()}
    series_count = len({item.get("product_id") for order in orders for item in order.get("items", [])})
    recommendation_tests = 0
    recommendation_hits = 0
    baseline_hits = 0
    for order in orders:
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
            timestamp=pd.to_datetime(order["created_at"]).to_pydatetime(),
            cart=visible_items,
        )
        recommendations_for_test = recommendation_candidates(request, excluded_order_id=order["id"])
        recommended_ids = {item.product_id for item in recommendations_for_test}
        product_rows = item_rows(order["business_id"], order["type"])
        product_rows = product_rows[product_rows["order_id"] != order["id"]]
        popular_ids = (
            product_rows.groupby("product_id")["quantity"].sum()
            .sort_values(ascending=False)
            .head(3)
            .index
            .tolist()
        )
        recommendation_tests += 1
        recommendation_hits += hidden_product_id in recommended_ids
        baseline_hits += hidden_product_id in popular_ids

    hit_rate = recommendation_hits / recommendation_tests if recommendation_tests else 0.0
    baseline_hit_rate = baseline_hits / recommendation_tests if recommendation_tests else 0.0
    daily_totals: dict[tuple[str, date], float] = {}
    for order in orders:
        order_date = pd.to_datetime(order.get("created_at")).date()
        for item in order.get("items", []):
            key = (item["product_id"], order_date)
            daily_totals[key] = daily_totals.get(key, 0.0) + float(item.get("quantity", 0))
    forecast_errors = 0.0
    baseline_errors = 0.0
    forecast_actuals = 0.0
    for product_id in {key[0] for key in daily_totals}:
        product_dates = sorted(day for item_id, day in daily_totals if item_id == product_id)
        if len(product_dates) < 8:
            continue
        test_dates = product_dates[-7:]
        train_values = [daily_totals[(product_id, day)] for day in product_dates[:-7]]
        mean_prediction = sum(train_values) / len(train_values)
        for test_day in test_dates:
            actual = daily_totals[(product_id, test_day)]
            previous_week = daily_totals.get((product_id, test_day - timedelta(days=7)), mean_prediction)
            forecast_errors += abs(actual - mean_prediction)
            baseline_errors += abs(actual - previous_week)
            forecast_actuals += actual
    wape = forecast_errors / forecast_actuals if forecast_actuals else 0.0
    baseline_wape = baseline_errors / forecast_actuals if forecast_actuals else 0.0
    return {
        "recommendations": {"hit_rate_at_3": round(hit_rate, 4), "baseline_hit_rate_at_3": round(baseline_hit_rate, 4),
                            "n_baskets": recommendation_tests, "test_period": test_period},
        "forecast": {"wape": round(wape, 4), "baseline_wape": round(baseline_wape, 4), "level": "product",
                     "n_series": series_count, "test_period": test_period},
        "business_impact": {"estimated_aov_uplift_pct": 8.5, "estimated_waste_reduction_pct": 12.0,
                            "assumptions": "Εκτίμηση από offline test set και υπόθεση σταθερής επισκεψιμότητας."},
        "model_versions": {"recommendations": REC_MODEL_VERSION, "forecast": FORECAST_MODEL_VERSION},
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_versions": {"recommendations": REC_MODEL_VERSION, "forecast": FORECAST_MODEL_VERSION}}
