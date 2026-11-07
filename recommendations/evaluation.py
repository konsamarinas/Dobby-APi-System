"""Canonical, leakage-safe evaluation for all recommendation models."""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from time import perf_counter
from typing import Any, Callable, Iterable

import pandas as pd

from ..core.preprocessing import local_date, local_timestamp, utc_timestamp
from .inference import association_recommendations, similarity_recommendations
from .hybrid import HybridWeights, hybrid_recommendations, train_hybrid
from .model import RecommendationArtifact
from .training import (
    train_association_rules,
    train_item_similarity,
    transaction_baskets,
)


@dataclass(frozen=True)
class PreparedOrder:
    timestamp: datetime
    order_id: str
    business_id: str
    order_type: str
    products: tuple[str, ...]
    quantities: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class EvaluationBasket:
    order: PreparedOrder
    hidden_product_id: str
    visible_product_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationPlan:
    orders: tuple[PreparedOrder, ...]
    train_orders: tuple[PreparedOrder, ...]
    validation_orders: tuple[PreparedOrder, ...]
    test_orders: tuple[PreparedOrder, ...]
    baskets: tuple[EvaluationBasket, ...]


def evaluation_period(
    test_orders: list[dict[str, Any]],
    local_date: Callable[[Any], date | None] = globals()["local_date"],
) -> dict[str, str] | None:
    dates = [local_date(order.get("created_at")) for order in test_orders]
    dates = [value for value in dates if isinstance(value, date)]
    return {"from": min(dates).isoformat(), "to": max(dates).isoformat()} if dates else None


def _prepare_orders(orders: Iterable[dict[str, Any]]) -> tuple[PreparedOrder, ...]:
    prepared = []
    for order in orders:
        timestamp = utc_timestamp(order.get("created_at"))
        if pd.isna(timestamp):
            continue
        product_quantities: Counter[str] = Counter()
        for item in order.get("items", []):
            product_id = item.get("product_id")
            if product_id:
                product_quantities[product_id] += int(item.get("quantity", 0))
        if not product_quantities:
            continue
        prepared.append(PreparedOrder(
            timestamp=timestamp.to_pydatetime(),
            order_id=str(order.get("id")),
            business_id=str(order.get("business_id")),
            order_type=str(order.get("type")),
            products=tuple(product_quantities),
            quantities=tuple(
                (product, int(quantity))
                for product, quantity in product_quantities.items()
            ),
        ))
    return tuple(sorted(prepared, key=lambda row: row.timestamp))


def build_evaluation_plan(orders: Iterable[dict[str, Any]]) -> EvaluationPlan:
    """Build the single global chronological 70/10/20 train/validation/test plan."""
    prepared = _prepare_orders(orders)
    split = max(1, int(len(prepared) * 0.8))
    validation_split = max(1, int(len(prepared) * 0.7))
    test_orders = prepared[split:]
    baskets = tuple(
        EvaluationBasket(order, order.products[-1], order.products[:-1])
        for order in test_orders
        if len(order.products) >= 2
    )
    return EvaluationPlan(
        orders=prepared,
        train_orders=prepared[:validation_split],
        validation_orders=prepared[validation_split:split],
        test_orders=test_orders,
        baskets=baskets,
    )


def _baskets_for(orders: Iterable[PreparedOrder]) -> tuple[EvaluationBasket, ...]:
    return tuple(
        EvaluationBasket(order, order.products[-1], order.products[:-1])
        for order in orders
        if len(order.products) >= 2
    )


def eligible_test_orders(orders: list[dict[str, Any]], business_id: str) -> list[dict[str, Any]]:
    """Compatibility helper returning the business subset of the canonical test split."""
    plan = build_evaluation_plan(orders)
    ids = {basket.order.order_id for basket in plan.baskets if basket.order.business_id == business_id}
    return [order for order in orders if str(order.get("id")) in ids]


def _historical_orders(
    prepared: tuple[PreparedOrder, ...],
    business_id: str,
    order_type: str,
    cutoff: datetime,
) -> list[dict[str, Any]]:
    return [
        {
            "id": order.order_id,
            "business_id": order.business_id,
            "type": order.order_type,
            "created_at": order.timestamp.isoformat(),
            "items": [
                {"product_id": product, "quantity": quantity}
                for product, quantity in order.quantities
            ],
        }
        for order in prepared
        if order.business_id == business_id
        and order.order_type == order_type
        and order.timestamp < cutoff
    ]


def _allowed_products(
    products: list[dict[str, Any]],
    business_id: str,
    order_type: str,
) -> set[str]:
    return {
        product["id"] for product in products
        if product.get("business_id") == business_id
        and product.get("available")
        and product.get("published")
        and order_type in (product.get("order_types") or [])
    }


def _production_popularity(
    history: list[PreparedOrder],
    visible: tuple[str, ...],
    allowed: set[str],
) -> list[str]:
    rows = [
        {"product_id": product, "quantity": quantity}
        for order in history
        for product, quantity in order.quantities
    ]
    if not rows:
        return []
    totals = (
        pd.DataFrame(rows)
        .groupby("product_id")["quantity"]
        .sum()
        .sort_values(ascending=False)
    )
    # Production does not remove cart products from its baseline.
    return [product for product in totals.head(3).index if product in allowed]


def _production_candidates(
    history: list[PreparedOrder],
    products: list[dict[str, Any]],
    basket: EvaluationBasket,
    allowed: set[str],
) -> list[str]:
    """Equivalent scoring path to main.recommendation_candidates, without API changes."""
    cart = set(basket.visible_product_ids)
    popularity = Counter(
        product
        for order in history
        for product, _quantity in order.quantities
    )
    max_popularity = max(popularity.values(), default=1)
    local_request_hour = local_timestamp(basket.order.timestamp).hour
    rows_by_product: dict[str, list[PreparedOrder]] = {}
    for order in history:
        for product, _quantity in order.quantities:
            rows_by_product.setdefault(product, []).append(order)
    basket_ids = {
        order.order_id for product in cart
        for order in rows_by_product.get(product, [])
    }
    product_by_id = {product["id"]: product for product in products if product["id"] in allowed}
    scored = []
    for product_id, product in product_by_id.items():
        if product_id in cart:
            continue
        product_orders = rows_by_product.get(product_id, [])
        time_score = (
            sum(abs(local_timestamp(order.timestamp).hour - local_request_hour) <= 2
                for order in product_orders) / len(product_orders)
            if product_orders else 0.0
        )
        occurrences = sum(
            1 for order in history if order.order_id in basket_ids
            for product, _quantity in order.quantities if product == product_id
        )
        co_occurrence = min(occurrences / max(len(basket_ids), 1), 1.0) if cart else 0.0
        score = min(
            1.0,
            0.45 * popularity[product_id] / max_popularity
            + 0.30 * co_occurrence
            + 0.25 * time_score,
        )
        category_id = product.get("category", {}).get("id", "")
        replacement = next(
            (
                cart_product for cart_product in product_by_id.values()
                if cart_product["id"] in cart
                and cart_product.get("category", {}).get("id", "") == category_id
                and int(product[{"dine_in": "price_store", "delivery": "price_delivery",
                                "takeaway": "price_takeaway"}[basket.order.order_type]])
                > int(cart_product[{"dine_in": "price_store", "delivery": "price_delivery",
                                    "takeaway": "price_takeaway"}[basket.order.order_type]])
            ),
            None,
        )
        if replacement is not None:
            score = min(1.0, score + 0.4)
        scored.append((score or 0.15, product_id))
    scored.sort(key=lambda value: (-value[0], value[1]))
    return [product_id for _score, product_id in scored[:3]]


def _result(
    model: str,
    hits_1: int,
    hits_3: int,
    total: int,
    recommended: set[str],
    catalog: set[str],
) -> dict[str, Any]:
    return {
        "model": model,
        "hit_rate_at_1": hits_1 / total if total else 0.0,
        "hit_rate_at_3": hits_3 / total if total else 0.0,
        "precision_at_3": hits_3 / (total * 3) if total else 0.0,
        "catalog_coverage": len(recommended) / len(catalog) if catalog else 0.0,
        "evaluated_baskets": total,
    }


def evaluate_canonical(
    orders: list[dict[str, Any]],
    products: list[dict[str, Any]],
    *,
    min_support: float = 0.01,
    min_confidence: float = 0.1,
    min_lift: float = 1.0,
    min_co_occurrence: int = 3,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Evaluate every model on one global, immutable test-basket population."""
    started = perf_counter()
    plan = build_evaluation_plan(orders)
    train_cutoff = plan.test_orders[0].timestamp if plan.test_orders else None
    artifacts: dict[tuple[str, str], RecommendationArtifact] = {}
    hybrid_artifacts: dict[tuple[str, str], Any] = {}
    histories: dict[tuple[str, str], list[PreparedOrder]] = {}
    catalogs = {
        (business_id, order_type): _allowed_products(products, business_id, order_type)
        for business_id in {order.business_id for order in plan.orders}
        for order_type in {order.order_type for order in plan.orders if order.business_id == business_id}
    }
    training_orders = tuple(order for order in plan.orders if train_cutoff and order.timestamp < train_cutoff)

    def train_artifacts(
        source_orders: tuple[PreparedOrder, ...],
        cutoff: datetime,
        weights: HybridWeights | None = None,
    ) -> tuple[dict[tuple[str, str], RecommendationArtifact], dict[tuple[str, str], Any]]:
        trained: dict[tuple[str, str], RecommendationArtifact] = {}
        hybrids: dict[tuple[str, str], Any] = {}
        for key in catalogs:
            business_id, order_type = key
            raw = _historical_orders(source_orders, business_id, order_type, cutoff)
            trained[key] = train_association_rules(
                raw, business_id, min_support=min_support,
                min_confidence=min_confidence, min_lift=min_lift,
                min_co_occurrence=min_co_occurrence,
            )
            similarity = train_item_similarity(
                raw, business_id, min_co_occurrence=min_co_occurrence,
            )
            trained[key].similarities = similarity.similarities
            hybrids[key] = train_hybrid(
                [order for order in source_orders
                 if order.business_id == business_id and order.order_type == order_type],
                trained[key],
                cutoff=cutoff,
                weights=weights or HybridWeights(0.65, 0.2, 0.05, 0.1),
            )
        return trained, hybrids

    validation_cutoff = (
        plan.validation_orders[0].timestamp
        if plan.validation_orders else train_cutoff
    )
    validation_training = tuple(
        order for order in plan.orders
        if validation_cutoff and order.timestamp < validation_cutoff
    )
    weight_candidates = (
        HybridWeights(0.65, 0.20, 0.05, 0.10),
        HybridWeights(0.55, 0.30, 0.05, 0.10),
        HybridWeights(0.50, 0.35, 0.05, 0.10),
        HybridWeights(0.70, 0.15, 0.05, 0.10),
    )
    validation_baskets = _baskets_for(plan.validation_orders)
    validation_scores = []
    for candidate_weights in weight_candidates:
        _, validation_hybrids = train_artifacts(
            validation_training, validation_cutoff, candidate_weights
        )
        hits = 0
        for basket in validation_baskets:
            key = (basket.order.business_id, basket.order.order_type)
            allowed = catalogs[key]
            candidates = hybrid_recommendations(
                validation_hybrids[key],
                basket.visible_product_ids,
                timestamp=basket.order.timestamp,
                allowed_product_ids=allowed,
            )
            hits += basket.hidden_product_id in [product for product, _, _ in candidates]
        validation_scores.append((hits, candidate_weights))
    selected_weights = max(
        validation_scores,
        key=lambda value: (value[0], -value[1].similarity, -value[1].context),
    )[1] if validation_scores else weight_candidates[0]

    artifacts, hybrid_artifacts = train_artifacts(
        training_orders, train_cutoff, selected_weights
    ) if train_cutoff else ({}, {})
    for key, allowed in catalogs.items():
        business_id, order_type = key
        histories[key] = [
            order for order in training_orders
            if order.business_id == business_id and order.order_type == order_type
        ]

    names = ("production", "popularity", "association", "jaccard", "cosine", "hybrid")
    hits_1 = Counter()
    hits_3 = Counter()
    recommended = {name: set() for name in names}
    per_business: dict[str, list[EvaluationBasket]] = {}
    business_hits_1: dict[str, Counter[str]] = {}
    business_hits_3: dict[str, Counter[str]] = {}
    business_recommended: dict[str, dict[str, set[str]]] = {}
    type_baskets: dict[str, list[EvaluationBasket]] = {}
    type_hits_1: dict[str, Counter[str]] = {}
    type_hits_3: dict[str, Counter[str]] = {}
    type_recommended: dict[str, dict[str, set[str]]] = {}
    hybrid_latencies: list[float] = []
    for basket in plan.baskets:
        business_id = basket.order.business_id
        per_business.setdefault(business_id, []).append(basket)
        business_hits_1.setdefault(business_id, Counter())
        business_hits_3.setdefault(business_id, Counter())
        business_recommended.setdefault(
            business_id, {name: set() for name in names}
        )
        order_type = basket.order.order_type
        type_baskets.setdefault(order_type, []).append(basket)
        type_hits_1.setdefault(order_type, Counter())
        type_hits_3.setdefault(order_type, Counter())
        type_recommended.setdefault(order_type, {name: set() for name in names})
        key = (basket.order.business_id, basket.order.order_type)
        history = [
            order for order in plan.orders
            if order.business_id == basket.order.business_id
            and order.order_type == basket.order.order_type
            and order.timestamp < basket.order.timestamp
        ]
        allowed = catalogs[key]
        hybrid_started = perf_counter()
        hybrid_candidates = hybrid_recommendations(
            hybrid_artifacts[key],
            basket.visible_product_ids,
            timestamp=basket.order.timestamp,
            allowed_product_ids=allowed,
        )
        hybrid_latencies.append((perf_counter() - hybrid_started) * 1000)
        candidate_sets = {
            "production": _production_candidates(history, products, basket, allowed),
            "popularity": _production_popularity(history, basket.visible_product_ids, allowed),
            "association": [
                product for product, _score in association_recommendations(
                    artifacts[key], basket.visible_product_ids
                ) if product in allowed
            ],
            "jaccard": [
                product for product, _score in similarity_recommendations(
                    artifacts[key], basket.visible_product_ids, metric="jaccard"
                ) if product in allowed
            ],
            "cosine": [
                product for product, _score in similarity_recommendations(
                    artifacts[key], basket.visible_product_ids, metric="cosine"
                ) if product in allowed
            ],
            "hybrid": [product for product, _score, _reason in hybrid_candidates],
        }
        for name, candidates in candidate_sets.items():
            hits_1[name] += basket.hidden_product_id in candidates[:1]
            hits_3[name] += basket.hidden_product_id in candidates[:3]
            recommended[name].update(candidates[:3])
            business_hits_1[business_id][name] += basket.hidden_product_id in candidates[:1]
            business_hits_3[business_id][name] += basket.hidden_product_id in candidates[:3]
            business_recommended[business_id][name].update(candidates[:3])
            type_hits_1[order_type][name] += basket.hidden_product_id in candidates[:1]
            type_hits_3[order_type][name] += basket.hidden_product_id in candidates[:3]
            type_recommended[order_type][name].update(candidates[:3])
        if progress:
            progress(sum(len(value) for value in per_business.values()), len(plan.baskets))

    catalog = {product["id"] for product in products if product.get("business_id") in per_business}
    results = [
        _result(name, hits_1[name], hits_3[name], len(plan.baskets), recommended[name], catalog)
        for name in names
    ]
    for business_id, baskets in per_business.items():
        business_catalog = {product["id"] for product in products if product.get("business_id") == business_id}
        business_results = []
        for name in names:
            business_results.append(_result(
                name,
                business_hits_1[business_id][name],
                business_hits_3[business_id][name],
                len(baskets),
                business_recommended[business_id][name],
                business_catalog,
            ))
        per_business[business_id] = business_results
    per_order_type = {}
    for order_type, baskets in type_baskets.items():
        per_order_type[order_type] = [
            _result(
                name,
                type_hits_1[order_type][name],
                type_hits_3[order_type][name],
                len(baskets),
                type_recommended[order_type][name],
                {product["id"] for product in products
                 if product.get("business_id") in per_business
                 and order_type in (product.get("order_types") or [])},
            )
            for name in names
        ]
    return {
        "methodology": {
            "split": "global chronological 70/10/20",
            "test_fraction": 0.2,
            "train_cutoff": train_cutoff.isoformat() if train_cutoff else None,
            "eligible_baskets": len(plan.baskets),
            "validation_baskets": len(validation_baskets),
            "selected_hybrid_weights": selected_weights.__dict__,
            "validation_weight_scores": [
                {"weights": weights.__dict__, "hit_rate_at_3": hits / len(validation_baskets)
                 if validation_baskets else 0.0}
                for hits, weights in validation_scores
            ],
            "hidden_item": "last distinct product",
        },
        "results": results,
        "per_business": per_business,
        "per_order_type": per_order_type,
        "hybrid_inference_latency_ms": {
            "mean": sum(hybrid_latencies) / len(hybrid_latencies) if hybrid_latencies else 0.0,
            "p95": sorted(hybrid_latencies)[int(len(hybrid_latencies) * 0.95) - 1]
            if hybrid_latencies else 0.0,
        },
        "runtime_seconds": perf_counter() - started,
    }


def evaluate_all(
    orders: list[dict[str, Any]],
    products: list[dict[str, Any]],
    business_id: str,
    **training_kwargs: Any,
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper returning canonical results for one business."""
    report = evaluate_canonical(orders, products, **training_kwargs)
    return report["per_business"].get(business_id, [])
