"""Deterministic offline training for association and item-similarity models."""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Iterable

import pandas as pd

from ..core.preprocessing import utc_timestamp
from .model import AssociationRule, RecommendationArtifact, Similarity, cutoff_string


def transaction_baskets(
    orders: Iterable[dict[str, Any]],
    business_id: str,
    cutoff: datetime | None = None,
    order_type: str | None = None,
) -> list[tuple[datetime, frozenset[str], str]]:
    """Return sorted, de-duplicated binary baskets before an optional cutoff."""
    if cutoff is not None and (cutoff.tzinfo is None or cutoff.utcoffset() is None):
        raise ValueError("cutoff must be timezone-aware")
    cutoff_utc = cutoff.astimezone(timezone.utc) if cutoff else None
    transactions: list[tuple[datetime, frozenset[str], str]] = []
    for order in orders:
        if order.get("business_id") != business_id or (order_type and order.get("type") != order_type):
            continue
        timestamp = utc_timestamp(order.get("created_at"))
        if pd.isna(timestamp):
            continue
        timestamp_datetime = timestamp.to_pydatetime()
        if cutoff_utc is not None and timestamp_datetime >= cutoff_utc:
            continue
        products = frozenset(item["product_id"] for item in order.get("items", []) if item.get("product_id"))
        if products:
            transactions.append((timestamp_datetime, products, str(order.get("id"))))
    transactions.sort(key=lambda row: (row[0], row[2]))
    return transactions


def _support_counts(transactions: list[tuple[datetime, frozenset[str], str]]) -> Counter[str]:
    return Counter(product for _, basket, _ in transactions for product in basket)


def train_association_rules(
    orders: Iterable[dict[str, Any]],
    business_id: str,
    *,
    cutoff: datetime | None = None,
    order_type: str | None = None,
    min_support: float = 0.01,
    min_confidence: float = 0.1,
    min_lift: float = 1.0,
    min_co_occurrence: int = 3,
    model_version: str = "rec-association-v2",
) -> RecommendationArtifact:
    transactions = transaction_baskets(orders, business_id, cutoff, order_type)
    total = len(transactions)
    supports = _support_counts(transactions)
    pair_counts = Counter(
        pair for _, basket, _ in transactions for pair in combinations(sorted(basket), 2)
    )
    rules = []
    for (product_a, product_b), count in sorted(pair_counts.items()):
        pair_support = count / total if total else 0.0
        if count < min_co_occurrence or pair_support < min_support:
            continue
        for antecedent, consequent in ((product_a, product_b), (product_b, product_a)):
            confidence = count / supports[antecedent]
            consequent_support = supports[consequent] / total
            lift = confidence / consequent_support if consequent_support else 0.0
            if confidence >= min_confidence and lift >= min_lift:
                rules.append(AssociationRule(
                    business_id=business_id,
                    antecedent_product_id=antecedent,
                    consequent_product_id=consequent,
                    support=pair_support,
                    confidence=confidence,
                    lift=lift,
                    co_occurrence_count=count,
                ))
    rules.sort(key=lambda rule: (
        rule.antecedent_product_id, rule.consequent_product_id,
        -rule.confidence, -rule.lift,
    ))
    return RecommendationArtifact(
        model_version=model_version,
        business_id=business_id,
        training_cutoff=cutoff_string(cutoff),
        transaction_count=total,
        product_support=dict(sorted(supports.items())),
        association_rules=rules,
    )


def train_item_similarity(
    orders: Iterable[dict[str, Any]],
    business_id: str,
    *,
    cutoff: datetime | None = None,
    order_type: str | None = None,
    min_co_occurrence: int = 3,
    model_version: str = "rec-item-v2",
) -> RecommendationArtifact:
    transactions = transaction_baskets(orders, business_id, cutoff, order_type)
    supports = _support_counts(transactions)
    pair_counts = Counter(
        pair for _, basket, _ in transactions for pair in combinations(sorted(basket), 2)
    )
    similarities = []
    for (product_a, product_b), intersection in sorted(pair_counts.items()):
        if intersection < min_co_occurrence:
            continue
        union = supports[product_a] + supports[product_b] - intersection
        jaccard = intersection / union if union else 0.0
        cosine = intersection / math.sqrt(supports[product_a] * supports[product_b])
        similarities.append(Similarity(product_a, product_b, jaccard, cosine, intersection))
    return RecommendationArtifact(
        model_version=model_version,
        business_id=business_id,
        training_cutoff=cutoff_string(cutoff),
        transaction_count=len(transactions),
        product_support=dict(sorted(supports.items())),
        similarities=similarities,
    )


def train_recommendation_artifact(
    orders: Iterable[dict[str, Any]],
    business_id: str,
    *,
    cutoff: datetime | None = None,
    order_type: str | None = None,
    min_support: float = 0.01,
    min_confidence: float = 0.1,
    min_lift: float = 1.0,
    min_co_occurrence: int = 3,
) -> RecommendationArtifact:
    """Train both independent model families into one reusable artifact."""
    association = train_association_rules(
        orders, business_id, cutoff=cutoff, order_type=order_type,
        min_support=min_support, min_confidence=min_confidence,
        min_lift=min_lift, min_co_occurrence=min_co_occurrence,
    )
    similarity = train_item_similarity(
        orders, business_id, cutoff=cutoff, order_type=order_type,
        min_co_occurrence=min_co_occurrence,
    )
    association.similarities = similarity.similarities
    association.model_version = "rec-independent-v2"
    return association
