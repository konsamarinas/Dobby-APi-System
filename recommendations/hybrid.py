"""Offline hybrid recommendation model built from precomputed signals."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable

from ..core.preprocessing import local_timestamp
from .model import RecommendationArtifact


@dataclass(frozen=True)
class HybridWeights:
    popularity: float
    similarity: float
    association: float
    context: float

    def normalized(self) -> "HybridWeights":
        total = self.popularity + self.similarity + self.association + self.context
        if total <= 0:
            raise ValueError("hybrid weights must have a positive sum")
        return HybridWeights(
            self.popularity / total,
            self.similarity / total,
            self.association / total,
            self.context / total,
        )


@dataclass
class HybridArtifact:
    model_version: str
    base: RecommendationArtifact
    quantity_support: dict[str, int]
    hour_counts: dict[str, dict[int, int]]
    weekday_counts: dict[str, dict[int, int]]
    recent_quantity_support: dict[str, int]
    weights: HybridWeights
    cutoff: datetime | None


def train_hybrid(
    history: Iterable[Any],
    base: RecommendationArtifact,
    *,
    cutoff: datetime | None = None,
    weights: HybridWeights = HybridWeights(0.65, 0.2, 0.05, 0.1),
    recent_days: int = 28,
    model_version: str = "rec-hybrid-v1",
) -> HybridArtifact:
    quantity_support: Counter[str] = Counter()
    hour_counts: dict[str, Counter[int]] = defaultdict(Counter)
    weekday_counts: dict[str, Counter[int]] = defaultdict(Counter)
    recent_quantity_support: Counter[str] = Counter()
    recent_cutoff = cutoff - timedelta(days=recent_days) if cutoff else None
    for order in history:
        timestamp = order.timestamp
        local = local_timestamp(timestamp)
        for product, quantity in order.quantities:
            quantity_support[product] += quantity
            hour_counts[product][local.hour] += quantity
            weekday_counts[product][local.weekday()] += quantity
            if recent_cutoff is not None and timestamp >= recent_cutoff:
                recent_quantity_support[product] += quantity
    return HybridArtifact(
        model_version=model_version,
        base=base,
        quantity_support=dict(sorted(quantity_support.items())),
        hour_counts={key: dict(sorted(value.items())) for key, value in sorted(hour_counts.items())},
        weekday_counts={key: dict(sorted(value.items())) for key, value in sorted(weekday_counts.items())},
        recent_quantity_support=dict(sorted(recent_quantity_support.items())),
        weights=weights.normalized(),
        cutoff=cutoff,
    )


def _scale(values: dict[str, float]) -> dict[str, float]:
    maximum = max(values.values(), default=0.0)
    return {key: value / maximum if maximum else 0.0 for key, value in values.items()}


def hybrid_recommendations(
    artifact: HybridArtifact,
    cart_product_ids: Iterable[str],
    *,
    timestamp: datetime,
    allowed_product_ids: set[str],
    limit: int = 3,
) -> list[tuple[str, float, str]]:
    cart = set(cart_product_ids)
    candidates = allowed_product_ids - cart
    local = local_timestamp(timestamp)

    popularity = _scale({
        product: float(count)
        for product, count in artifact.quantity_support.items()
        if product in candidates
    })
    recent = _scale({
        product: float(count)
        for product, count in artifact.recent_quantity_support.items()
        if product in candidates
    })
    similarity: dict[str, float] = defaultdict(float)
    for item in artifact.base.similarities:
        value = item.jaccard
        if item.product_a in cart and item.product_b in candidates:
            similarity[item.product_b] += value
        elif item.product_b in cart and item.product_a in candidates:
            similarity[item.product_a] += value
    similarity = _scale(dict(similarity))

    association: dict[str, float] = defaultdict(float)
    for rule in artifact.base.association_rules:
        if rule.antecedent_product_id in cart and rule.consequent_product_id in candidates:
            evidence = min(1.0, rule.confidence * min(rule.lift, 2.0) / 2.0)
            association[rule.consequent_product_id] = max(
                association[rule.consequent_product_id], evidence
            )

    context = {}
    for product in candidates:
        product_total = artifact.quantity_support.get(product, 0)
        hour = artifact.hour_counts.get(product, {}).get(local.hour, 0)
        weekday = artifact.weekday_counts.get(product, {}).get(local.weekday(), 0)
        if product_total:
            context[product] = (hour + weekday) / (2 * product_total)
        else:
            context[product] = 0.0
    context = _scale(context)

    weights = artifact.weights
    scored = []
    for product in sorted(candidates):
        score = (
            weights.popularity * popularity.get(product, 0.0)
            + weights.similarity * similarity.get(product, 0.0)
            + weights.association * association.get(product, 0.0)
            + weights.context * (0.7 * context.get(product, 0.0) + 0.3 * recent.get(product, 0.0))
        )
        components = {
            "association": association.get(product, 0.0),
            "similarity": similarity.get(product, 0.0),
            "context": context.get(product, 0.0),
        }
        reason = max(components, key=components.get) if max(components.values(), default=0.0) > 0 else "popularity"
        scored.append((product, score, reason))
    scored.sort(key=lambda value: (-value[1], value[0]))
    return scored[:limit]
