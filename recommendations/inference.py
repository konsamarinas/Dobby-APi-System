"""Request-time inference over precomputed recommendation artifacts."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .model import RecommendationArtifact


def association_recommendations(
    artifact: RecommendationArtifact,
    cart_product_ids: Iterable[str],
    limit: int = 3,
) -> list[tuple[str, float]]:
    cart = set(cart_product_ids)
    scores: dict[str, float] = defaultdict(float)
    for rule in artifact.association_rules:
        if rule.antecedent_product_id in cart and rule.consequent_product_id not in cart:
            scores[rule.consequent_product_id] = max(
                scores[rule.consequent_product_id],
                min(1.0, rule.confidence * min(rule.lift, 2.0) / 2.0),
            )
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def similarity_recommendations(
    artifact: RecommendationArtifact,
    cart_product_ids: Iterable[str],
    metric: str = "jaccard",
    limit: int = 3,
) -> list[tuple[str, float]]:
    if metric not in {"jaccard", "cosine"}:
        raise ValueError("metric must be 'jaccard' or 'cosine'")
    cart = set(cart_product_ids)
    scores: dict[str, float] = defaultdict(float)
    for similarity in artifact.similarities:
        if similarity.product_a in cart and similarity.product_b not in cart:
            scores[similarity.product_b] += getattr(similarity, metric)
        elif similarity.product_b in cart and similarity.product_a not in cart:
            scores[similarity.product_a] += getattr(similarity, metric)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
