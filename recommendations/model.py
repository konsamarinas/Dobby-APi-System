"""Serializable recommendation model artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class AssociationRule:
    business_id: str
    antecedent_product_id: str
    consequent_product_id: str
    support: float
    confidence: float
    lift: float
    co_occurrence_count: int


@dataclass(frozen=True)
class Similarity:
    product_a: str
    product_b: str
    jaccard: float
    cosine: float
    co_occurrence_count: int


@dataclass
class RecommendationArtifact:
    model_version: str
    business_id: str
    training_cutoff: str | None
    transaction_count: int
    product_support: dict[str, int] = field(default_factory=dict)
    association_rules: list[AssociationRule] = field(default_factory=list)
    similarities: list[Similarity] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["association_rules"] = [asdict(rule) for rule in self.association_rules]
        value["similarities"] = [asdict(similarity) for similarity in self.similarities]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RecommendationArtifact":
        return cls(
            model_version=value["model_version"],
            business_id=value["business_id"],
            training_cutoff=value.get("training_cutoff"),
            transaction_count=int(value["transaction_count"]),
            product_support={key: int(count) for key, count in value.get("product_support", {}).items()},
            association_rules=[AssociationRule(**rule) for rule in value.get("association_rules", [])],
            similarities=[Similarity(**similarity) for similarity in value.get("similarities", [])],
        )


def cutoff_string(cutoff: datetime | None) -> str | None:
    if cutoff is None:
        return None
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("training cutoff must be timezone-aware")
    return cutoff.isoformat()
