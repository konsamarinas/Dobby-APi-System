import unittest
from datetime import datetime, timezone

from src.recommendations.evaluation import build_evaluation_plan, evaluate_all
from src.recommendations.hybrid import HybridWeights, hybrid_recommendations, train_hybrid
from src.recommendations.inference import association_recommendations, similarity_recommendations
from src.recommendations.training import (
    train_association_rules,
    train_item_similarity,
    transaction_baskets,
)


def orders_fixture():
    def order(number, products, business="a", timestamp=f"2026-01-{1:02d}T10:00:00+02:00"):
        return {
            "id": f"{business}-order-{number}", "business_id": business, "type": "dine_in",
            "created_at": timestamp,
            "items": [{"id": f"{business}-item-{number}-{i}", "product_id": product, "quantity": 1}
                      for i, product in enumerate(products)],
        }

    return [
        order(1, ["a", "b"]), order(2, ["a", "b"]),
        order(3, ["a", "c"]), order(4, ["b", "c"]),
        order(5, ["a", "a", "b"], timestamp="2026-01-02T10:00:00+02:00"),
        order(6, ["x"], business="b"),
    ]


class RecommendationModelTests(unittest.TestCase):
    def test_binary_baskets_cutoff_and_business_isolation(self):
        rows = transaction_baskets(
            orders_fixture(), "a",
            cutoff=datetime(2026, 1, 2, 8, tzinfo=timezone.utc),
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0][1], frozenset({"a", "b"}))
        self.assertNotIn("x", set().union(*(basket for _, basket, _ in rows)))

    def test_association_support_confidence_lift_and_direction(self):
        artifact = train_association_rules(
            orders_fixture(), "a", min_support=0.5, min_confidence=0.5,
            min_lift=0.0, min_co_occurrence=2,
        )
        rule = next(rule for rule in artifact.association_rules
                    if rule.antecedent_product_id == "a" and rule.consequent_product_id == "b")
        self.assertAlmostEqual(rule.support, 3 / 5)
        self.assertAlmostEqual(rule.confidence, 3 / 4)
        self.assertAlmostEqual(rule.lift, (3 / 4) / (4 / 5))
        self.assertTrue(any(rule.antecedent_product_id == "b" and rule.consequent_product_id == "a"
                            for rule in artifact.association_rules))

    def test_similarity_formulas_symmetry_and_zero_diagonal(self):
        artifact = train_item_similarity(orders_fixture(), "a", min_co_occurrence=2)
        similarity = next(item for item in artifact.similarities
                          if item.product_a == "a" and item.product_b == "b")
        self.assertAlmostEqual(similarity.jaccard, 3 / 5)
        self.assertAlmostEqual(similarity.cosine, 3 / (4 * 4) ** 0.5)
        self.assertTrue(all(item.product_a != item.product_b for item in artifact.similarities))
        self.assertEqual(
            similarity.jaccard,
            next(item.jaccard for item in artifact.similarities
                 if item.product_a == "a" and item.product_b == "b"),
        )

    def test_filters_and_cold_start(self):
        artifact = train_association_rules(
            orders_fixture(), "a", min_support=0.9, min_confidence=0.9,
            min_co_occurrence=10,
        )
        self.assertEqual(artifact.association_rules, [])
        self.assertEqual(association_recommendations(artifact, ["cold"]), [])

    def test_inference_excludes_cart_and_supports_both_metrics(self):
        artifact = train_item_similarity(orders_fixture(), "a", min_co_occurrence=2)
        for metric in ("jaccard", "cosine"):
            result = similarity_recommendations(artifact, ["a"], metric=metric)
            self.assertTrue(result)
            self.assertNotIn("a", {product for product, _ in result})

    def test_training_is_deterministic(self):
        first = train_association_rules(orders_fixture(), "a", min_co_occurrence=1).to_dict()
        second = train_association_rules(orders_fixture(), "a", min_co_occurrence=1).to_dict()
        self.assertEqual(first, second)

    def test_identical_evaluation_population_for_independent_models(self):
        orders = orders_fixture() + [
            {
                "id": f"a-order-{number}", "business_id": "a", "type": "dine_in",
                "created_at": f"2026-01-{number:02d}T10:00:00+02:00",
                "items": [{"id": f"a-item-{number}-0", "product_id": "a", "quantity": 1},
                          {"id": f"a-item-{number}-1", "product_id": "b", "quantity": 1}],
            }
            for number in range(7, 12)
        ]
        results = evaluate_all(orders, [{"id": product, "business_id": "a"} for product in "abc"], "a",
                               min_co_occurrence=1, min_lift=0.0)
        counts = {result["evaluated_baskets"] for result in results}
        self.assertEqual(len(counts), 1)
        self.assertGreater(next(iter(counts)), 0)

    def test_canonical_split_preserves_global_order_and_hidden_item(self):
        orders = [
            {
                "id": f"order-{index}", "business_id": "a", "type": "dine_in",
                "created_at": f"2026-01-{index:02d}T10:00:00+00:00",
                "items": [{"product_id": product, "quantity": 1} for product in products],
            }
            for index, products in enumerate(
                (["a", "b"], ["b", "c"], ["c", "a"], ["a", "c"], ["b", "a"]),
                start=1,
            )
        ]
        plan = build_evaluation_plan(orders)
        self.assertEqual([order.order_id for order in plan.orders], [
            "order-1", "order-2", "order-3", "order-4", "order-5",
        ])
        self.assertEqual(len(plan.baskets), 1)
        self.assertEqual(plan.baskets[0].hidden_product_id, "a")
        self.assertEqual(plan.baskets[0].visible_product_ids, ("b",))

    def test_hybrid_is_deterministic_and_excludes_cart(self):
        plan = build_evaluation_plan(orders_fixture() + [
            {
                "id": "a-order-7", "business_id": "a", "type": "dine_in",
                "created_at": "2026-01-03T10:00:00+00:00",
                "items": [{"product_id": "a", "quantity": 1},
                          {"product_id": "b", "quantity": 1}],
            }
        ])
        history = list(plan.orders)
        artifact = train_item_similarity(orders_fixture(), "a", min_co_occurrence=1)
        hybrid = train_hybrid(
            [order for order in history if order.business_id == "a"],
            artifact,
            cutoff=history[-1].timestamp,
            weights=HybridWeights(2, 1, 0, 1),
        )
        results = hybrid_recommendations(
            hybrid, ["a"], timestamp=history[-1].timestamp,
            allowed_product_ids={"a", "b", "c"},
        )
        self.assertEqual(hybrid.weights, HybridWeights(0.5, 0.25, 0.0, 0.25))
        self.assertEqual(results, hybrid_recommendations(
            hybrid, ["a"], timestamp=history[-1].timestamp,
            allowed_product_ids={"a", "b", "c"},
        ))
        self.assertNotIn("a", {product for product, _, _ in results})


if __name__ == "__main__":
    unittest.main()
