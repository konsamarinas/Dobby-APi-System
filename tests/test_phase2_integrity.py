import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from src.core.data_loader import load_datasets
from src.core.preprocessing import local_date, utc_timestamp
from src.main import ORDERS, metrics


def fixture_payloads() -> dict[str, list[dict]]:
    product = {
        "id": "business_a_product_1",
        "business_id": "business_a",
        "category": {"id": "business_a_category_1", "business_id": "business_a"},
        "available": True,
        "published": True,
        "order_types": ["dine_in"],
        "price_store": 100,
        "price_delivery": 100,
        "price_takeaway": 100,
    }
    return {
        "products.json": [product],
        "orders.json": [{
            "id": "business_a_order_1", "business_id": "business_a", "type": "dine_in",
            "created_at": "2026-01-01T10:00:00+02:00",
            "items": [{"id": "business_a_item_1", "product_id": product["id"], "quantity": 1,
                       "category": {"id": product["category"]["id"]}}],
        }],
        "sales_stats.json": [{
            "id": product["id"], "category": {"id": product["category"]["id"]},
        }],
    }


def load_fixture(payloads: dict[str, list[dict]], omit: str | None = None) -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        for name, payload in payloads.items():
            if name != omit:
                (path / name).write_text(json.dumps(payload), encoding="utf-8")
        load_datasets(path)


class Phase2IntegrityTests(unittest.TestCase):
    def assert_fixture_rejected(self, mutate, message: str) -> None:
        payloads = fixture_payloads()
        mutate(payloads)
        with self.assertRaisesRegex(RuntimeError, message):
            load_fixture(payloads)

    def test_missing_required_dataset_file_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError, r"Required dataset file is missing: .*sales_stats\.json"):
            load_fixture(fixture_payloads(), omit="sales_stats.json")

    def test_cross_business_product_reference_is_rejected(self):
        def mutate(payloads):
            payloads["products.json"].append({
                "id": "business_b_product_1", "business_id": "business_b",
                "category": {"id": "business_b_category_1", "business_id": "business_b"},
                "available": True, "published": True, "order_types": ["dine_in"],
                "price_store": 100, "price_delivery": 100, "price_takeaway": 100,
            })
            payloads["orders.json"][0]["items"][0]["product_id"] = "business_b_product_1"

        self.assert_fixture_rejected(mutate, "references a product from another business")

    def test_item_category_mismatch_is_rejected(self):
        def mutate(payloads):
            payloads["orders.json"][0]["items"][0]["category"]["id"] = "wrong_category"

        self.assert_fixture_rejected(mutate, "category does not match product")

    def test_unknown_sales_stat_and_duplicate_stat_ids_are_rejected(self):
        def mutate(payloads):
            payloads["sales_stats.json"][0]["id"] = "unknown_product"

        self.assert_fixture_rejected(mutate, "references an unknown product")

        def duplicate(payloads):
            payloads["sales_stats.json"].append(dict(payloads["sales_stats.json"][0]))

        self.assert_fixture_rejected(duplicate, "sales_stats.json contains duplicate IDs")

    def test_dst_offsets_are_sorted_by_absolute_time(self):
        timestamps = [
            "2026-10-25T02:15:00+02:00",
            "2026-10-25T02:30:00+03:00",
        ]
        ordered = sorted(timestamps, key=utc_timestamp)
        self.assertEqual(ordered[0], timestamps[1])
        self.assertEqual(local_date(timestamps[0]), date(2026, 10, 25))

    def test_recommendation_period_contains_only_eligible_baskets(self):
        business_id = ORDERS[0]["business_id"]
        orders = [order for order in ORDERS if order["business_id"] == business_id]
        test_orders = orders[max(1, int(len(orders) * 0.8)):]
        eligible_dates = [
            local_date(order["created_at"])
            for order in test_orders
            if len({item.get("product_id") for item in order.get("items", [])}) >= 2
        ]
        result = metrics(business_id)["recommendations"]
        self.assertEqual(
            result["test_period"],
            {"from": min(eligible_dates).isoformat(), "to": max(eligible_dates).isoformat()},
        )


if __name__ == "__main__":
    unittest.main()
