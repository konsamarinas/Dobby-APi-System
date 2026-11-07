import unittest
from datetime import datetime

from src.main import RecommendationRequest, business_products, recommendation_candidates


TIMESTAMP = datetime.fromisoformat("2026-10-31T09:20:00+02:00")


class RecommendationQualityTests(unittest.TestCase):
    def setUp(self):
        self.business_id = "shop_university"
        self.products = business_products(self.business_id)

    def product_id(self, internal_name):
        return next(product["id"] for product in self.products
                    if product["internal_name"] == internal_name)

    def recommend(self, names, business_id=None):
        business_id = business_id or self.business_id
        products = business_products(business_id)
        ids = {product["internal_name"]: product["id"] for product in products}
        request = RecommendationRequest(
            business_id=business_id, order_type="dine_in", channel="kiosk",
            timestamp=TIMESTAMP,
            cart=[{"product_id": ids[name], "quantity": 1} for name in names],
            limit=3,
        )
        return recommendation_candidates(request)

    def test_burger_prefers_complementary_side_and_drink(self):
        results = self.recommend(["beef burger"])
        names = {result.name for result in results}
        self.assertTrue({"Πατάτες τηγανητές", "Αναψυκτικό 330ml"} & names)
        self.assertNotIn("Freddo espresso", names)
        self.assertTrue(all(result.type == "cross_sell" for result in results))

    def test_coffee_has_upgrade_and_breakfast_cross_sell(self):
        results = self.recommend(["freddo espresso"])
        self.assertEqual(results[0].type, "upsell")
        self.assertEqual(results[0].replaces_product_id, self.product_id("freddo espresso"))
        self.assertTrue(any(result.name in {"Κρουασάν βουτύρου", "Κουλούρι ολικής", "Σπανακόπιτα"}
                            for result in results))

    def test_cart_items_are_excluded_and_existing_upgrade_is_not_repeated(self):
        cart = ["beef burger", "soft drink 330ml", "french fries"]
        results = self.recommend(cart)
        ids = {result.product_id for result in results}
        self.assertTrue(ids.isdisjoint({self.product_id(name) for name in cart}))
        results = self.recommend(["freddo cappuccino"])
        self.assertNotIn(self.product_id("freddo cappuccino"), {result.product_id for result in results})
        self.assertNotIn("freddo espresso", {result.name for result in results})

    def test_breakfast_gets_beverage_opportunity(self):
        results = self.recommend(["whole wheat koulouri"])
        self.assertTrue(any(result.category_id.endswith(("cat_coffee", "cat_cold_beverage"))
                            for result in results))

    def test_business_context_changes_results_and_is_deterministic(self):
        first = self.recommend(["beef burger"])
        second = self.recommend(["beef burger"])
        self.assertEqual(first, second)
        commercial = self.recommend(["beef burger"], "shop_commercial")
        self.assertNotEqual([item.product_id for item in first],
                            [item.product_id for item in commercial])
        self.assertTrue(all(item.product_id.startswith("shop_commercial_") for item in commercial))


if __name__ == "__main__":
    unittest.main()
