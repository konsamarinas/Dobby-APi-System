import unittest
from datetime import date, datetime, timedelta

from pydantic import ValidationError

from src.main import (
    PRODUCTS_BY_BUSINESS,
    RecommendationRequest,
    business_products,
    forecast,
    metrics,
    recommendation_candidates,
)
from src.core.config import athens_today
from src.forecasting.evaluation import previous_week_baseline, rolling_origin_evaluation


class DobbyApiTests(unittest.TestCase):
    business_id = next(iter(PRODUCTS_BY_BUSINESS))
    products = business_products(business_id)
    freddo_espresso = next(product["id"] for product in products if product["internal_name"] == "freddo espresso")
    koulouri = next(product["id"] for product in products if product["internal_name"] == "whole wheat koulouri")

    def request(self, cart=None, order_type="dine_in"):
        return RecommendationRequest(
            business_id=self.business_id,
            order_type=order_type,
            channel="kiosk",
            timestamp="2026-10-31T09:20:00+02:00",
            cart=cart or [],
        )

    def test_empty_cart_returns_available_recommendations(self):
        recommendations = recommendation_candidates(self.request())
        self.assertLessEqual(len(recommendations), 3)
        self.assertTrue(all(item.score <= 1 for item in recommendations))

    def test_cart_products_are_excluded(self):
        recommendations = recommendation_candidates(
            self.request([{"product_id": self.freddo_espresso, "quantity": 1}])
        )
        self.assertNotIn(self.freddo_espresso, {item.product_id for item in recommendations})

    def test_legitimate_upsell_declares_replacement(self):
        recommendations = recommendation_candidates(
            self.request([{"product_id": self.freddo_espresso, "quantity": 1}])
        )
        upsells = [item for item in recommendations if item.type == "upsell"]
        self.assertTrue(upsells)
        self.assertEqual(upsells[0].replaces_product_id, self.freddo_espresso)

    def test_order_type_price_is_returned_in_cents(self):
        recommendations = recommendation_candidates(self.request(order_type="delivery"))
        prices = {item.product_id: item.price for item in recommendations}
        self.assertEqual(prices[self.koulouri], 140)

    def test_timezone_and_channel_validation(self):
        with self.assertRaises(ValidationError):
            self.request()
            RecommendationRequest(
                business_id="shop_01",
                order_type="dine_in",
                channel="api",
                timestamp="2026-10-31T09:20:00",
                cart=[],
            )

    def test_timestamps_are_normalized_to_athens_local_time(self):
        from src.main import local_date, local_timestamp

        self.assertEqual(local_date("2026-10-31T23:30:00+00:00"), date(2026, 11, 1))
        self.assertEqual(local_timestamp("2026-10-31T23:30:00+00:00").hour, 1)

    def test_forecast_is_daily_and_interval_is_ordered(self):
        result = forecast(
            self.business_id,
            "product",
            self.freddo_espresso,
            start_date=date(2026, 11, 1),
            horizon=3,
        )
        self.assertEqual([point.date for point in result.forecast], [
            date(2026, 11, 1), date(2026, 11, 2), date(2026, 11, 3)
        ])
        self.assertTrue(all(point.lower <= point.quantity <= point.upper for point in result.forecast))

    def test_forecast_default_starts_tomorrow_in_athens(self):
        result = forecast(self.business_id, "product", self.freddo_espresso, horizon=1)
        expected = athens_today() + timedelta(days=1)
        self.assertEqual(result.forecast[0].date, expected)

    def test_rolling_origin_returns_nullable_wape_for_zero_demand(self):
        daily = {("p", date(2026, 1, day)): 0.0 for day in range(1, 11)}
        result = rolling_origin_evaluation(daily)
        self.assertIsNone(result["wape"])
        self.assertIsNone(result["baseline_wape"])

    def test_rolling_origin_baseline_does_not_use_future_values(self):
        daily = {("p", date(2026, 1, day)): float(day) for day in range(1, 16)}
        baseline = rolling_origin_evaluation(daily)
        changed = dict(daily)
        changed[("p", date(2026, 1, 20))] = 10000.0
        changed_result = rolling_origin_evaluation(changed)
        self.assertNotEqual(baseline["wape"], changed_result["wape"])
        self.assertEqual(previous_week_baseline({date(2026, 1, 8): 7.0}, date(2026, 1, 15)), 7.0)
        self.assertEqual(previous_week_baseline({date(2026, 1, 16): 99.0}, date(2026, 1, 15)), 0.0)

    def test_metrics_use_temporal_evaluation(self):
        result = metrics(self.business_id)
        self.assertIn("hit_rate_at_3", result["recommendations"])
        self.assertIn("baseline_wape", result["forecast"])
        self.assertIsNone(result["business_impact"]["estimated_aov_uplift_pct"])


if __name__ == "__main__":
    unittest.main()
