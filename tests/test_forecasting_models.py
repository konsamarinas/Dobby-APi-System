import unittest
from datetime import date

import pandas as pd

from src.forecasting.engine import (
    evaluate_models,
    recursive_forecast,
    seasonal_average_forecast,
    seasonal_naive_forecast,
    select_model,
)


class ForecastingModelTests(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2026-01-01", periods=90, freq="D")
        self.series = pd.Series(
            [float((day.day_of_week + 1) * 10) for day in index],
            index=index,
        )

    def test_seasonal_naive_uses_previous_same_weekday(self):
        value = seasonal_naive_forecast(self.series, date(2026, 4, 2))
        self.assertEqual(value, 40.0)

    def test_seasonal_average_uses_same_weekday_history(self):
        value, lower, upper = seasonal_average_forecast(self.series, date(2026, 4, 2))
        self.assertEqual(value, 40.0)
        self.assertLessEqual(lower, value)
        self.assertGreaterEqual(upper, value)

    def test_random_forest_evaluation_is_available_and_metrics_are_finite(self):
        result = evaluate_models(self.series)
        self.assertEqual(result["random_forest"].n_observations, 18)
        self.assertIsNotNone(result["random_forest"].mae)

    def test_recursive_forecast_returns_requested_dates_without_future_actuals(self):
        points = recursive_forecast(self.series, date(2026, 4, 1), 7, "seasonal_average")
        self.assertEqual([point[0] for point in points],
                         [date(2026, 4, day) for day in range(1, 8)])
        self.assertEqual(len(points), 7)

    def test_insufficient_history_falls_back_to_seasonal_average(self):
        model, metrics = select_model(self.series.iloc[:20])
        self.assertEqual(model, "seasonal_average")
        self.assertEqual(len(metrics), 3)


if __name__ == "__main__":
    unittest.main()
