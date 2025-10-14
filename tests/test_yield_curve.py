import unittest

from src.core.yield_curve import YieldCurve


class YieldCurveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tenors = [3, 6, 12, 24, 36, 60, 84, 120]
        self.rates = [0.0425, 0.0415, 0.0410, 0.0395, 0.0385, 0.0380, 0.0395, 0.0415]
        self.curve = YieldCurve(self.tenors, self.rates)

    def test_exact_tenor_lookup(self) -> None:
        self.assertAlmostEqual(self.curve.get_rate(12), 0.0410)

    def test_linear_interpolation_midpoint(self) -> None:
        rate_18m = self.curve.get_rate(18)
        expected = (0.0410 + 0.0395) / 2
        self.assertAlmostEqual(rate_18m, expected, places=3)

    def test_discount_factor_calculation(self) -> None:
        df_24m = self.curve.get_discount_factor(24)
        expected = 1 / (1 + 0.0395) ** 2
        self.assertAlmostEqual(df_24m, expected, places=4)

    def test_parallel_shock_adjusts_rates(self) -> None:
        shocked = self.curve.apply_parallel_shock(100)
        self.assertAlmostEqual(shocked.get_rate(12), 0.0510)
        self.assertAlmostEqual(shocked.get_rate(120), 0.0515)

    def test_non_parallel_shock_applies_pointwise(self) -> None:
        shocks = {3: -100, 120: 100}
        shocked = self.curve.apply_non_parallel_shock(shocks)
        self.assertAlmostEqual(shocked.get_rate(3), 0.0325)
        self.assertAlmostEqual(shocked.get_rate(120), 0.0515)


if __name__ == "__main__":
    unittest.main()
