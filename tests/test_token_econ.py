import unittest

from token_econ import UnitEconomics, price_for_margin, sweep

BASE = UnitEconomics(price_per_user=20, requests_per_user=200, input_tokens=1500,
                     output_tokens=500, input_price_per_1m=3, output_price_per_1m=15)


class EconomicsTests(unittest.TestCase):
    def test_cost_per_request(self):
        self.assertAlmostEqual(BASE.cost_per_request(), 0.012)

    def test_margin_and_profit(self):
        self.assertAlmostEqual(BASE.inference_cost(), 2.4)
        self.assertAlmostEqual(BASE.profit_per_user(), 17.6)
        self.assertAlmostEqual(BASE.gross_margin(), 0.88)

    def test_breakeven_users(self):
        self.assertEqual(BASE.breakeven_users(17_600), 1000)
        self.assertEqual(BASE.breakeven_users(17_601), 1001)

    def test_no_breakeven_when_losing_money(self):
        losing = BASE.with_(requests_per_user=5000)
        self.assertIsNone(losing.breakeven_users(1000))

    def test_max_requests_for_margin(self):
        self.assertEqual(BASE.max_requests_for_margin(0.90), 166)
        hit = BASE.with_(requests_per_user=166).gross_margin()
        miss = BASE.with_(requests_per_user=167).gross_margin()
        self.assertGreaterEqual(hit, 0.90)
        self.assertLess(miss, 0.90)

    def test_sweep_is_monotonic_in_usage(self):
        margins = [m for _, m in sweep(BASE, "requests_per_user", [100, 200, 400])]
        self.assertEqual(margins, sorted(margins, reverse=True))

    def test_price_for_margin_round_trips(self):
        price = price_for_margin(BASE, 0.9)
        self.assertAlmostEqual(BASE.with_(price_per_user=price).gross_margin(), 0.9)


if __name__ == "__main__":
    unittest.main()
