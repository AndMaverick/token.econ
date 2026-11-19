import unittest

from token_econ import (
    Cohort,
    Tier,
    Trend,
    UnitEconomics,
    economics_at,
    expected_cost,
    grid,
    max_escalation_to_beat,
    project,
    squeeze_month,
    tornado,
)
from token_econ.routing import reach, savings

BASE = UnitEconomics(20, 200, 1500, 500, 3, 15)


class ProjectionTests(unittest.TestCase):
    def test_falling_prices_alone_raise_margin(self):
        pts = project(BASE, Trend(price_decline=0.1), 3)
        self.assertEqual([p.month for p in pts], [0, 1, 2, 3])
        self.assertAlmostEqual(pts[0].margin, 0.88)
        self.assertAlmostEqual(pts[1].inference_cost, 2.4 * 0.9)
        self.assertAlmostEqual(pts[1].margin, 1 - 2.16 / 20)
        self.assertIsNone(squeeze_month(BASE, Trend(price_decline=0.1), 0.8))

    def test_usage_outrunning_price_declines_squeezes(self):
        # cost x1.5 a month: 2.4, 3.6, 5.4. A 0.8 target allows 4.0, so month 2 is the first below it.
        self.assertEqual(squeeze_month(BASE, Trend(usage_growth=0.5), 0.8), 2)
        pts = project(BASE, Trend(usage_growth=0.5), 2)
        self.assertGreaterEqual(pts[1].margin, 0.8)
        self.assertLess(pts[2].margin, 0.8)

    def test_growth_and_decline_cancel_at_the_critical_rate(self):
        t = Trend(usage_growth=0.25)
        self.assertAlmostEqual(t.critical_price_decline(), 0.2)
        flat = Trend(usage_growth=0.25, price_decline=0.2)
        self.assertAlmostEqual(flat.cost_factor, 1.0)
        self.assertAlmostEqual(project(BASE, flat, 12)[12].margin, BASE.gross_margin())
        self.assertTrue(Trend(usage_growth=0.25, price_decline=0.1).squeezing)
        self.assertFalse(Trend(usage_growth=0.25, price_decline=0.25).squeezing)

    def test_token_growth_compounds_with_usage_growth(self):
        t = Trend(price_decline=0.03, usage_growth=0.04, token_growth=0.01)
        self.assertAlmostEqual(t.cost_factor, 1.04 * 1.01 * 0.97)
        e = economics_at(BASE, t, 6)
        self.assertAlmostEqual(e.inference_cost(), 2.4 * t.cost_factor ** 6)

    def test_already_below_target_is_month_zero(self):
        self.assertEqual(squeeze_month(BASE, Trend(), 0.95), 0)

    def test_never_squeezed_returns_none_within_the_horizon(self):
        self.assertIsNone(squeeze_month(BASE, Trend(usage_growth=0.001), 0.8, horizon=50))

    def test_price_growth_is_handled_by_search(self):
        self.assertIsNone(squeeze_month(BASE, Trend(usage_growth=0.05, price_growth=0.05), 0.8, horizon=60))

    def test_squeeze_month_agrees_with_the_full_model_at_the_boundary(self):
        for usage in (0.02, 0.05, 0.1, 0.3):
            t = Trend(usage_growth=usage, price_decline=0.01)
            m = squeeze_month(BASE, t, 0.75)
            self.assertIsNotNone(m)
            self.assertLess(economics_at(BASE, t, m).gross_margin(), 0.75)
            self.assertGreaterEqual(economics_at(BASE, t, m - 1).gross_margin(), 0.75)

    def test_bad_trends(self):
        for bad in (lambda: Trend(price_decline=1.5), lambda: Trend(usage_growth=-2), lambda: project(BASE, Trend(), -1)):
            with self.assertRaises(ValueError):
                bad()


def paid_after(q: float, payments: int) -> float:
    return sum(10 * q ** t for t in range(payments))


class CohortTests(unittest.TestCase):
    def test_hand_computed(self):
        c = Cohort(profit_per_month=10, monthly_churn=0.1, cac=30)
        self.assertAlmostEqual(c.ltv(), 100)                # 10 / 0.1
        self.assertAlmostEqual(c.ltv_to_cac(), 100 / 30)
        self.assertAlmostEqual(c.expected_months(), 10)
        # cumulative expected profit after n payments is 10 x (1 - 0.9^n) / 0.1: 27.1 after 3, 34.39 after 4, and CAC is 30
        self.assertEqual(c.payments_to_recover_cac(), 4)

    def test_lifetime_value_matches_a_long_explicit_sum(self):
        for churn, disc in [(0.05, 0.0), (0.1, 0.01), (0.02, 0.02)]:
            c = Cohort(12.5, churn, 40, disc)
            q = (1 - churn) / (1 + disc)
            self.assertAlmostEqual(c.ltv(), sum(12.5 * q ** t for t in range(5000)), places=6)

    def test_no_churn_no_discount_never_ends(self):
        c = Cohort(10, 0.0, 30)
        self.assertIsNone(c.ltv())
        self.assertEqual(c.payments_to_recover_cac(), 3)
        self.assertIsNone(Cohort(10, 0.0, 0.0).ltv_to_cac())

    def test_never_pays_back(self):
        self.assertIsNone(Cohort(1, 0.5, 5).payments_to_recover_cac())      # lifetime value is 2, below the 5 it cost to win
        self.assertIsNone(Cohort(-1, 0.1, 5).payments_to_recover_cac())
        self.assertEqual(Cohort(10, 0.1, 0).payments_to_recover_cac(), 0)

    def test_payback_is_the_first_payment_that_covers_cac(self):
        for churn, disc, cac in [(0.1, 0.0, 30), (0.03, 0.01, 55), (0.2, 0.0, 12), (0.0, 0.02, 20)]:
            c = Cohort(10, churn, cac, disc)
            n = c.payments_to_recover_cac()
            self.assertIsNotNone(n)
            self.assertGreaterEqual(paid_after(c.retention, n), cac)
            self.assertLess(paid_after(c.retention, n - 1), cac)

    def test_from_economics(self):
        c = Cohort.from_economics(BASE, 0.05, cac=50)
        self.assertAlmostEqual(c.ltv(), 17.6 / 0.05)


class RoutingTests(unittest.TestCase):
    def test_expected_cost_and_reach(self):
        tiers = [Tier("small", 0.002, 0.3), Tier("large", 0.02)]
        self.assertAlmostEqual(expected_cost(tiers), 0.002 + 0.3 * 0.02)
        self.assertEqual(reach(tiers), [1.0, 0.3])
        self.assertAlmostEqual(savings(tiers), 1 - 0.008 / 0.02)

    def test_three_tiers(self):
        tiers = [Tier("a", 1, 0.5), Tier("b", 4, 0.5), Tier("c", 10)]
        self.assertAlmostEqual(expected_cost(tiers), 1 + 0.5 * 4 + 0.25 * 10)

    def test_the_break_even_escalation_rate(self):
        threshold = max_escalation_to_beat(0.002, 0.02)
        self.assertAlmostEqual(threshold, 0.9)
        at = expected_cost([Tier("s", 0.002, threshold), Tier("l", 0.02)])
        self.assertAlmostEqual(at, 0.02)                                       # equal to always-strong exactly at the threshold
        self.assertLess(expected_cost([Tier("s", 0.002, 0.8), Tier("l", 0.02)]), 0.02)
        self.assertGreater(expected_cost([Tier("s", 0.002, 0.95), Tier("l", 0.02)]), 0.02)
        self.assertEqual(max_escalation_to_beat(0.05, 0.02), 0.0)             # a "cheap" model that costs more can never win

    def test_bad_cascades(self):
        for bad in (lambda: expected_cost([]), lambda: expected_cost([Tier("a", 1, 0.5)]), lambda: Tier("a", 1, 1.5), lambda: max_escalation_to_beat(1, 0)):
            with self.assertRaises(ValueError):
                bad()


class SensitivityTests(unittest.TestCase):
    def test_tornado_ranks_by_margin_swing(self):
        rows = tornado(BASE, 0.2)
        by = {r.field: r for r in rows}
        self.assertEqual(rows[0].field, "price_per_user")                    # 0.9 vs 0.85: five points, hand computed
        self.assertAlmostEqual(by["price_per_user"].span, 0.05)
        self.assertEqual(rows[1].field, "requests_per_user")                 # 0.904 vs 0.856: 4.8 points
        self.assertAlmostEqual(by["requests_per_user"].span, 0.048)
        self.assertAlmostEqual(by["output_tokens"].span, 0.03)
        self.assertAlmostEqual(by["input_tokens"].span, 0.018)
        self.assertNotIn("cache_hit_rate", by)                                # zero inputs are skipped
        self.assertEqual([r.span for r in rows], sorted((r.span for r in rows), reverse=True))

    def test_rates_stay_inside_zero_to_one(self):
        e = BASE.with_(cache_hit_rate=0.9, cached_input_price_per_1m=0.3)
        row = next(r for r in tornado(e, 0.5) if r.field == "cache_hit_rate")
        self.assertEqual(row.high_value, 1.0)

    def test_grid(self):
        g = grid(BASE, "requests_per_user", [100, 200], "price_per_user", [10, 20])
        self.assertAlmostEqual(g[0][1], 1 - 1.2 / 20)
        self.assertAlmostEqual(g[1][0], 1 - 2.4 / 10)
        self.assertEqual(len(g), 2)

    def test_bad_pct(self):
        for pct in (0, 1, -0.1):
            with self.assertRaises(ValueError):
                tornado(BASE, pct)


if __name__ == "__main__":
    unittest.main()
