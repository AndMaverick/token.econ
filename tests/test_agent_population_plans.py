import math
import unittest
from statistics import NormalDist

from token_econ import (
    AgentRun,
    Empirical,
    Fixed,
    FreeTier,
    Lognormal,
    Pareto,
    Plan,
    TokenPrice,
    UnitEconomics,
    compare,
    concentration,
    cost_curve,
    max_steps_for_budget,
    simulate,
    solve_cap,
)

BASE = UnitEconomics(20, 200, 1500, 500, 3, 15)
PRICE = TokenPrice(3, 15)


class AgentTests(unittest.TestCase):
    RUN = AgentRun(steps=4, base_context_tokens=1000, added_tokens_per_step=500, output_tokens_per_step=200)

    def test_input_tokens_are_quadratic_in_steps(self):
        self.assertEqual(self.RUN.input_tokens(), 1000 + 1500 + 2000 + 2500)   # each step resends the whole context
        self.assertEqual(self.RUN.input_tokens(), 4 * 1000 + 500 * 4 * 3 / 2)
        self.assertAlmostEqual(self.RUN.cost(PRICE), (7000 * 3 + 800 * 15) / 1e6)
        self.assertAlmostEqual(self.RUN.cost(PRICE), 0.033)

    def test_cost_more_than_doubles_when_steps_double(self):
        curve = dict(cost_curve(self.RUN, PRICE, [5, 10, 20, 40]))
        self.assertGreater(curve[10], 2 * curve[5])
        self.assertGreater(curve[40], 2 * curve[20])

    def test_caching_hand_computed(self):
        # step 0 writes 1000 at 3.75; later steps read the prior context at 0.3 and write 500 new tokens at 3.75; output 200 x 15 each step
        cached = self.RUN.with_(cached=True)
        price = TokenPrice(3, 15, 0.3, 3.75)
        expected = (1000 * 3.75 + (1000 * 0.3 + 500 * 3.75) + (1500 * 0.3 + 500 * 3.75) + (2000 * 0.3 + 500 * 3.75) + 800 * 15) / 1e6
        self.assertAlmostEqual(cached.cost(price), expected)
        self.assertAlmostEqual(cached.cost(price), 0.022725)
        self.assertLess(cached.cost(price), self.RUN.cost(price))

    def test_budget_search(self):
        self.assertEqual(max_steps_for_budget(self.RUN, PRICE, 0.033), 4)          # exactly affordable
        self.assertEqual(max_steps_for_budget(self.RUN, PRICE, 0.0329999), 3)
        self.assertEqual(max_steps_for_budget(self.RUN, PRICE, 0.0), 0)
        n = max_steps_for_budget(self.RUN, PRICE, 1.0)
        self.assertLessEqual(self.RUN.with_(steps=n).cost(PRICE), 1.0)
        self.assertGreater(self.RUN.with_(steps=n + 1).cost(PRICE), 1.0)

    def test_zero_steps_and_bad_input(self):
        self.assertEqual(self.RUN.with_(steps=0).cost(PRICE), 0.0)
        with self.assertRaises(ValueError):
            self.RUN.with_(steps=-1)
        with self.assertRaises(ValueError):
            self.RUN.with_(steps=2.5)


class PopulationTests(unittest.TestCase):
    def test_points_average_to_the_exact_mean_for_any_n(self):
        for dist in [Lognormal(100, 1.0), Lognormal(40, 0.3), Pareto(50, 2.0), Pareto(10, 1.2), Fixed(7)]:
            for n in (1, 10, 1000):
                pts = dist.points(n)
                self.assertEqual(len(pts), n)
                self.assertAlmostEqual(sum(pts) / n, dist.mean(), delta=dist.mean() * 1e-9, msg=f"{dist} n={n}")

    def test_lognormal_mean_and_median(self):
        d = Lognormal(100, 1.0)
        self.assertAlmostEqual(d.mean(), 100 * math.exp(0.5))
        pts = d.points(1000)
        self.assertAlmostEqual(d.quantile(0.5), 100)
        self.assertEqual(pts, sorted(pts))

    def test_top_share_matches_the_analytic_value(self):
        d = Lognormal(100, 1.0)
        z = NormalDist().inv_cdf(0.99)
        analytic = 1 - NormalDist().cdf(z - 1.0)       # the top 1 percent of a lognormal holds 1 - Phi(z - sigma) of the total
        self.assertAlmostEqual(concentration(d.points(1000), 0.01), analytic, places=3)
        p = Pareto(50, 2.0)
        self.assertAlmostEqual(p.top_share(0.01), 0.1)    # 0.01 ** (1 - 1/2)
        self.assertAlmostEqual(concentration(p.points(1000), 0.01), 0.1, places=9)

    def test_heavier_tails_concentrate_more(self):
        shares = [concentration(Lognormal(100, s).points(1000), 0.01) for s in (0.3, 0.8, 1.3)]
        self.assertEqual(shares, sorted(shares))

    def test_empirical_and_fixed(self):
        e = Empirical([30, 10, 20])
        self.assertEqual(e.points(), [10.0, 20.0, 30.0])
        self.assertEqual(e.mean(), 20)
        self.assertEqual(Fixed(5).points(3), [5, 5, 5])
        self.assertEqual(concentration([0, 0, 0]), 0.0)

    def test_bad_distributions(self):
        for bad in (lambda: Lognormal(0, 1), lambda: Lognormal(1, -1), lambda: Pareto(1, 1.0), lambda: Pareto(0, 2), lambda: Empirical([]),
                    lambda: Empirical([-1]), lambda: Fixed(1).points(0)):
            with self.assertRaises(ValueError):
                bad()


class PlanTests(unittest.TestCase):
    def test_flat_plan_on_a_uniform_population_matches_unit_economics(self):
        r = simulate(BASE, Plan("flat", 20), Fixed(200))
        self.assertAlmostEqual(r.profit_per_user, BASE.profit_per_user())
        self.assertAlmostEqual(r.margin, BASE.gross_margin())
        self.assertEqual((r.loss_share, r.revenue_per_user), (0.0, 20))
        self.assertAlmostEqual(r.cost_per_user, 2.4)

    def test_cap_overage_and_included(self):
        self.assertAlmostEqual(simulate(BASE, Plan("c", 20, hard_cap=100), Fixed(200)).profit_per_user, 20 - 1.2)
        # 10 covers 100 requests, then 5 cents each: revenue 10 + 100 x 0.05 = 15, cost 2.4
        self.assertAlmostEqual(simulate(BASE, Plan("h", 10, included_requests=100, overage_per_request=0.05), Fixed(200)).profit_per_user, 12.6)
        self.assertAlmostEqual(simulate(BASE, Plan("m", 0, included_requests=0, overage_per_request=0.02), Fixed(200)).revenue_per_user, 4.0)

    def test_fees_follow_revenue(self):
        e = BASE.with_(payment_fee_rate=0.03, payment_fee_fixed=0.3)
        self.assertAlmostEqual(simulate(e, Plan("flat", 20), Fixed(200)).profit_per_user, 20 - 0.9 - 2.4)
        self.assertAlmostEqual(simulate(e, Plan("free", 0), Fixed(0)).profit_per_user, 0.0)   # no revenue, no fixed fee

    def test_heavy_users_can_lose_money_on_a_flat_plan(self):
        r = simulate(BASE, Plan("flat", 20), Fixed(5000))
        self.assertEqual(r.loss_share, 1.0)
        self.assertLess(r.profit_per_user, 0)
        skewed = simulate(BASE, Plan("flat", 20), Lognormal(150, 1.4))
        self.assertGreater(skewed.loss_share, 0.0)
        self.assertLess(skewed.p99_profit, 0.0)
        self.assertGreater(skewed.p50_profit, 0.0)
        self.assertGreater(skewed.top_cost_share, 0.1)

    def test_a_cap_helps_most_when_the_tail_is_heavy(self):
        pop = Lognormal(150, 1.4)
        flat, capped = simulate(BASE, Plan("f", 20), pop), simulate(BASE, Plan("c", 20, hard_cap=500), pop)
        self.assertGreater(capped.profit_per_user, flat.profit_per_user)
        self.assertEqual(capped.loss_share, 0.0)

    def test_compare_orders_by_profit(self):
        results = compare(BASE, [Plan("a", 10), Plan("b", 30), Plan("c", 20)], Fixed(100))
        self.assertEqual([r.plan for r in results], ["b", "c", "a"])

    def test_solve_cap_hand_computed(self):
        # four users at 100, 200, 300, 400 requests; cost 0.012 each; flat 20; target 0.9 means cost per user at most 2.0,
        # so mean served requests at most 166.67. With cap c between 100 and 200 the mean is (100 + 3c) / 4, so c = 188.888...
        pop = Empirical([100, 200, 300, 400])
        cap = solve_cap(BASE, Plan("f", 20), pop, 0.9)
        self.assertAlmostEqual(cap, 188.8888888, places=5)
        self.assertGreaterEqual(simulate(BASE, Plan("f", 20, hard_cap=cap), pop).margin, 0.9 - 1e-9)
        self.assertAlmostEqual(solve_cap(BASE, Plan("f", 20), pop, 0.99999), 20 * 0.00001 / 0.012, places=6)   # only a sliver of a request
        self.assertIsNone(solve_cap(BASE.with_(other_cost_per_user=3), Plan("f", 20), pop, 0.9))   # fixed costs alone already miss it
        self.assertEqual(solve_cap(BASE, Plan("f", 20), Fixed(10), 0.5), math.inf)      # already fine with no cap

    def test_free_tier(self):
        ft = FreeTier(free_per_paid=4, free_requests=30)
        self.assertAlmostEqual(ft.cost_per_free_user(BASE), 0.36)
        self.assertAlmostEqual(ft.blended_profit_per_paid_user(BASE), 17.6 - 4 * 0.36)
        self.assertAlmostEqual(ft.required_conversion(BASE), 0.36 / (17.6 + 0.36))
        self.assertIsNone(ft.required_conversion(BASE.with_(requests_per_user=5000)))

    def test_bad_plans(self):
        for bad in (lambda: Plan("x", -1), lambda: Plan("x", 1, hard_cap=-1), lambda: FreeTier(-1, 1)):
            with self.assertRaises(ValueError):
                bad()


if __name__ == "__main__":
    unittest.main()
