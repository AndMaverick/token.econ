"""Properties over seeded random inputs, checked against a second implementation that uses exact fractions and no shared code."""
import math
import random
import unittest
from fractions import Fraction

from token_econ import (
    AgentRun,
    Cohort,
    Fixed,
    Plan,
    Tier,
    TokenPrice,
    Trend,
    UnitEconomics,
    economics_at,
    expected_cost,
    simulate,
    squeeze_month,
)
from token_econ.synthetic import random_economics

N = 3000


def get(f: dict[str, Fraction], key: str) -> Fraction:
    return f.get(key, Fraction(0))


def oracle_cost_per_request(f: dict[str, Fraction]) -> Fraction:
    g = lambda k: get(f, k)   # noqa: E731
    pin = f["input_price_per_1m"]
    read, write = f.get("cached_input_price_per_1m", pin), f.get("cache_write_price_per_1m", pin)
    tin, hit, wr = f["input_tokens"], g("cache_hit_rate"), g("cache_write_rate")
    base = (tin * (1 - hit - wr) * pin + tin * hit * read + tin * wr * write + (f["output_tokens"] + g("reasoning_tokens")) * f["output_price_per_1m"]) / 1_000_000
    return base * (1 + g("retry_rate")) * (1 + g("infra_overhead_rate"))


def oracle_profit(f: dict[str, Fraction]) -> Fraction:
    g = lambda k: get(f, k)   # noqa: E731
    price = f["price_per_user"]
    return price - (price * g("payment_fee_rate") + g("payment_fee_fixed")) - g("other_cost_per_user") - g("support_cost_per_user") \
        - f["requests_per_user"] * oracle_cost_per_request(f)


class ExactnessTests(unittest.TestCase):
    def test_float_answers_are_the_correctly_rounded_exact_answers(self):
        for seed in range(N):
            e, f = random_economics(seed)
            self.assertEqual(e.cost_per_request(), float(oracle_cost_per_request(f)), f"seed {seed}")
            self.assertEqual(e.profit_per_user(), float(oracle_profit(f)), f"seed {seed}")
            self.assertEqual(e.gross_margin(), float(oracle_profit(f) / f["price_per_user"]), f"seed {seed}")

    def test_breakeven_users_equals_the_exact_ceiling(self):
        rng = random.Random(1)
        solved = 0
        for seed in range(N):
            e, f = random_economics(seed)
            fixed = Fraction(rng.randint(1, 5_000_000), 100)
            profit = oracle_profit(f)
            expected = None if profit <= 0 else math.ceil(fixed / profit)
            self.assertEqual(e.breakeven_users(float(fixed)), expected, f"seed {seed}")
            if expected is not None:
                solved += 1
                self.assertGreaterEqual(expected * profit, fixed)              # that many users cover it...
                self.assertLess((expected - 1) * profit, fixed)                # ...and one fewer does not
        self.assertGreater(solved, N // 5)

    def test_max_requests_is_the_exact_boundary(self):
        rng = random.Random(2)
        checked = 0
        for seed in range(N):
            e, f = random_economics(seed)
            target = Fraction(rng.randint(0, 95), 100)
            got = e.max_requests_for_margin(float(target))
            per = oracle_cost_per_request(f)
            if per == 0:
                continue
            price = f["price_per_user"]
            budget = (price * (1 - target) - (price * get(f, "payment_fee_rate") + get(f, "payment_fee_fixed")) - get(f, "other_cost_per_user")
                      - get(f, "support_cost_per_user"))
            self.assertEqual(got, max(0, math.floor(budget / per)), f"seed {seed}")
            if got > 0:
                checked += 1
                at, over = f | {"requests_per_user": Fraction(got)}, f | {"requests_per_user": Fraction(got + 1)}
                self.assertGreaterEqual(oracle_profit(at) / price, target)
                self.assertLess(oracle_profit(over) / price, target)
        self.assertGreater(checked, N // 3)

    def test_price_for_margin_is_the_lowest_price_that_reaches_it(self):
        rng = random.Random(3)
        for seed in range(N):
            e, f = random_economics(seed)
            target = rng.randint(0, 90) / 100
            if 1 - float(f.get("payment_fee_rate", 0)) - target <= 0:
                with self.assertRaises(ValueError):
                    e.price_for_margin(target)
                continue
            price = e.price_for_margin(target)
            exact_price = Fraction(price)
            at = f | {"price_per_user": exact_price}
            self.assertGreaterEqual(float(oracle_profit(at) / exact_price), target - 1e-12, f"seed {seed}")
            below = f | {"price_per_user": exact_price * Fraction(999, 1000)}
            if exact_price > 0:
                self.assertLess(float(oracle_profit(below) / below["price_per_user"]), target, f"seed {seed}")


class ClosedFormTests(unittest.TestCase):
    def test_squeeze_month_equals_brute_force_search(self):
        rng = random.Random(4)
        found = 0
        for seed in range(600):
            e, _ = random_economics(seed)
            if e.gross_margin() <= 0:
                continue
            trend = Trend(price_decline=rng.randint(0, 8) / 100, usage_growth=rng.randint(0, 15) / 100, token_growth=rng.randint(0, 5) / 100)
            target = max(0.0, e.gross_margin() - rng.randint(1, 30) / 100)
            fast = squeeze_month(e, trend, target, horizon=120)
            slow = next((m for m in range(121) if economics_at(e, trend, m).gross_margin() < target), None)
            self.assertEqual(fast, slow, f"seed {seed} {trend} target {target}")
            found += slow is not None
        self.assertGreater(found, 100)

    def test_cohort_payback_equals_summation(self):
        rng = random.Random(5)
        never = 0
        for _ in range(2000):
            profit = rng.randint(1, 100) / 4
            c = Cohort(profit, rng.randint(0, 30) / 100, rng.randint(1, 600), rng.randint(0, 3) / 100)
            n = c.payments_to_recover_cac()
            q = c.retention
            if q >= 1:
                self.assertEqual(n, math.ceil(c.cac / profit))
                continue
            if n is None:
                never += 1
                self.assertLess(profit / (1 - q), c.cac * (1 + 1e-9))               # lifetime value never reaches CAC
                continue
            self.assertGreaterEqual(sum(profit * q ** t for t in range(n)), c.cac * (1 - 1e-12))
            self.assertLess(sum(profit * q ** t for t in range(n - 1)), c.cac * (1 + 1e-12))
        self.assertGreater(never, 50)

    def test_agent_input_tokens_formula_equals_the_loop(self):
        rng = random.Random(6)
        price = TokenPrice(3, 15)
        for _ in range(500):
            run = AgentRun(rng.randint(0, 60), rng.randint(0, 8000), rng.randint(0, 3000), rng.randint(0, 1000))
            looped = sum(run.step_usage(k).input_tokens for k in range(run.steps))
            self.assertEqual(run.input_tokens(), looped)
            self.assertAlmostEqual(run.cost(price), (looped * 3 + run.output_tokens() * 15) / 1e6, places=9)

    def test_cached_agent_never_costs_more_when_reads_are_cheaper_than_writes_and_fresh_input(self):
        rng = random.Random(7)
        price = TokenPrice(3, 15, 0.3, 3.0)           # write at the plain input price, read at a tenth
        for _ in range(300):
            run = AgentRun(rng.randint(1, 40), rng.randint(100, 8000), rng.randint(0, 3000), rng.randint(0, 1000))
            self.assertLessEqual(run.with_(cached=True).cost(price), run.cost(price) + 1e-12)

    def test_cascade_cost_equals_the_recursive_definition(self):
        rng = random.Random(8)

        def recursive(tiers: list[Tier]) -> float:
            head, rest = tiers[0], tiers[1:]
            return head.cost + (head.escalate * recursive(rest) if rest else 0.0)

        for _ in range(500):
            n = rng.randint(1, 5)
            tiers = [Tier(f"t{i}", rng.randint(1, 100) / 10, rng.randint(0, 100) / 100 if i < n - 1 else 0.0) for i in range(n)]
            self.assertAlmostEqual(expected_cost(tiers), recursive(tiers), places=12)


class PlanConsistencyTests(unittest.TestCase):
    def test_a_flat_plan_on_a_uniform_population_reproduces_unit_economics(self):
        for seed in range(500):
            e, _ = random_economics(seed)
            r = simulate(e, Plan("flat", e.price_per_user), Fixed(e.requests_per_user), users=10)
            self.assertAlmostEqual(r.profit_per_user, e.profit_per_user(), places=9, msg=f"seed {seed}")

    def test_unit_economics_is_never_mutated_by_scenarios(self):
        e, _ = random_economics(11)
        before = e.to_dict()
        e.with_(requests_per_user=1)
        economics_at(e, Trend(usage_growth=0.1), 12)
        simulate(e, Plan("flat", e.price_per_user), Fixed(10))
        self.assertEqual(e.to_dict(), before)
        self.assertIsInstance(e, UnitEconomics)


if __name__ == "__main__":
    unittest.main()
