"""Measures the library and prints the numbers the README quotes. Run: python examples/evaluate.py

Every case is seeded, so every figure reproduces. "Exact" below means computed on fractions with no floating point, by code in
tests/test_properties.py that shares nothing with the library. It shows the arithmetic is right. It does not show that your
assumptions are: the prices, usage and churn are yours."""
from __future__ import annotations

import math
import random
import sys
import time
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_properties import get, oracle_cost_per_request, oracle_profit  # noqa: E402

from token_econ import Cohort, Lognormal, Pareto, Trend, UnitEconomics, concentration, economics_at, squeeze_month  # noqa: E402
from token_econ.synthetic import random_economics  # noqa: E402


def main() -> int:
    n = 20000
    exact_ok = break_ok = max_ok = 0
    naive_break_wrong = naive_max_wrong = break_cases = max_cases = 0
    rng = random.Random(1)
    start = time.perf_counter()
    for seed in range(n):
        e, f = random_economics(seed)
        profit, per = oracle_profit(f), oracle_cost_per_request(f)
        exact_ok += e.profit_per_user() == float(profit) and e.cost_per_request() == float(per)
        fixed = Fraction(rng.randint(1, 5_000_000), 100)
        want = None if profit <= 0 else math.ceil(fixed / profit)
        got = e.breakeven_users(float(fixed))
        break_ok += got == want
        if want is not None:
            break_cases += 1
            naive_profit = e.price_per_user - e.payment_fees() - e.other_cost_per_user - e.support_cost_per_user - e.requests_per_user * e.cost_per_request()
            naive_break_wrong += math.ceil(float(fixed) / naive_profit) != want if naive_profit > 0 else True
        target = Fraction(rng.randint(0, 95), 100)
        if per != 0:
            price = f["price_per_user"]
            budget = (price * (1 - target) - (price * get(f, "payment_fee_rate") + get(f, "payment_fee_fixed")) - get(f, "other_cost_per_user")
                      - get(f, "support_cost_per_user"))
            want_r = max(0, math.floor(budget / per))
            max_ok += e.max_requests_for_margin(float(target)) == want_r
            max_cases += 1
            naive_budget = e.price_per_user * (1 - float(target)) - e.payment_fees() - e.other_cost_per_user - e.support_cost_per_user
            naive_max_wrong += max(0, math.floor(naive_budget / e.cost_per_request())) != want_r
    elapsed = time.perf_counter() - start
    # random inputs almost never make the answer land exactly on a whole number, which is where floating point goes wrong. Build cases that do:
    # fixed cost = k x profit exactly, so the right answer is exactly k users.
    trap_total = trap_naive_wrong = trap_ok = 0
    rng = random.Random(9)
    for _ in range(20000):
        profit_cents, k = rng.randint(1, 5000), rng.randint(1, 500)
        e = UnitEconomics(profit_cents / 100, 0, 0, 0, 0, 0)
        fixed_cents = profit_cents * k
        trap_total += 1
        trap_ok += e.breakeven_users(fixed_cents / 100) == k
        trap_naive_wrong += math.ceil((fixed_cents / 100) / (profit_cents / 100)) != k
    print(f"float answers exact     {exact_ok}/{n} profit and cost per request equal the correctly rounded exact value")
    print(f"breakeven users         {break_ok}/{n} equal the exact ceiling; a plain float ceil is wrong in {naive_break_wrong} of {break_cases} profitable cases")
    print(f"max requests            {max_ok}/{max_cases} equal the exact floor; a plain float floor is wrong in {naive_max_wrong} of them")
    print(f"exact-multiple traps    {trap_ok}/{trap_total} correct; a plain float ceil is wrong in {trap_naive_wrong} ({trap_naive_wrong / trap_total:.1%}) of cases where the answer is a whole number")

    agree = total = 0
    rng = random.Random(4)
    for seed in range(1500):
        e, _ = random_economics(seed)
        if e.gross_margin() <= 0:
            continue
        trend = Trend(price_decline=rng.randint(0, 8) / 100, usage_growth=rng.randint(0, 15) / 100, token_growth=rng.randint(0, 5) / 100)
        target = max(0.0, e.gross_margin() - rng.randint(1, 30) / 100)
        slow = next((m for m in range(121) if economics_at(e, trend, m).gross_margin() < target), None)
        agree += squeeze_month(e, trend, target, horizon=120) == slow
        total += 1
    print(f"squeeze month           {agree}/{total} closed-form answers equal a month-by-month search of the full model")

    worst = 0.0
    for dist in [Lognormal(100, 0.5), Lognormal(100, 1.0), Lognormal(100, 1.5), Pareto(50, 1.3), Pareto(50, 2.0), Pareto(50, 3.0)]:
        for k in (10, 100, 1000):
            worst = max(worst, abs(sum(dist.points(k)) / k - dist.mean()) / dist.mean())
    print(f"population means        worst relative error {worst:.1e} across 18 distribution and size combinations")
    nd = NormalDist()
    draws = random.Random(3)
    sample = sorted(draws.lognormvariate(math.log(100), 1.0) for _ in range(400_000))
    simulated = sum(sample[-4000:]) / sum(sample)
    analytic = 1 - nd.cdf(nd.inv_cdf(0.99) - 1.0)
    print(f"top 1% share            lognormal sigma 1.0: library {concentration(Lognormal(100, 1.0).points(1000), 0.01):.4f}, analytic {analytic:.4f}, "
          f"400,000 random draws {simulated:.4f}; pareto alpha 2 exactly {Pareto(50, 2.0).top_share(0.01):.3f} = 0.01^(1-1/2)")

    worst = 0.0
    for churn, disc in [(0.02, 0.0), (0.05, 0.01), (0.1, 0.02), (0.01, 0.01), (0.3, 0.0), (0.005, 0.003)]:
        c = Cohort(12.5, churn, 40, disc)
        q = (1 - churn) / (1 + disc)
        worst = max(worst, abs(c.ltv() - sum(12.5 * q ** t for t in range(20000))) / c.ltv())
    print(f"cohort LTV              worst relative error {worst:.1e} against a 20,000-term sum")
    print(f"speed                   {n} random economics checked three ways in {elapsed:.1f}s on this machine")
    return 0 if exact_ok == n and break_ok == n and max_ok == max_cases and agree == total and trap_ok == trap_total else 1


if __name__ == "__main__":
    sys.exit(main())
