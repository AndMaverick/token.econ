"""Plan design: what a pricing plan earns across a realistic population of users, not at the average one."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .economics import UnitEconomics, exact
from .population import Population, concentration
from .pricing import _check


@dataclass(frozen=True)
class Plan:
    """A monthly price with optional included requests, an overage rate and a hard cap.

    ``Plan("flat", 20)``                                   unlimited use for 20
    ``Plan("metered", 0, included_requests=0, overage_per_request=0.02)``   pure pay per request
    ``Plan("hybrid", 10, included_requests=100, overage_per_request=0.03)``  10 covers 100, then 3 cents each
    ``Plan("capped", 20, hard_cap=250)``                   flat, but requests stop at 250 (heavier users are held at the cap)

    Users who hit a hard cap are assumed to stop there. They do not churn and they do not buy more: if throttling drives them away, that
    effect is not in these numbers."""

    name: str
    price: float
    included_requests: float | None = None
    overage_per_request: float = 0.0
    hard_cap: float | None = None

    def __post_init__(self) -> None:
        _check("price", self.price)
        _check("overage_per_request", self.overage_per_request)
        if self.included_requests is not None:
            _check("included_requests", self.included_requests)
        if self.hard_cap is not None:
            _check("hard_cap", self.hard_cap)

    def served(self, requests: float) -> float:
        return requests if self.hard_cap is None else min(requests, self.hard_cap)

    def revenue(self, requests: float) -> float:
        served = self.served(requests)
        extra = 0.0 if self.included_requests is None else max(0.0, served - self.included_requests)
        return self.price + extra * self.overage_per_request


@dataclass(frozen=True)
class PlanResult:
    plan: str
    revenue_per_user: float
    cost_per_user: float
    profit_per_user: float
    margin: float | None           # None when the plan earns nothing
    loss_share: float              # fraction of users who cost more than they pay
    top_cost_share: float          # share of total inference cost from the heaviest 1 percent of users
    p50_profit: float
    p99_profit: float              # profit of the user at the 99th usage percentile (negative means whales lose you money)

    def to_dict(self) -> dict[str, float | str | None]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def simulate(econ: UnitEconomics, plan: Plan, population: Population, *, users: int = 1000) -> PlanResult:
    """Run one plan across a population. The economics supply token costs, fixed per-user costs and the payment fee rate; the plan
    supplies price and limits, and the population supplies how many requests each user makes (``econ.requests_per_user`` is ignored)."""
    pts = sorted(population.points(users))
    per_request = float(econ._cost_per_request_t(exact))
    fixed = float(econ._fixed_cost_t(exact))
    fee_rate, fee_fixed = econ.payment_fee_rate, econ.payment_fee_fixed
    revenue: list[float] = []
    inference: list[float] = []
    profit: list[float] = []
    for r in pts:
        rev = plan.revenue(r)
        inf = plan.served(r) * per_request
        fee = rev * fee_rate + (fee_fixed if rev > 0 else 0.0)
        revenue.append(rev)
        inference.append(inf)
        profit.append(rev - fee - inf - fixed)
    n = len(pts)
    total_rev, total_profit = math.fsum(revenue), math.fsum(profit)
    losers = sum(1 for x in profit if x < 0)
    return PlanResult(plan.name, total_rev / n, (total_rev - total_profit) / n, total_profit / n,
                      None if total_rev == 0 else total_profit / total_rev, losers / n, concentration(inference, 0.01),
                      profit[n // 2], profit[min(n - 1, int(n * 0.99))])


def compare(econ: UnitEconomics, plans: Sequence[Plan], population: Population, *, users: int = 1000) -> list[PlanResult]:
    """Every plan on the same population, best profit per user first."""
    return sorted((simulate(econ, p, population, users=users) for p in plans), key=lambda r: -r.profit_per_user)


def solve_cap(econ: UnitEconomics, plan: Plan, population: Population, target_margin: float, *, users: int = 1000) -> float | None:
    """The highest hard cap (requests per month) at which the plan reaches the target margin. ``None`` if it cannot reach it even with a
    cap of zero, ``math.inf`` if it already reaches it with no cap. On a flat plan cost falls as the cap tightens and revenue is
    unchanged, so margin is monotone in the cap and this is a bisection; with an overage rate that monotonicity can fail, and the
    answer is then only a boundary the search found, so check it with ``simulate``."""
    def margin(cap: float) -> float:
        m = simulate(econ, Plan(plan.name, plan.price, plan.included_requests, plan.overage_per_request, cap), population, users=users).margin
        return -math.inf if m is None else m

    pts = population.points(users)
    hi = max(pts) * 1.0001 + 1
    if margin(0.0) < target_margin:
        return None
    if margin(hi) >= target_margin:
        return math.inf
    return _bisect_max(lambda c: margin(c) >= target_margin, 0.0, hi)


def _bisect_max(ok: Callable[[float], bool], lo: float, hi: float, iters: int = 80) -> float:
    """Largest x in [lo, hi] where ``ok`` holds, given ok(lo) is true and ok(hi) is false and ok flips once."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


@dataclass(frozen=True)
class FreeTier:
    """Free users cost inference and bring no revenue. ``free_per_paid`` is how many free users there are for each paying one at the
    point you are modelling, and ``free_requests`` how many requests each makes a month."""

    free_per_paid: float
    free_requests: float

    def __post_init__(self) -> None:
        _check("free_per_paid", self.free_per_paid)
        _check("free_requests", self.free_requests)

    def cost_per_free_user(self, econ: UnitEconomics) -> float:
        return float(exact(self.free_requests) * econ._cost_per_request_t(exact))

    def blended_profit_per_paid_user(self, econ: UnitEconomics) -> float:
        """Profit on one paying user after paying for the free users that come with them."""
        return econ.profit_per_user() - self.free_per_paid * self.cost_per_free_user(econ)

    def required_conversion(self, econ: UnitEconomics) -> float | None:
        """Smallest share of all users that must pay for the free tier to cost nothing overall: ``cost_free / (profit_paid + cost_free)``,
        where profit_paid is before free cost. ``None`` if a paying user earns nothing."""
        profit = econ._profit_t(exact)
        if profit <= 0:
            return None
        cost_free = exact(self.free_requests) * econ._cost_per_request_t(exact)
        return float(cost_free / (profit + cost_free))
