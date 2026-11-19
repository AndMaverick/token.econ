"""Unit economics of one paying user per month, and the solvers that answer "how much", "how many" and "what price"."""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields, replace
from fractions import Fraction
from typing import Any

from .pricing import TokenPrice, Usage, _check


def exact(x: float) -> Fraction:
    """The decimal you wrote, as an exact fraction: 0.1 is one tenth, not the nearest binary float. Integer answers (users, requests)
    are computed on these so they can never be off by one from floating point noise."""
    return Fraction(repr(float(x)))


@dataclass(frozen=True)
class UnitEconomics:
    """One monthly paying user.

    price_per_user          monthly revenue
    requests_per_user       monthly requests
    input_tokens            average input tokens per request (all of it, cached or not)
    output_tokens           average visible output tokens per request
    input_price_per_1m      dollars per million input tokens
    output_price_per_1m     dollars per million output tokens
    other_cost_per_user     any other monthly cost per user (storage, vector search, tools)

    The remaining fields default to "not applicable":
    cached_input_price_per_1m, cache_write_price_per_1m   prompt caching prices
    cache_hit_rate, cache_write_rate                      share of input tokens read from, written to, the cache
    reasoning_tokens        hidden reasoning tokens per request, billed as output
    retry_rate              extra billed requests from retries and failed attempts, as a fraction (0.05 is 5 percent more)
    infra_overhead_rate     gateway, logging, evals and orchestration as a fraction of inference cost
    support_cost_per_user   monthly support and success cost per user
    payment_fee_rate / payment_fee_fixed   processor fee as a fraction of price plus a fixed amount per user per month
    """

    price_per_user: float
    requests_per_user: float
    input_tokens: float
    output_tokens: float
    input_price_per_1m: float
    output_price_per_1m: float
    other_cost_per_user: float = 0.0
    cached_input_price_per_1m: float | None = None
    cache_write_price_per_1m: float | None = None
    cache_hit_rate: float = 0.0
    cache_write_rate: float = 0.0
    reasoning_tokens: float = 0.0
    retry_rate: float = 0.0
    infra_overhead_rate: float = 0.0
    support_cost_per_user: float = 0.0
    payment_fee_rate: float = 0.0
    payment_fee_fixed: float = 0.0

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                _check(f.name, value)
        _check("cache_hit_rate", self.cache_hit_rate, high=1.0)
        _check("cache_write_rate", self.cache_write_rate, high=1.0)
        _check("payment_fee_rate", self.payment_fee_rate, high=1.0)
        if self.cache_hit_rate + self.cache_write_rate > 1.0 + 1e-12:
            raise ValueError("cache_hit_rate plus cache_write_rate cannot exceed 1")

    # the arithmetic is written once, over a number type, so the float answers and the exact (Fraction) answers cannot drift apart ----
    def _cost_per_request_t(self, c: Callable[[Any], Any]) -> Any:
        pin, pout = c(self.input_price_per_1m), c(self.output_price_per_1m)
        pread = pin if self.cached_input_price_per_1m is None else c(self.cached_input_price_per_1m)
        pwrite = pin if self.cache_write_price_per_1m is None else c(self.cache_write_price_per_1m)
        tokens_in = c(self.input_tokens)
        hit, write = c(self.cache_hit_rate), c(self.cache_write_rate)
        base = (tokens_in * (1 - hit - write) * pin + tokens_in * hit * pread + tokens_in * write * pwrite
                + (c(self.output_tokens) + c(self.reasoning_tokens)) * pout) / 1_000_000
        return base * (1 + c(self.retry_rate)) * (1 + c(self.infra_overhead_rate))

    def _fixed_cost_t(self, c: Callable[[Any], Any]) -> Any:
        """Monthly costs that do not depend on requests, other than the payment fee."""
        return c(self.other_cost_per_user) + c(self.support_cost_per_user)

    def _fee_t(self, c: Callable[[Any], Any]) -> Any:
        return c(self.price_per_user) * c(self.payment_fee_rate) + c(self.payment_fee_fixed)

    def _profit_t(self, c: Callable[[Any], Any]) -> Any:
        return (c(self.price_per_user) - self._fee_t(c) - self._fixed_cost_t(c)
                - c(self.requests_per_user) * self._cost_per_request_t(c))

    # float answers ---------------------------------------------------------------------------------------------------------------
    def price(self) -> TokenPrice:
        return TokenPrice(self.input_price_per_1m, self.output_price_per_1m, self.cached_input_price_per_1m, self.cache_write_price_per_1m)

    def usage(self) -> Usage:
        return Usage(self.input_tokens, self.output_tokens, self.input_tokens * self.cache_hit_rate, self.input_tokens * self.cache_write_rate,
                     self.reasoning_tokens)

    def cost_per_request(self) -> float:
        """Dollars per request including retries and infrastructure overhead."""
        return float(self._cost_per_request_t(exact))

    def inference_cost(self) -> float:
        return float(exact(self.requests_per_user) * self._cost_per_request_t(exact))

    def payment_fees(self) -> float:
        return float(self._fee_t(exact))

    def profit_per_user(self) -> float:
        return float(self._profit_t(exact))

    def gross_margin(self) -> float:
        """Profit per user over price. Needs a positive price (use ``plans`` for free users)."""
        if self.price_per_user <= 0:
            raise ValueError("gross margin needs a positive price_per_user; model free users with a FreeTier")
        return float(self._profit_t(exact) / exact(self.price_per_user))

    # integer answers, exact ------------------------------------------------------------------------------------------------------
    def breakeven_users(self, fixed_monthly_cost: float) -> int | None:
        """Users needed so total profit covers a fixed monthly cost, or ``None`` if each user loses money (so no number of users does)."""
        _check("fixed_monthly_cost", fixed_monthly_cost)
        profit = self._profit_t(exact)
        if profit <= 0:
            return None
        return int(math.ceil(exact(fixed_monthly_cost) / profit))

    def max_requests_for_margin(self, target_margin: float) -> int | None:
        """The most monthly requests per user that still reach the target gross margin: the largest whole number, computed exactly.
        0 if even no requests miss the target; ``None`` if requests cost nothing, so there is no limit."""
        _check("target_margin", target_margin, low=-math.inf, high=1.0)
        per_request = self._cost_per_request_t(exact)
        price = exact(self.price_per_user)
        budget = price * (1 - exact(target_margin)) - self._fee_t(exact) - self._fixed_cost_t(exact)
        if per_request == 0:
            return None if budget >= 0 else 0
        return max(0, int(math.floor(budget / per_request)))

    # closed-form solvers ---------------------------------------------------------------------------------------------------------
    def price_for_margin(self, target_margin: float) -> float:
        """Lowest monthly price that reaches the target margin at current usage, allowing for the payment fee. Raises ValueError when
        the target cannot be reached at any price (the fee rate plus the target margin is 100 percent or more)."""
        room = 1 - exact(self.payment_fee_rate) - exact(target_margin)
        if room <= 0:
            raise ValueError(f"a {target_margin:.0%} margin is unreachable at any price with a {self.payment_fee_rate:.1%} payment fee")
        costs = exact(self.payment_fee_fixed) + self._fixed_cost_t(exact) + exact(self.requests_per_user) * self._cost_per_request_t(exact)
        return float(costs / room)

    def required_cache_hit_rate(self, target_margin: float) -> float | None:
        """Smallest cache hit rate that reaches the target margin, or ``None`` if even a full hit rate falls short. Cost is linear
        in the hit rate, so this is exact."""
        def margin_at(hit: float) -> float:
            return self.with_(cache_hit_rate=hit, cache_write_rate=min(self.cache_write_rate, 1 - hit)).gross_margin()

        return _solve_linear(margin_at, target_margin, 0.0, 1.0)

    def max_output_tokens_for_margin(self, target_margin: float, *, ceiling: float = 1_000_000.0) -> float | None:
        """Most average output tokens per request that still reach the target margin (margin is linear in output tokens), or ``None`` if
        even zero output misses it."""
        return _solve_linear(lambda t: self.with_(output_tokens=t).gross_margin(), target_margin, 0.0, ceiling)

    def with_(self, **changes: Any) -> UnitEconomics:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """The inputs that differ from their defaults, plus the required ones."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name in _REQUIRED or getattr(self, f.name) != f.default}


_REQUIRED = {"price_per_user", "requests_per_user", "input_tokens", "output_tokens", "input_price_per_1m", "output_price_per_1m"}


def _solve_linear(margin_at: Callable[[float], float], target: float, lo: float, hi: float) -> float | None:
    """Margin is linear in the variable between lo and hi. Find where it equals the target using two evaluations. When margin falls with
    the variable (output tokens) the answer is the largest value that still meets the target; when it rises (cache hit rate) the smallest."""
    m_lo, m_hi = margin_at(lo), margin_at(hi)
    if m_lo == m_hi:
        return lo if m_lo >= target else None
    x = lo + (target - m_lo) * (hi - lo) / (m_hi - m_lo)
    if m_hi < m_lo:  # margin falls as the variable grows: want the largest x meeting the target
        if m_lo < target:
            return None
        return min(x, hi)
    if m_hi < target:  # margin rises: want the smallest x meeting the target
        return None
    return max(x, lo)


def sweep(base: UnitEconomics, field: str, values: Iterable[float]) -> list[tuple[float, float]]:
    """Gross margin as one input varies: ``[(value, margin), ...]``."""
    return [(v, base.with_(**{field: v}).gross_margin()) for v in values]


def price_for_margin(base: UnitEconomics, target_margin: float) -> float:
    """See ``UnitEconomics.price_for_margin``."""
    return base.price_for_margin(target_margin)
