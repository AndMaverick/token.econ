"""The squeeze: token prices fall, but usage per user and tokens per request often grow faster. This projects margin month by month and
finds the month it crosses your floor."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .economics import UnitEconomics, exact
from .pricing import _check


@dataclass(frozen=True)
class Trend:
    """Monthly rates of change as fractions: ``price_decline`` 0.03 means token prices fall 3 percent a month, ``usage_growth`` 0.05
    means requests per user grow 5 percent a month, ``token_growth`` 0.02 means tokens per request grow 2 percent a month (longer
    contexts, more agent steps). Price per user is held fixed unless ``price_growth`` is set."""

    price_decline: float = 0.0
    usage_growth: float = 0.0
    token_growth: float = 0.0
    price_growth: float = 0.0

    def __post_init__(self) -> None:
        _check("price_decline", self.price_decline, high=1.0)
        _check("usage_growth", self.usage_growth, low=-1.0)
        _check("token_growth", self.token_growth, low=-1.0)
        _check("price_growth", self.price_growth, low=-1.0)

    @property
    def cost_factor(self) -> float:
        """What inference cost per user is multiplied by each month. Above 1 means cost is outrunning price declines."""
        return (1 + self.usage_growth) * (1 + self.token_growth) * (1 - self.price_decline)

    @property
    def squeezing(self) -> bool:
        return self.cost_factor > 1

    def critical_price_decline(self) -> float:
        """The monthly token-price decline that exactly cancels the growth in usage and tokens. Faster than this and cost per user falls."""
        return 1 - 1 / ((1 + self.usage_growth) * (1 + self.token_growth))


@dataclass(frozen=True)
class MonthPoint:
    month: int
    inference_cost: float
    profit: float
    margin: float


def economics_at(base: UnitEconomics, trend: Trend, month: int) -> UnitEconomics:
    """The economics ``month`` months from now: prices, usage and tokens scaled by the trend, compounding."""
    return base.with_(
        price_per_user=base.price_per_user * (1 + trend.price_growth) ** month,
        requests_per_user=base.requests_per_user * (1 + trend.usage_growth) ** month,
        input_tokens=base.input_tokens * (1 + trend.token_growth) ** month,
        output_tokens=base.output_tokens * (1 + trend.token_growth) ** month,
        reasoning_tokens=base.reasoning_tokens * (1 + trend.token_growth) ** month,
        input_price_per_1m=base.input_price_per_1m * (1 - trend.price_decline) ** month,
        output_price_per_1m=base.output_price_per_1m * (1 - trend.price_decline) ** month,
        cached_input_price_per_1m=None if base.cached_input_price_per_1m is None else base.cached_input_price_per_1m * (1 - trend.price_decline) ** month,
        cache_write_price_per_1m=None if base.cache_write_price_per_1m is None else base.cache_write_price_per_1m * (1 - trend.price_decline) ** month,
    )


def project(base: UnitEconomics, trend: Trend, months: int) -> list[MonthPoint]:
    """Month 0 (today) through ``months``, evaluated with the full model each month."""
    if months < 0 or int(months) != months:
        raise ValueError("months must be a whole number of at least 0")
    out = []
    for m in range(months + 1):
        e = economics_at(base, trend, m)
        out.append(MonthPoint(m, e.inference_cost(), e.profit_per_user(), e.gross_margin()))
    return out


def squeeze_month(base: UnitEconomics, trend: Trend, target_margin: float, *, horizon: int = 600) -> int | None:
    """The first month in which gross margin is below the target, or ``None`` if it never gets there within ``horizon`` months.
    0 means it is already below. Found in closed form from the compounding cost, then confirmed (and corrected for floating point) by
    evaluating the full model at the neighbouring months."""
    def below(month: int) -> bool:
        return economics_at(base, trend, month).gross_margin() < target_margin

    if below(0):
        return 0
    inference = float(base._cost_per_request_t(exact) * exact(base.requests_per_user))
    fixed_and_fees = base.payment_fees() + float(base._fixed_cost_t(exact))
    budget = base.price_per_user * (1 - target_margin) - fixed_and_fees
    k = trend.cost_factor
    guess: int
    if inference > 0 and k > 1 and budget > 0 and trend.price_growth == 0:
        guess = max(1, math.floor(math.log(budget / inference) / math.log(k)) + 1)
        for m in range(max(1, guess - 2), guess + 3):  # confirm in the neighbourhood
            if below(m) and not below(m - 1):
                return m if m <= horizon else None
    for m in range(1, horizon + 1):  # fall back to the plain search (also handles price_growth, where there is no simple closed form)
        if below(m):
            return m
    return None
