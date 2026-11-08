"""token.econ: unit economics for AI products. Margin per user, breakeven, and sensitivity."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

__version__ = "0.1.0"


@dataclass(frozen=True)
class UnitEconomics:
    price_per_user: float          # monthly revenue per user
    requests_per_user: float       # monthly requests per user
    input_tokens: float            # average input tokens per request
    output_tokens: float           # average output tokens per request
    input_price_per_1m: float
    output_price_per_1m: float
    other_cost_per_user: float = 0.0

    def cost_per_request(self) -> float:
        return (self.input_tokens * self.input_price_per_1m
                + self.output_tokens * self.output_price_per_1m) / 1_000_000

    def inference_cost(self) -> float:
        return self.requests_per_user * self.cost_per_request()

    def profit_per_user(self) -> float:
        return self.price_per_user - self.inference_cost() - self.other_cost_per_user

    def gross_margin(self) -> float:
        return self.profit_per_user() / self.price_per_user

    def breakeven_users(self, fixed_monthly_cost: float):
        """Users needed to cover fixed cost, or None if each user loses money."""
        profit = self.profit_per_user()
        return math.ceil(fixed_monthly_cost / profit) if profit > 0 else None

    def max_requests_for_margin(self, target_margin: float) -> int:
        """Most monthly requests per user that still hits the target gross margin."""
        budget = self.price_per_user * (1 - target_margin) - self.other_cost_per_user
        per_request = self.cost_per_request()
        return max(0, math.floor(budget / per_request)) if per_request > 0 else 0

    def with_(self, **changes) -> "UnitEconomics":
        return replace(self, **changes)


def sweep(base: UnitEconomics, field: str, values) -> list:
    """Gross margin as one input varies: [(value, margin), ...]."""
    return [(v, base.with_(**{field: v}).gross_margin()) for v in values]


def price_for_margin(base: UnitEconomics, target_margin: float) -> float:
    """Lowest monthly price that reaches the target margin at current usage."""
    return (base.inference_cost() + base.other_cost_per_user) / (1 - target_margin)
