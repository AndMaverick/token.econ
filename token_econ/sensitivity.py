"""Which assumption moves the margin most? A tornado ranks inputs by how far a plus or minus swing moves it."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .economics import UnitEconomics

DEFAULT_FIELDS = ("price_per_user", "requests_per_user", "input_tokens", "output_tokens", "input_price_per_1m", "output_price_per_1m",
                  "cache_hit_rate", "reasoning_tokens", "retry_rate", "other_cost_per_user", "support_cost_per_user")


@dataclass(frozen=True)
class Swing:
    field: str
    base_value: float
    low_value: float
    high_value: float
    margin_at_low: float
    margin_at_high: float

    @property
    def span(self) -> float:
        """How far the margin moves between the low and high settings, in margin points (0.05 is five points)."""
        return abs(self.margin_at_high - self.margin_at_low)


def tornado(base: UnitEconomics, pct: float = 0.2, fields: Sequence[str] = DEFAULT_FIELDS) -> list[Swing]:
    """Vary each input by plus and minus ``pct`` of its value (rates are kept inside 0 to 1) and rank by margin swing, largest first.
    Inputs that are zero are skipped, since a percentage of zero is zero."""
    if not 0 < pct < 1:
        raise ValueError("pct must be between 0 and 1")
    rows = []
    for name in fields:
        value = getattr(base, name)
        if value is None or value == 0:
            continue
        low, high = value * (1 - pct), value * (1 + pct)
        if name in ("cache_hit_rate", "payment_fee_rate", "cache_write_rate"):
            high = min(high, 1.0)
        rows.append(Swing(name, value, low, high, base.with_(**{name: low}).gross_margin(), base.with_(**{name: high}).gross_margin()))
    return sorted(rows, key=lambda s: (-s.span, s.field))


def grid(base: UnitEconomics, row_field: str, row_values: Sequence[float], col_field: str, col_values: Sequence[float]) -> list[list[float]]:
    """Gross margin over two inputs at once: ``result[i][j]`` is the margin at ``row_values[i]`` and ``col_values[j]``."""
    return [[base.with_(**{row_field: r, col_field: c}).gross_margin() for c in col_values] for r in row_values]
