"""Model cascades: try a cheap model first and escalate to a stronger one only when needed. The arithmetic of when that pays."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .pricing import _check


@dataclass(frozen=True)
class Tier:
    """One model in a cascade. ``cost`` is dollars per attempt. ``escalate`` is the fraction of the requests that reach this tier that
    it hands on to the next (its failure or low-confidence rate). The last tier's ``escalate`` must be 0."""

    name: str
    cost: float
    escalate: float = 0.0

    def __post_init__(self) -> None:
        _check("cost", self.cost)
        _check("escalate", self.escalate, high=1.0)


def expected_cost(tiers: Sequence[Tier]) -> float:
    """Expected dollars per request: every request pays tier 1, the fraction that escalates also pays tier 2, and so on."""
    if not tiers:
        raise ValueError("a cascade needs at least one tier")
    if tiers[-1].escalate != 0:
        raise ValueError("the last tier cannot escalate")
    total, reach = 0.0, 1.0
    for t in tiers:
        total += reach * t.cost
        reach *= t.escalate
    return total


def reach(tiers: Sequence[Tier]) -> list[float]:
    """The fraction of all requests that reach each tier."""
    out, share = [], 1.0
    for t in tiers:
        out.append(share)
        share *= t.escalate
    return out


def max_escalation_to_beat(cheap_cost: float, strong_cost: float) -> float:
    """A two-tier cascade is cheaper than always using the strong model exactly when the escalation rate is below
    ``1 - cheap_cost / strong_cost``. This is that threshold. At or above it, the cheap model's wasted attempts cost more than they save."""
    _check("cheap_cost", cheap_cost)
    if strong_cost <= 0:
        raise ValueError("strong_cost must be positive")
    return max(0.0, 1 - cheap_cost / strong_cost)


def savings(tiers: Sequence[Tier]) -> float:
    """Fraction saved against sending everything to the last tier."""
    return 1 - expected_cost(tiers) / tiers[-1].cost if tiers[-1].cost > 0 else 0.0
