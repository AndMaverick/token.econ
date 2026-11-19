"""Seeded random inputs for property tests and measurement. Values are short decimals, like numbers a person types."""
from __future__ import annotations

import random
from decimal import Decimal
from fractions import Fraction

from .economics import UnitEconomics

REQUIRED = ("price_per_user", "requests_per_user", "input_tokens", "output_tokens", "input_price_per_1m", "output_price_per_1m")


def _dec(rng: random.Random, low: int, high: int, places: int) -> str:
    return str(Decimal(rng.randint(low * 10 ** places, high * 10 ** places)).scaleb(-places))


def random_economics(seed: int) -> tuple[UnitEconomics, dict[str, Fraction]]:
    """A random UnitEconomics and, beside it, every field as an exact Fraction read from the decimal string it was made from. Tests use
    the fractions to compute answers a second way, with no floating point at all."""
    rng = random.Random(seed)
    text: dict[str, str] = {
        "price_per_user": _dec(rng, 1, 200, 2), "requests_per_user": _dec(rng, 0, 3000, 0), "input_tokens": _dec(rng, 0, 20000, 0),
        "output_tokens": _dec(rng, 0, 4000, 0), "input_price_per_1m": _dec(rng, 0, 20, 2), "output_price_per_1m": _dec(rng, 0, 80, 2),
    }
    if rng.random() < 0.5:
        text["other_cost_per_user"] = _dec(rng, 0, 5, 2)
    if rng.random() < 0.5:
        text["support_cost_per_user"] = _dec(rng, 0, 3, 2)
    if rng.random() < 0.5:
        text["cached_input_price_per_1m"] = _dec(rng, 0, 5, 2)
        text["cache_hit_rate"] = str(Decimal(rng.randint(0, 80)) / 100)
        if rng.random() < 0.5:
            text["cache_write_price_per_1m"] = _dec(rng, 0, 25, 2)
            text["cache_write_rate"] = str(Decimal(rng.randint(0, 15)) / 100)
    if rng.random() < 0.4:
        text["reasoning_tokens"] = _dec(rng, 0, 3000, 0)
    if rng.random() < 0.4:
        text["retry_rate"] = str(Decimal(rng.randint(0, 20)) / 100)
    if rng.random() < 0.4:
        text["infra_overhead_rate"] = str(Decimal(rng.randint(0, 25)) / 100)
    if rng.random() < 0.4:
        text["payment_fee_rate"] = str(Decimal(rng.randint(0, 6)) / 100)
        text["payment_fee_fixed"] = _dec(rng, 0, 1, 2)
    econ = UnitEconomics(**{k: float(v) for k, v in text.items()})
    return econ, {k: Fraction(v) for k, v in text.items()}
