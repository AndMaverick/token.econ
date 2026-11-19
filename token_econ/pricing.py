"""Token prices and what a request costs. Prices are always supplied by you: this library ships none, because a price table that
is not current is worse than none."""
from __future__ import annotations

from dataclasses import dataclass


def _check(name: str, value: float, *, low: float = 0.0, high: float | None = None) -> None:
    if value != value or value < low or (high is not None and value > high):
        bound = f"between {low} and {high}" if high is not None else f"at least {low}"
        raise ValueError(f"{name} must be {bound}, got {value}")


@dataclass(frozen=True)
class Usage:
    """Tokens in one request. ``cached_input_tokens`` and ``cache_write_tokens`` are parts of ``input_tokens``, not extra.
    ``reasoning_tokens`` are billed as output."""

    input_tokens: float = 0.0
    output_tokens: float = 0.0
    cached_input_tokens: float = 0.0
    cache_write_tokens: float = 0.0
    reasoning_tokens: float = 0.0

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "cached_input_tokens", "cache_write_tokens", "reasoning_tokens"):
            _check(name, getattr(self, name))
        if self.cached_input_tokens + self.cache_write_tokens > self.input_tokens * (1 + 1e-12):
            raise ValueError("cached_input_tokens plus cache_write_tokens cannot exceed input_tokens")


@dataclass(frozen=True)
class TokenPrice:
    """Dollars per million tokens. ``cached_input_per_1m`` is the price of input read from a prompt cache and ``cache_write_per_1m`` of
    input written to one; each defaults to the plain input price, meaning no caching discount or premium. ``batch_discount`` is the
    fraction taken off when the work runs as an asynchronous batch."""

    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: float | None = None
    cache_write_per_1m: float | None = None
    batch_discount: float = 0.0

    def __post_init__(self) -> None:
        _check("input_per_1m", self.input_per_1m)
        _check("output_per_1m", self.output_per_1m)
        if self.cached_input_per_1m is not None:
            _check("cached_input_per_1m", self.cached_input_per_1m)
        if self.cache_write_per_1m is not None:
            _check("cache_write_per_1m", self.cache_write_per_1m)
        _check("batch_discount", self.batch_discount, high=1.0)

    @property
    def cached_read(self) -> float:
        return self.input_per_1m if self.cached_input_per_1m is None else self.cached_input_per_1m

    @property
    def cache_write(self) -> float:
        return self.input_per_1m if self.cache_write_per_1m is None else self.cache_write_per_1m

    def cost(self, usage: Usage, *, batch: bool = False) -> float:
        fresh = usage.input_tokens - usage.cached_input_tokens - usage.cache_write_tokens
        total = (max(fresh, 0.0) * self.input_per_1m + usage.cached_input_tokens * self.cached_read
                 + usage.cache_write_tokens * self.cache_write + (usage.output_tokens + usage.reasoning_tokens) * self.output_per_1m) / 1_000_000
        return total * (1 - self.batch_discount) if batch else total
