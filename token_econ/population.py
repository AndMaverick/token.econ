"""Usage is not average. A few heavy users carry most of the cost, and a price set on the mean loses money on them.

Each distribution gives ``points(n)``: n equal-weight representative users whose usage is the exact conditional mean of their slice
of the population, so the average of the points equals the distribution mean for any n."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

_N = NormalDist()


def _check_n(n: int) -> None:
    if n < 1 or int(n) != n:
        raise ValueError(f"n must be a whole number of at least 1, got {n}")


@dataclass(frozen=True)
class Fixed:
    """Every user makes the same number of requests."""

    value: float

    def mean(self) -> float:
        return self.value

    def points(self, n: int = 1000) -> list[float]:
        _check_n(n)
        return [self.value] * n


@dataclass(frozen=True)
class Lognormal:
    """Requests per user follow a lognormal: ``median`` is the typical user and ``sigma`` how spread out usage is. Around 0.5 is mild,
    1 to 1.5 is a heavy tail. The mean is ``median * exp(sigma^2 / 2)``, always above the median."""

    median: float
    sigma: float

    def __post_init__(self) -> None:
        if self.median <= 0 or self.sigma < 0:
            raise ValueError("lognormal needs median > 0 and sigma >= 0")

    def mean(self) -> float:
        return self.median * math.exp(self.sigma ** 2 / 2)

    def quantile(self, p: float) -> float:
        return self.median * math.exp(self.sigma * _N.inv_cdf(p))

    def points(self, n: int = 1000) -> list[float]:
        _check_n(n)
        if self.sigma == 0:
            return [self.median] * n
        mean = self.mean()
        edges = [-math.inf] + [_N.inv_cdf(i / n) for i in range(1, n)] + [math.inf]
        # E[X; X <= q(p)] = mean * Phi(z - sigma): the share of total usage below the slice edge z (Phi of -inf is 0, of +inf is 1)
        share = [_N.cdf(e - self.sigma) for e in edges]
        return [mean * (share[i + 1] - share[i]) * n for i in range(n)]


@dataclass(frozen=True)
class Pareto:
    """A power-law tail: most users make about ``minimum`` requests and a few make far more. ``alpha`` near 1 is extreme (a handful
    of users dominate); the mean exists only for alpha above 1."""

    minimum: float
    alpha: float

    def __post_init__(self) -> None:
        if self.minimum <= 0 or self.alpha <= 1:
            raise ValueError("pareto needs minimum > 0 and alpha > 1 (the mean is infinite otherwise)")

    def mean(self) -> float:
        return self.alpha * self.minimum / (self.alpha - 1)

    def quantile(self, p: float) -> float:
        return float(self.minimum * (1 - p) ** (-1 / self.alpha))

    def top_share(self, top: float) -> float:
        """Exact share of total usage from the top ``top`` fraction of users: ``top ** (1 - 1/alpha)``."""
        return float(top ** (1 - 1 / self.alpha))

    def points(self, n: int = 1000) -> list[float]:
        _check_n(n)
        k = 1 - 1 / self.alpha
        mean = self.mean()
        # share of usage from users below quantile p is 1 - (1 - p)^k
        return [mean * ((1 - (1 - (i + 1) / n) ** k) - (1 - (1 - i / n) ** k)) * n for i in range(n)]


@dataclass(frozen=True)
class Empirical:
    """Your own measured requests per user. Each value is one user."""

    values: tuple[float, ...]

    def __init__(self, values: Sequence[float]) -> None:
        if not values or any(v < 0 for v in values):
            raise ValueError("empirical needs at least one value, none negative")
        object.__setattr__(self, "values", tuple(sorted(float(v) for v in values)))

    def mean(self) -> float:
        return sum(self.values) / len(self.values)

    def points(self, n: int | None = None) -> list[float]:
        return list(self.values)


Population = Fixed | Lognormal | Pareto | Empirical


def concentration(points: Sequence[float], top: float = 0.01) -> float:
    """Share of the total in the heaviest ``top`` fraction of users (at least one point)."""
    if not 0 < top <= 1:
        raise ValueError("top must be in (0, 1]")
    ordered = sorted(points, reverse=True)
    total = sum(ordered)
    if total == 0:
        return 0.0
    k = max(1, round(len(ordered) * top))
    return sum(ordered[:k]) / total


def from_dict(data: dict[str, float | str]) -> Population:
    kind = str(data.get("kind", ""))
    if kind == "fixed":
        return Fixed(float(data["value"]))
    if kind == "lognormal":
        return Lognormal(float(data["median"]), float(data["sigma"]))
    if kind == "pareto":
        return Pareto(float(data["minimum"]), float(data["alpha"]))
    raise ValueError(f"population kind must be fixed, lognormal or pareto, got {kind!r}")
