"""Why agents blow budgets: every step resends the growing context, so input tokens grow with the square of the step count."""
from __future__ import annotations

from dataclasses import dataclass

from .pricing import TokenPrice, Usage, _check


@dataclass(frozen=True)
class AgentRun:
    """One agent task. Step k (counting from 0) sends ``base_context_tokens + k * added_tokens_per_step`` input tokens, because the
    model's output and the tool results from each earlier step are appended to the context. ``output_tokens_per_step`` is what the model
    writes each step (it is part of ``added_tokens_per_step`` in the next step's input).

    With ``cached=True`` and a cached price in the TokenPrice, the context from the previous step is read from the cache and only the
    newly added tokens are billed as fresh input (and as cache writes, if a write price is set)."""

    steps: int
    base_context_tokens: float
    added_tokens_per_step: float
    output_tokens_per_step: float
    cached: bool = False

    def __post_init__(self) -> None:
        if self.steps < 0 or int(self.steps) != self.steps:
            raise ValueError(f"steps must be a whole number of at least 0, got {self.steps}")
        _check("base_context_tokens", self.base_context_tokens)
        _check("added_tokens_per_step", self.added_tokens_per_step)
        _check("output_tokens_per_step", self.output_tokens_per_step)

    def input_tokens(self) -> float:
        """Total input tokens across all steps, without caching: ``n * base + added * n * (n - 1) / 2``. Quadratic in steps."""
        n = self.steps
        return n * self.base_context_tokens + self.added_tokens_per_step * n * (n - 1) / 2

    def output_tokens(self) -> float:
        return self.steps * self.output_tokens_per_step

    def step_usage(self, k: int) -> Usage:
        context = self.base_context_tokens + k * self.added_tokens_per_step
        if not self.cached or k == 0:
            write = context if self.cached else 0.0  # the first step writes the whole prefix
            return Usage(context, self.output_tokens_per_step, 0.0, write)
        previous = context - self.added_tokens_per_step
        return Usage(context, self.output_tokens_per_step, previous, self.added_tokens_per_step)

    def cost(self, price: TokenPrice) -> float:
        return sum(price.cost(self.step_usage(k)) for k in range(self.steps))

    def with_(self, **changes: float | int | bool) -> AgentRun:
        from dataclasses import replace
        return replace(self, **changes)  # type: ignore[arg-type]


def cost_curve(run: AgentRun, price: TokenPrice, steps: list[int]) -> list[tuple[int, float]]:
    """Cost of the same task at several step counts: the shape shows the curve bending upward."""
    return [(n, run.with_(steps=n).cost(price)) for n in steps]


def max_steps_for_budget(run: AgentRun, price: TokenPrice, budget: float) -> int:
    """Most steps a task can take within a dollar budget. Cost grows with every step, so this is a binary search for the boundary."""
    _check("budget", budget)
    if run.with_(steps=1).cost(price) > budget:
        return 0
    lo, hi = 1, 2
    while run.with_(steps=hi).cost(price) <= budget:
        lo, hi = hi, hi * 2
        if hi > 1_000_000:
            raise ValueError("budget allows more than a million steps; the cost per step may be zero")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if run.with_(steps=mid).cost(price) <= budget:
            lo = mid
        else:
            hi = mid
    return lo
