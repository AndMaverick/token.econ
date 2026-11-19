"""token.econ: unit economics for AI products. Exact margin and breakeven, caching, agent loops, usage distributions and plan design,
the price-versus-usage squeeze, cohort LTV, model cascades and sensitivity. You supply the prices; nothing here is a price list."""
from __future__ import annotations

__version__ = "0.2.0"

from .agent import AgentRun, cost_curve, max_steps_for_budget  # noqa: E402
from .cohort import Cohort  # noqa: E402
from .economics import UnitEconomics, exact, price_for_margin, sweep  # noqa: E402
from .plans import FreeTier, Plan, PlanResult, compare, simulate, solve_cap  # noqa: E402
from .population import Empirical, Fixed, Lognormal, Pareto, concentration  # noqa: E402
from .pricing import TokenPrice, Usage  # noqa: E402
from .projection import MonthPoint, Trend, economics_at, project, squeeze_month  # noqa: E402
from .routing import Tier, expected_cost, max_escalation_to_beat  # noqa: E402
from .scenario import Analysis, Scenario, analyze  # noqa: E402
from .sensitivity import Swing, grid, tornado  # noqa: E402

__all__ = [
    "AgentRun", "Analysis", "Cohort", "Empirical", "Fixed", "FreeTier", "Lognormal", "MonthPoint", "Pareto", "Plan", "PlanResult",
    "Scenario", "Swing", "Tier", "TokenPrice", "Trend", "UnitEconomics", "Usage", "analyze", "compare", "concentration", "cost_curve",
    "economics_at", "exact", "expected_cost", "grid", "max_escalation_to_beat", "max_steps_for_budget", "price_for_margin", "project",
    "simulate", "solve_cap", "squeeze_month", "sweep", "tornado",
]
