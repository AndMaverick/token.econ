"""A scenario file: one JSON document that holds your assumptions and the gates they must pass. ``analyze`` turns it into a report you
can put in a pull request, and a run fails (exit 1 from the CLI) when a gate is missed."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

from . import __version__
from .cohort import Cohort
from .economics import UnitEconomics
from .plans import FreeTier, Plan, compare
from .population import Population, from_dict
from .projection import Trend, project, squeeze_month
from .sensitivity import tornado

T = TypeVar("T")
KEYS = {"name", "economics", "population", "plans", "trend", "months", "cohort", "free_tier", "fixed_monthly_cost", "target_margin",
        "tornado_pct", "gates"}
GATES = {"min_margin", "min_ltv_cac", "min_squeeze_month", "max_loss_share", "max_top_cost_share", "min_plan_margin", "min_breakeven_headroom"}


def _build(cls: type[T], data: Any, where: str) -> T:
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected an object")
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    extra = sorted(set(data) - known)
    if extra:
        raise ValueError(f"{where}: unknown key {extra[0]!r} (allowed: {', '.join(sorted(known))})")
    try:
        return cls(**data)
    except TypeError as exc:
        raise ValueError(f"{where}: {exc}") from exc


@dataclass(frozen=True)
class Scenario:
    economics: UnitEconomics
    name: str = ""
    population: Population | None = None
    plans: tuple[Plan, ...] = ()
    trend: Trend | None = None
    months: int = 24
    cohort: dict[str, float] | None = None
    free_tier: FreeTier | None = None
    fixed_monthly_cost: float | None = None
    target_margin: float = 0.7
    tornado_pct: float = 0.2
    gates: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scenario:
        if not isinstance(data, dict):
            raise ValueError("scenario: expected an object")
        extra = sorted(set(data) - KEYS)
        if extra:
            raise ValueError(f"scenario: unknown key {extra[0]!r} (allowed: {', '.join(sorted(KEYS))})")
        if "economics" not in data:
            raise ValueError("scenario.economics: required")
        gates = data.get("gates", {})
        bad = sorted(set(gates) - GATES)
        if bad:
            raise ValueError(f"scenario.gates: unknown gate {bad[0]!r} (allowed: {', '.join(sorted(GATES))})")
        cohort = data.get("cohort")
        if cohort is not None:
            _build(Cohort, {"profit_per_month": 0.0, **cohort}, "scenario.cohort")
        return cls(
            _build(UnitEconomics, data["economics"], "scenario.economics"), str(data.get("name", "")),
            from_dict(data["population"]) if "population" in data else None,
            tuple(_build(Plan, p, f"scenario.plans[{i}]") for i, p in enumerate(data.get("plans", []))),
            _build(Trend, data["trend"], "scenario.trend") if "trend" in data else None, int(data.get("months", 24)), cohort,
            _build(FreeTier, data["free_tier"], "scenario.free_tier") if "free_tier" in data else None,
            data.get("fixed_monthly_cost"), float(data.get("target_margin", 0.7)), float(data.get("tornado_pct", 0.2)), dict(gates), data)

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: not valid JSON ({exc})") from exc
        except OSError as exc:
            raise ValueError(f"{path}: cannot read ({exc.strerror})") from exc

    def fingerprint(self) -> str:
        """SHA-256 of the canonical scenario, so a report names exactly the assumptions it was computed from."""
        canon = json.dumps(self.raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canon.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class Gate:
    name: str
    ok: bool
    actual: float | None
    limit: float
    detail: str = ""


@dataclass
class Analysis:
    name: str
    fingerprint: str
    sections: dict[str, Any]
    gates: list[Gate]

    @property
    def ok(self) -> bool:
        return all(g.ok for g in self.gates)

    def to_dict(self) -> dict[str, Any]:
        return {"tool": "token.econ", "version": __version__, "name": self.name, "fingerprint": self.fingerprint, "ok": self.ok,
                "gates": [{"name": g.name, "ok": g.ok, "actual": g.actual, "limit": g.limit, "detail": g.detail} for g in self.gates],
                **self.sections}


def analyze(s: Scenario) -> Analysis:
    e, target = s.economics, s.target_margin
    sections: dict[str, Any] = {}
    gates: list[Gate] = []
    margin = e.gross_margin()
    unit: dict[str, Any] = {
        "cost_per_request": e.cost_per_request(), "inference_cost_per_user": e.inference_cost(), "payment_fees_per_user": e.payment_fees(),
        "profit_per_user": e.profit_per_user(), "gross_margin": margin, "target_margin": target,
        "max_requests_per_user_at_target": e.max_requests_for_margin(target),
    }
    try:
        unit["price_for_target_margin"] = e.price_for_margin(target)
    except ValueError as exc:
        unit["price_for_target_margin"] = None
        unit["price_note"] = str(exc)
    if e.cached_input_price_per_1m is not None:
        unit["cache_hit_rate_for_target"] = e.required_cache_hit_rate(target)
    unit["max_output_tokens_at_target"] = e.max_output_tokens_for_margin(target)
    if s.fixed_monthly_cost is not None:
        unit["breakeven_users"] = e.breakeven_users(s.fixed_monthly_cost)
    sections["unit"] = unit
    sections["tornado"] = [{"field": w.field, "margin_low": w.margin_at_low, "margin_high": w.margin_at_high, "span": w.span}
                           for w in tornado(e, s.tornado_pct)[:6]]

    if "min_margin" in s.gates:
        gates.append(Gate("min_margin", margin >= s.gates["min_margin"], margin, s.gates["min_margin"], "gross margin at today's assumptions"))

    if s.population is not None:
        plans = s.plans or (Plan("flat", e.price_per_user),)
        results = compare(e, plans, s.population)
        sections["plans"] = [r.to_dict() for r in results]
        sections["population"] = {"mean_requests": s.population.mean(), "kind": type(s.population).__name__.lower()}
        for key, attr, worse_when_above in (("min_plan_margin", "margin", False), ("max_loss_share", "loss_share", True), ("max_top_cost_share", "top_cost_share", True)):
            if key in s.gates:
                limit = s.gates[key]
                values = [getattr(r, attr) for r in results]
                actual = (max if worse_when_above else min)(v if v is not None else -1.0 for v in values)
                ok = actual <= limit if worse_when_above else actual >= limit
                gates.append(Gate(key, ok, actual, limit, "worst plan" if len(results) > 1 else results[0].plan))

    if s.trend is not None:
        points = project(e, s.trend, s.months)
        sq = squeeze_month(e, s.trend, target)
        sections["projection"] = {
            "cost_factor_per_month": s.trend.cost_factor, "squeezing": s.trend.squeezing,
            "critical_price_decline": s.trend.critical_price_decline(), "margin_now": points[0].margin, "margin_end": points[-1].margin,
            "months": s.months, "below_target_from_month": sq,
            "margins": [round(p.margin, 6) for p in points],
        }
        if "min_squeeze_month" in s.gates:
            need = s.gates["min_squeeze_month"]
            gates.append(Gate("min_squeeze_month", sq is None or sq >= need, None if sq is None else float(sq), need,
                              "months until margin falls below the target; none within the horizon passes"))

    if s.cohort is not None:
        c = Cohort(e.profit_per_user(), **{k: v for k, v in s.cohort.items() if k != "profit_per_month"})
        ratio = c.ltv_to_cac()
        sections["cohort"] = {"ltv": c.ltv(), "ltv_to_cac": ratio, "payments_to_recover_cac": c.payments_to_recover_cac(),
                              "expected_paid_months": c.expected_months()}
        if "min_ltv_cac" in s.gates:
            never_ends = c.ltv() is None  # no churn and no discounting: lifetime value is unbounded
            gates.append(Gate("min_ltv_cac", c.cac == 0 or never_ends or (ratio is not None and ratio >= s.gates["min_ltv_cac"]), ratio,
                              s.gates["min_ltv_cac"], "lifetime value over acquisition cost"))

    if s.free_tier is not None:
        sections["free_tier"] = {"cost_per_free_user": s.free_tier.cost_per_free_user(e),
                                 "blended_profit_per_paid_user": s.free_tier.blended_profit_per_paid_user(e),
                                 "required_paid_share": s.free_tier.required_conversion(e)}

    if "min_breakeven_headroom" in s.gates and s.fixed_monthly_cost is not None:
        users = e.breakeven_users(s.fixed_monthly_cost)
        gates.append(Gate("min_breakeven_headroom", users is not None, None if users is None else float(users), s.gates["min_breakeven_headroom"],
                          "breakeven needs each user to be profitable"))
    return Analysis(s.name, s.fingerprint(), sections, gates)


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _num(x: float | int | None, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{int(x):,}" if float(x).is_integer() else f"{x:,.{digits}f}"


def to_text(a: Analysis) -> str:
    u = a.sections["unit"]
    lines = [f"token.econ {__version__}  {a.name or 'scenario'}  fingerprint {a.fingerprint[:16]}",
             f"  cost per request        ${u['cost_per_request']:.6f}",
             f"  inference per user      ${u['inference_cost_per_user']:.2f}   payment fees ${u['payment_fees_per_user']:.2f}",
             f"  profit per user         ${u['profit_per_user']:.2f}   gross margin {_pct(u['gross_margin'])} (target {_pct(u['target_margin'])})",
             f"  max requests at target  {_num(u['max_requests_per_user_at_target'])}",
             f"  price for target        {_num(u.get('price_for_target_margin'), 2)}"]
    if "breakeven_users" in u:
        lines.append(f"  breakeven users         {_num(u['breakeven_users'])}")
    for p in a.sections.get("plans", []):
        lines.append(f"  plan {p['plan']:<14} margin {_pct(p['margin'])}  profit/user ${p['profit_per_user']:.2f}  losing users {_pct(p['loss_share'])}  "
                     f"top 1% of users = {_pct(p['top_cost_share'])} of cost")
    pr = a.sections.get("projection")
    if pr:
        sq = pr["below_target_from_month"]
        lines.append(f"  projection              margin {_pct(pr['margin_now'])} now, {_pct(pr['margin_end'])} in {pr['months']} months; "
                     f"{'below target from month ' + str(sq) if sq is not None else 'stays above target'}; "
                     f"cost x{pr['cost_factor_per_month']:.4f} a month (break-even price decline {_pct(pr['critical_price_decline'])})")
    c = a.sections.get("cohort")
    if c:
        lines.append(f"  cohort                  LTV ${_num(c['ltv'], 2)}  LTV/CAC {_num(c['ltv_to_cac'], 2)}  payback {_num(c['payments_to_recover_cac'])} payments")
    f = a.sections.get("free_tier")
    if f:
        lines.append(f"  free tier               ${f['cost_per_free_user']:.2f} per free user; paid share needed {_pct(f['required_paid_share'])}")
    for g in a.gates:
        lines.append(f"  gate {g.name:<22} {'PASS' if g.ok else 'FAIL'}  actual {_num(g.actual)}  limit {_num(g.limit)}  ({g.detail})")
    lines.append("PASS" if a.ok else "FAIL")
    return "\n".join(lines)


def to_markdown(a: Analysis) -> str:
    u = a.sections["unit"]
    out = [f"# {a.name or 'Unit economics'}", "", f"`{a.fingerprint}`  ", f"**{'PASS' if a.ok else 'FAIL'}**", "",
           "| Metric | Value |", "|---|---|",
           f"| Cost per request | ${u['cost_per_request']:.6f} |", f"| Profit per user | ${u['profit_per_user']:.2f} |",
           f"| Gross margin | {_pct(u['gross_margin'])} (target {_pct(u['target_margin'])}) |",
           f"| Max requests at target | {_num(u['max_requests_per_user_at_target'])} |",
           f"| Price for target margin | {_num(u.get('price_for_target_margin'), 2)} |"]
    if "breakeven_users" in u:
        out.append(f"| Breakeven users | {_num(u['breakeven_users'])} |")
    if a.sections.get("plans"):
        out += ["", "## Plans", "", "| Plan | Margin | Profit / user | Losing users | Top 1% of cost |", "|---|---|---|---|---|"]
        out += [f"| {p['plan']} | {_pct(p['margin'])} | ${p['profit_per_user']:.2f} | {_pct(p['loss_share'])} | {_pct(p['top_cost_share'])} |"
                for p in a.sections["plans"]]
    if a.sections.get("tornado"):
        out += ["", "## What moves the margin most", "", "| Input (+/- swing) | Margin low | Margin high |", "|---|---|---|"]
        out += [f"| {t['field']} | {_pct(t['margin_low'])} | {_pct(t['margin_high'])} |" for t in a.sections["tornado"]]
    if a.gates:
        out += ["", "## Gates", "", "| Gate | Result | Actual | Limit |", "|---|---|---|---|"]
        out += [f"| {g.name} | {'pass' if g.ok else '**FAIL**'} | {_num(g.actual)} | {_num(g.limit)} |" for g in a.gates]
    return "\n".join(out) + "\n"
