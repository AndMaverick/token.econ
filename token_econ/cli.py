"""Command line. Exit codes: 0 pass, 1 a gate in the scenario was missed, 2 bad input.

    token-econ analyze SCENARIO [--format text|json|markdown] [--fail-on never]
    token-econ project SCENARIO [--months N]
    token-econ tornado SCENARIO [--pct 0.2]
    token-econ sweep SCENARIO FIELD V1 V2 ...
    token-econ fingerprint SCENARIO
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import __version__
from .projection import Trend, project
from .scenario import Scenario, analyze, to_markdown, to_text
from .sensitivity import tornado


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="token-econ", description="Unit economics for AI products, with gates for CI.")
    parser.add_argument("--version", action="version", version=f"token.econ {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("analyze", help="full report for a scenario; exit 1 if a gate is missed")
    p.add_argument("scenario")
    p.add_argument("--format", choices=["text", "json", "markdown"], default="text")
    p.add_argument("--fail-on", choices=["gate", "never"], default="gate")
    p = sub.add_parser("project", help="margin month by month under the scenario's trend")
    p.add_argument("scenario")
    p.add_argument("--months", type=int)
    p = sub.add_parser("tornado", help="which inputs move the margin most")
    p.add_argument("scenario")
    p.add_argument("--pct", type=float)
    p = sub.add_parser("sweep", help="margin as one input varies")
    p.add_argument("scenario")
    p.add_argument("field")
    p.add_argument("values", nargs="+", type=float)
    p = sub.add_parser("fingerprint", help="the SHA-256 of the scenario")
    p.add_argument("scenario")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except (ValueError, TypeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(a: argparse.Namespace) -> int:
    s = Scenario.load(a.scenario)
    if a.command == "analyze":
        result = analyze(s)
        if a.format == "json":
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        elif a.format == "markdown":
            print(to_markdown(result), end="")
        else:
            print(to_text(result))
        return 0 if a.fail_on == "never" or result.ok else 1
    if a.command == "project":
        trend = s.trend or Trend()
        for p in project(s.economics, trend, a.months if a.months is not None else s.months):
            print(f"month {p.month:>3}  inference ${p.inference_cost:8.2f}  profit ${p.profit:8.2f}  margin {p.margin:7.1%}")
    elif a.command == "tornado":
        for w in tornado(s.economics, a.pct if a.pct is not None else s.tornado_pct):
            print(f"{w.field:<22} {w.margin_at_low:7.1%} .. {w.margin_at_high:7.1%}   span {w.span * 100:5.1f} points")
    elif a.command == "sweep":
        for v in a.values:
            print(f"{a.field}={v:g}  margin {s.economics.with_(**{a.field: v}).gross_margin():.4f}")
    else:
        print(s.fingerprint())
    return 0


if __name__ == "__main__":
    sys.exit(main())
