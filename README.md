# token.econ

A product sells for $20 a month. Token prices fall every quarter, so the model behind it gets cheaper. Then a few heavy users start sending ten times what the average user sends, and the margin that looked safe is gone by the spring. Falling token prices do not save a product whose usage grows faster than its price falls. Whether yours is one of them is arithmetic, and it is arithmetic that depends on assumptions you can write down.

token.econ does that arithmetic exactly. Given price, usage, token prices, caching and the costs around them, it tells you the margin per user, how many users cover your fixed costs, how many requests a user can make before you miss your target margin, what to charge to hit it, which users lose you money, and the month your margin crosses your floor if usage grows faster than token prices fall. It also runs as a gate in CI, so a pricing or product change that breaks the margin fails the build. Standard library only, Python 3.10 to 3.13, typed.

## Use

```python
from token_econ import UnitEconomics, price_for_margin, sweep

plan = UnitEconomics(price_per_user=20, requests_per_user=200, input_tokens=1500,
                     output_tokens=500, input_price_per_1m=3, output_price_per_1m=15)

plan.gross_margin()                  # 0.88
plan.breakeven_users(17_600)         # 1000
plan.max_requests_for_margin(0.90)   # 166 requests a month keeps a 90 percent margin
price_for_margin(plan, 0.90)         # 24.0, the lowest price that reaches 90 percent at current usage
sweep(plan, "requests_per_user", [100, 200, 400])   # margin as usage grows
```

Everything in 0.1 works unchanged. The model now also takes prompt caching, hidden reasoning tokens, retries, infrastructure overhead, support cost and payment fees:

```python
from token_econ import UnitEconomics, Lognormal, Plan, Trend, Cohort, compare, squeeze_month

e = UnitEconomics(price_per_user=24, requests_per_user=220, input_tokens=3200, output_tokens=450,
                  input_price_per_1m=3, output_price_per_1m=15, cached_input_price_per_1m=0.3, cache_hit_rate=0.55,
                  reasoning_tokens=120, retry_rate=0.04, infra_overhead_rate=0.08,
                  payment_fee_rate=0.029, payment_fee_fixed=0.30)
e.required_cache_hit_rate(0.80)          # the cache hit rate you need for an 80 percent margin, or None if out of reach
e.max_output_tokens_for_margin(0.80)     # how long answers can get before you miss it

# usage is not average: a few heavy users carry the cost
users = Lognormal(median=130, sigma=0.9)
compare(e, [Plan("flat", 24), Plan("capped", 24, hard_cap=600), Plan("hybrid", 14, included_requests=150, overage_per_request=0.06)], users)

# the squeeze: prices fall 3 percent a month, usage grows 4 percent, tokens per request 1 percent
squeeze_month(e, Trend(price_decline=0.03, usage_growth=0.04, token_growth=0.01), target_margin=0.70)   # 34, the first month below a 70 percent margin

Cohort.from_economics(e, monthly_churn=0.05, cac=70).ltv_to_cac()
```

## What is in it

| Module | Answers |
|---|---|
| `economics` | margin, profit, breakeven users, max requests for a margin, price for a margin, cache hit rate and output length needed for a margin |
| `pricing` | what one request costs with cached reads, cache writes, reasoning tokens and a batch discount (you supply every price) |
| `agent` | why agent loops blow budgets: input tokens grow with the square of the steps; cost with and without caching; the most steps a budget buys |
| `population` | lognormal, Pareto and measured usage distributions with exact means and the share held by the top users |
| `plans` | flat, capped, metered and hybrid plans run across a population: margin, share of users losing money, whale cost share; the highest cap that hits a margin; free tier cost and the paid share it needs |
| `projection` | margin month by month under price decline, usage growth and token growth; the first month below your floor; the price decline that exactly cancels growth |
| `cohort` | lifetime value with churn and discounting, LTV to CAC, payments to recover CAC |
| `routing` | expected cost of a model cascade and the escalation rate above which the cheap model stops paying |
| `sensitivity` | tornado ranking of the inputs that move margin most, and grids over two inputs |
| `scenario`, CLI | one JSON file of assumptions plus gates; a report with a fingerprint; exit code 1 when a gate is missed |

## Design

- **Exact where it counts.** Integer answers (users, requests) are computed on fractions made from the decimals you typed, so they are never off by one from floating point. A plain float `ceil(2.1 / 0.3)` is 8; the exact answer is 7. Float results are the correctly rounded exact values: `0.88`, not `0.8800000000000001`.
- **One definition of cost.** The arithmetic is written once over a number type, so the float and exact answers cannot drift apart.
- **Honest breakeven.** If each user loses money, or earns exactly nothing, breakeven is `None`, not a number that looks reassuring. A target that no price can reach (fee rate plus margin of 100 percent or more) raises instead of returning nonsense. Requests that cost nothing have no limit (`None`), not a limit of zero.
- **Immutable inputs.** `with_()` returns a modified copy, so scenarios never contaminate each other.
- **You supply the prices.** This library ships no price table, because a stale one is worse than none. The example files use round numbers and say so.
- **Strict scenario files.** An unknown key or gate name is an error naming it, because a misspelt `min_marjin` would otherwise silently pass.
- **Deterministic.** The same scenario gives output identical byte for byte in any process under any hash seed, and the report carries the SHA-256 of the scenario it was computed from.

## Command line

```
token-econ analyze SCENARIO [--format text|json|markdown] [--fail-on gate|never]
token-econ project SCENARIO [--months N]     token-econ tornado SCENARIO [--pct 0.2]
token-econ sweep SCENARIO FIELD V1 V2 ...    token-econ fingerprint SCENARIO
```

Exit codes: 0 every gate passed, 1 a gate was missed, 2 bad input. Gates available: `min_margin`, `min_plan_margin`, `max_loss_share`, `max_top_cost_share`, `min_squeeze_month`, `min_ltv_cac`, `min_breakeven_headroom`. Put `analyze` in CI beside the pricing file and a change that breaks unit economics cannot merge.

## Two examples, same shape

`examples/saas_assistant.json` is a healthy product and passes every gate:

```
  cost per request        $0.015049
  inference per user      $3.31   payment fees $1.00
  profit per user         $18.39   gross margin 76.6% (target 70.0%)
  max requests at target  325
  price for target        18.12
  breakeven users         979
  plan flat-capped    margin 79.1%  profit/user $18.99  losing users 0.0%  top 1% of users = 3.3% of cost
  plan flat           margin 78.2%  profit/user $18.77  losing users 0.4%  top 1% of users = 7.7% of cost
  plan hybrid         margin 73.3%  profit/user $13.99  losing users 0.0%  top 1% of users = 7.7% of cost
  projection              margin 76.6% now, 68.8% in 24 months; below target from month 21; cost x1.0189 a month (break-even price decline 4.8%)
  cohort                  LTV $309.62  LTV/CAC 4.42  payback 5 payments
  free tier               $0.38 per free user; paid share needed 2.0%
  gate min_margin             PASS  actual 0.7664  limit 0.7000  (gross margin at today's assumptions)
  gate min_plan_margin        PASS  actual 0.7334  limit 0.5000  (worst plan)
  gate max_loss_share         PASS  actual 0.0040  limit 0.2500  (worst plan)
  gate min_squeeze_month      PASS  actual 21  limit 12  (months until margin falls below the target; none within the horizon passes)
  gate min_ltv_cac            PASS  actual 4.4232  limit 3  (lifetime value over acquisition cost)
PASS
```

`examples/squeezed_assistant.json` has heavier users, longer contexts and slower price declines. The flat plan loses money on the average user, a third of users cost more than they pay, and every gate fails:

```
  cost per request        $0.046364
  inference per user      $18.55   payment fees $0.88
  profit per user         $0.57   gross margin 2.9% (target 70.0%)
  max requests at target  110
  price for target        69.54
  breakeven users         43,536
  plan flat-capped    margin 35.1%  profit/user $7.02  losing users 31.4%  top 1% of users = 1.9% of cost
  plan flat           margin -23.1%  profit/user $-4.63  losing users 31.4%  top 1% of users = 15.2% of cost
  projection              margin 2.9% now, -217.6% in 18 months; below target from month 0; cost x1.0700 a month (break-even price decline 8.4%)
  cohort                  LTV $7.18  LTV/CAC 0.08  payback n/a payments
  gate min_margin             FAIL  actual 0.0287  limit 0.7000  (gross margin at today's assumptions)
  gate max_loss_share         FAIL  actual 0.3140  limit 0.2000  (worst plan)
  gate min_squeeze_month      FAIL  actual 0  limit 12  (months until margin falls below the target; none within the horizon passes)
  gate min_ltv_cac            FAIL  actual 0.0798  limit 3  (lifetime value over acquisition cost)
FAIL
```

Both use round example prices, not any vendor's list.

## Measured

`python examples/evaluate.py` prints these. Every case is seeded, so every figure reproduces.

```
float answers exact     20000/20000 profit and cost per request equal the correctly rounded exact value
breakeven users         20000/20000 equal the exact ceiling; a plain float ceil is wrong in 0 of 4951 profitable cases
max requests            20000/20000 equal the exact floor; a plain float floor is wrong in 0 of them
exact-multiple traps    20000/20000 correct; a plain float ceil is wrong in 2816 (14.1%) of cases where the answer is a whole number
squeeze month           334/334 closed-form answers equal a month by month search of the full model
population means        worst relative error 1.3e-16 across 18 distribution and size combinations
top 1% share            lognormal sigma 1.0: library 0.0924, analytic 0.0924, 400,000 random draws 0.0929; pareto alpha 2 exactly 0.100 = 0.01^(1-1/2)
cohort LTV              worst relative error 1.5e-16 against a 20,000-term sum
speed                   20000 random economics checked three ways in 7.6s on this machine
```

"Exact" means computed on fractions by code in `tests/test_properties.py` that shares nothing with the library. The random cases rarely land the answer on a whole number, which is where floating point fails, so the traps row builds cases where it does: there a plain float ceiling is wrong in about one case in seven. The squeeze month is found in closed form and then confirmed against the full model at the boundary; the row shows it equals a month by month search. Timing depends on the machine.

## Honest limits

- **The arithmetic is checked. The assumptions are yours.** If your usage distribution, churn or price trend is wrong, the exactness of the answer is no comfort. Run the tornado to see which assumption matters most, then measure that one.
- **Usage distributions are stylised.** Lognormal and Pareto describe many products and not yours. Use `Empirical` with your own counts per user when you have them.
- **Plans assume users behave the same under a limit.** A hard cap holds heavy users at the cap. It does not model them churning, buying more, or complaining.
- **Averages per request.** Tokens per request are one average per user. A product whose requests are bimodal (many short, few huge) should be modelled as two products, or as an agent run.
- **Cohort maths is the simple kind.** Constant churn, payment at the start of each month, no expansion revenue and no seasonality.
- **Projection is compounding rates.** It does not know about the next model release or a repricing you have not entered.
- **Plan results at a finite sample.** `simulate` represents the population with 1,000 equal slices by default. Means are exact for any slice count, but plans with caps or overages are slightly more accurate with more slices.
- **Not financial advice.** It computes what your inputs imply.

## Tests

```
python -m unittest discover -s tests -t . -v
```

91 tests, run on Python 3.10 to 3.13 in CI with ruff and mypy in strict mode. CI also runs the healthy example's gates.

Pairs with [model.ledger](https://github.com/AndMaverick/model.ledger) for real spend and [roi.lens](https://github.com/AndMaverick/roi.lens) for the value side.

MIT licensed. Built by [AndMaverick LLC](https://www.andmaverick.com).

---

**Read the thinking:** [The Meter Is Running. Nobody Built the Architecture to Stop It.](https://blog.andmaverick.com/the-meter-is-running-nobody-built-the-architecture-to-stop-it?utm_source=github&utm_medium=repo&utm_campaign=token.econ)

**If this is your problem.** If your AI spend is outrunning your results, that is an architecture problem, and it is solvable. I am happy to look at it with you. [Begin a conversation](https://calendly.com/andmaverick?utm_source=github&utm_medium=repo&utm_campaign=token.econ).
