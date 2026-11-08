# token.econ

The unit economics of an AI product on one page of code. Given price, usage, and token prices, it tells you the margin per user, how many users cover your fixed costs, how many requests a user can make before you miss your target margin, and what to charge to hit it. Standard library only.

Falling token prices do not save a product whose usage grows faster than its price. This is the arithmetic that shows whether yours is one of them.

## Use

```python
from token_econ import UnitEconomics, price_for_margin, sweep

plan = UnitEconomics(price_per_user=20, requests_per_user=200, input_tokens=1500,
                     output_tokens=500, input_price_per_1m=3, output_price_per_1m=15)

plan.gross_margin()                  # 0.88
plan.breakeven_users(17_600)         # 1000
plan.max_requests_for_margin(0.90)   # 166 requests a month keeps a 90 percent margin
price_for_margin(plan, 0.90)         # the lowest price that reaches 90 percent at current usage
sweep(plan, "requests_per_user", [100, 200, 400])   # margin as usage grows
```

## Design

- **Immutable inputs.** `with_()` returns a modified copy, so scenarios never contaminate each other.
- **Honest breakeven.** If each user loses money, breakeven is `None`, not a large number that looks reassuring.
- **Usage limits from margin.** `max_requests_for_margin` turns a margin target into a fair-use cap you can actually set.
- **Pairs with** [model.ledger](https://github.com/AndMaverick/model.ledger) for real spend and [roi.lens](https://github.com/AndMaverick/roi.lens) for the value side.

## Tests

```
python -m unittest discover -s tests -t . -v
```

MIT licensed. Built by [AndMaverick LLC](https://www.andmaverick.com).

---

**Related reading:** [The Meter Is Running. Nobody Built the Architecture to Stop It.](https://blog.andmaverick.com/the-meter-is-running-nobody-built-the-architecture-to-stop-it?utm_source=github&utm_medium=repo&utm_campaign=token.econ)

**If this is your problem.** If your AI spend is outrunning your results, that is an architecture problem, and it is solvable. I am happy to look at it with you. [Begin a conversation](https://calendly.com/andmaverick?utm_source=github&utm_medium=repo&utm_campaign=token.econ).
