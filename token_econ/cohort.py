"""Customer lifetime value with churn and discounting, and how long it takes to earn back what a customer cost to win."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .economics import UnitEconomics
from .pricing import _check


@dataclass(frozen=True)
class Cohort:
    """A customer pays at the start of each month they are still active. Each month a ``monthly_churn`` fraction leave, and money
    received later is worth less at ``monthly_discount`` (0 for no discounting). ``profit_per_month`` is what one active customer
    earns you after variable costs, for example ``UnitEconomics.profit_per_user()``."""

    profit_per_month: float
    monthly_churn: float
    cac: float = 0.0
    monthly_discount: float = 0.0

    def __post_init__(self) -> None:
        _check("monthly_churn", self.monthly_churn, high=1.0)
        _check("cac", self.cac)
        _check("monthly_discount", self.monthly_discount)

    @classmethod
    def from_economics(cls, econ: UnitEconomics, monthly_churn: float, cac: float = 0.0, monthly_discount: float = 0.0) -> Cohort:
        return cls(econ.profit_per_user(), monthly_churn, cac, monthly_discount)

    @property
    def retention(self) -> float:
        """Month-over-month value kept: survival divided by discounting. Below 1 the lifetime sum converges."""
        return (1 - self.monthly_churn) / (1 + self.monthly_discount)

    def ltv(self) -> float | None:
        """Expected lifetime profit per customer: ``profit / (1 - retention)``. ``None`` when there is neither churn nor discounting,
        because the sum never ends."""
        q = self.retention
        if q >= 1:
            return None
        return self.profit_per_month / (1 - q)

    def ltv_to_cac(self) -> float | None:
        ltv = self.ltv()
        if ltv is None or self.cac == 0:
            return None
        return ltv / self.cac

    def expected_months(self) -> float | None:
        """Expected number of paid months per customer (no discounting): ``1 / churn``."""
        return None if self.monthly_churn == 0 else 1 / self.monthly_churn

    def payments_to_recover_cac(self) -> int | None:
        """How many monthly payments from a customer (counting survivors and discounting) it takes for expected profit to cover CAC, or
        ``None`` if a customer never pays back what they cost. A cohort of customers has recovered CAC once this many months have
        passed."""
        if self.cac == 0:
            return 0
        if self.profit_per_month <= 0:
            return None
        q = self.retention
        if q >= 1:
            return math.ceil(self.cac / self.profit_per_month)
        need = self.cac * (1 - q) / self.profit_per_month  # fraction of lifetime value that must be earned
        if need >= 1:
            return None
        n = math.ceil(math.log(1 - need) / math.log(q))  # smallest n with 1 - q**n >= need
        while self._recovered(n - 1) and n > 1:  # floating point can land one off either way; check the neighbours
            n -= 1
        while not self._recovered(n):
            n += 1
        return n

    def _recovered(self, payments: int) -> bool:
        q = self.retention
        total = self.profit_per_month * payments if q == 1 else self.profit_per_month * (1 - q ** payments) / (1 - q)
        return total >= self.cac
