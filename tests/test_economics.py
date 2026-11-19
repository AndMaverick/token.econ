import unittest

from token_econ import TokenPrice, UnitEconomics, Usage, exact, price_for_margin, sweep

BASE = UnitEconomics(20, 200, 1500, 500, 3, 15)


class PricingTests(unittest.TestCase):
    def test_cost_with_cache_and_write_prices(self):
        # fresh 500 x 3 + cached 400 x 0.3 + written 100 x 3.75 + output 500 x 15, per million tokens
        price = TokenPrice(3, 15, 0.3, 3.75)
        self.assertAlmostEqual(price.cost(Usage(1000, 500, 400, 100)), 0.009495)
        self.assertAlmostEqual(price.cost(Usage(1000, 500, 400, 100), batch=True), 0.009495)  # no discount set
        self.assertAlmostEqual(TokenPrice(3, 15, batch_discount=0.5).cost(Usage(1000, 500), batch=True), 0.0105 / 2)

    def test_reasoning_tokens_bill_as_output(self):
        self.assertAlmostEqual(TokenPrice(3, 15).cost(Usage(0, 100, reasoning_tokens=100)), 200 * 15 / 1e6)

    def test_missing_cache_prices_mean_no_discount(self):
        self.assertAlmostEqual(TokenPrice(3, 15).cost(Usage(1000, 0, cached_input_tokens=1000)), 0.003)

    def test_bad_input_is_refused(self):
        for bad in [lambda: TokenPrice(-1, 1), lambda: TokenPrice(1, 1, batch_discount=1.5), lambda: Usage(100, 0, 80, 40),
                    lambda: Usage(-1), lambda: TokenPrice(float("nan"), 1)]:
            with self.assertRaises(ValueError):
                bad()


class UnitEconomicsTests(unittest.TestCase):
    def test_the_original_numbers_are_unchanged(self):
        self.assertEqual(BASE.cost_per_request(), 0.012)
        self.assertEqual(BASE.inference_cost(), 2.4)
        self.assertEqual(BASE.profit_per_user(), 17.6)
        self.assertEqual(BASE.gross_margin(), 0.88)  # exactly, not 0.8800000000000001
        self.assertEqual(BASE.breakeven_users(17_600), 1000)
        self.assertEqual(BASE.max_requests_for_margin(0.90), 166)
        self.assertEqual(price_for_margin(BASE, 0.9), 24.0)

    def test_caching_reasoning_retries_overhead(self):
        e = UnitEconomics(24, 220, 3200, 450, 3, 15, cached_input_price_per_1m=0.3, cache_hit_rate=0.55, reasoning_tokens=120, retry_rate=0.04,
                          infra_overhead_rate=0.08)
        # (1440 fresh x 3 + 1760 cached x 0.3 + 570 output x 15) / 1e6 = 0.013398, then x1.04 x1.08
        self.assertAlmostEqual(e.cost_per_request(), 0.013398 * 1.04 * 1.08, places=12)

    def test_fees_and_per_user_costs_reduce_profit(self):
        e = BASE.with_(payment_fee_rate=0.03, payment_fee_fixed=0.3, support_cost_per_user=1, other_cost_per_user=0.5)
        self.assertAlmostEqual(e.payment_fees(), 0.9)
        self.assertAlmostEqual(e.profit_per_user(), 20 - 0.9 - 1 - 0.5 - 2.4)
        self.assertAlmostEqual(e.gross_margin(), e.profit_per_user() / 20)

    def test_integer_answers_are_exact_where_floats_are_not(self):
        # 2.1 / 0.3 is 7.000000000000001 in floating point, so a float ceil says 8 users. Exactly it is 7.
        e = UnitEconomics(0.3, 0, 0, 0, 0, 0)
        self.assertEqual(e.breakeven_users(2.1), 7)
        self.assertEqual(e.breakeven_users(2.1000000001), 8)

    def test_no_breakeven_when_each_user_loses_money(self):
        self.assertIsNone(BASE.with_(requests_per_user=5000).breakeven_users(1000))
        self.assertIsNone(UnitEconomics(12, 1000, 1500, 500, 3, 15).breakeven_users(1000))  # 12 - 1000 x 0.012 is exactly zero: still no breakeven

    def test_max_requests_edges(self):
        hit, miss = BASE.with_(requests_per_user=166).gross_margin(), BASE.with_(requests_per_user=167).gross_margin()
        self.assertGreaterEqual(hit, 0.90)
        self.assertLess(miss, 0.90)
        self.assertEqual(BASE.max_requests_for_margin(0.999999), 0)               # the margin is out of reach even with no requests
        self.assertIsNone(BASE.with_(input_tokens=0, output_tokens=0).max_requests_for_margin(0.5))  # free requests have no limit
        self.assertEqual(BASE.with_(input_tokens=0, output_tokens=0, other_cost_per_user=30).max_requests_for_margin(0.5), 0)

    def test_price_for_margin_allows_for_the_fee(self):
        e = BASE.with_(payment_fee_rate=0.1, payment_fee_fixed=0.5)
        price = e.price_for_margin(0.8)
        self.assertAlmostEqual(e.with_(price_per_user=price).gross_margin(), 0.8)
        with self.assertRaisesRegex(ValueError, "unreachable"):
            e.price_for_margin(0.9)

    def test_required_cache_hit_rate_and_output_ceiling_are_exact_inverses(self):
        e = UnitEconomics(24, 220, 3200, 450, 3, 15, cached_input_price_per_1m=0.3, cache_hit_rate=0.2)
        need = e.required_cache_hit_rate(0.9)
        self.assertAlmostEqual(need, 0.5 + 0.25 * (0.9 - 0.889725) / (0.909525 - 0.889725))  # margins at 0.5 and 0.75, read off by hand
        self.assertAlmostEqual(e.with_(cache_hit_rate=need).gross_margin(), 0.9)
        self.assertEqual(e.required_cache_hit_rate(0.8), 0.0)                      # already met with no cache at all
        self.assertIsNone(e.required_cache_hit_rate(0.99))                        # out of reach even at a 100 percent hit rate
        cap = e.max_output_tokens_for_margin(0.9)
        self.assertAlmostEqual(e.with_(output_tokens=cap).gross_margin(), 0.9)
        self.assertGreater(e.with_(output_tokens=cap - 1).gross_margin(), 0.9)
        self.assertLess(e.with_(output_tokens=cap + 1).gross_margin(), 0.9)
        self.assertIsNone(e.max_output_tokens_for_margin(0.999))

    def test_a_free_user_has_no_margin(self):
        with self.assertRaisesRegex(ValueError, "positive price"):
            BASE.with_(price_per_user=0).gross_margin()

    def test_validation(self):
        for changes in [{"price_per_user": -1}, {"cache_hit_rate": 1.2}, {"cache_hit_rate": 0.7, "cache_write_rate": 0.5}, {"retry_rate": -0.1},
                        {"payment_fee_rate": 1.5}]:
            with self.assertRaises(ValueError, msg=str(changes)):
                BASE.with_(**changes)

    def test_sweep_and_immutability(self):
        out = sweep(BASE, "requests_per_user", [100, 200, 400])
        self.assertEqual([v for v, _ in out], [100, 200, 400])
        margins = [m for _, m in out]
        self.assertEqual(margins, sorted(margins, reverse=True))
        BASE.with_(requests_per_user=1)
        self.assertEqual(BASE.requests_per_user, 200)

    def test_exact_reads_the_decimal_you_wrote(self):
        self.assertEqual(exact(0.1) * 3, exact(0.3))
        self.assertNotEqual(0.1 * 3, 0.3)

    def test_to_dict_keeps_required_and_changed_fields(self):
        d = BASE.with_(cache_hit_rate=0.5, cached_input_price_per_1m=0.3).to_dict()
        self.assertEqual(set(d) - {"price_per_user", "requests_per_user", "input_tokens", "output_tokens", "input_price_per_1m",
                                   "output_price_per_1m"}, {"cache_hit_rate", "cached_input_price_per_1m"})


if __name__ == "__main__":
    unittest.main()
