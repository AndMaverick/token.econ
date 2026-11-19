import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from token_econ import Scenario, analyze
from token_econ.cli import main
from token_econ.scenario import to_markdown, to_text

ROOT = Path(__file__).resolve().parent.parent
HEALTHY = ROOT / "examples" / "saas_assistant.json"
SQUEEZED = ROOT / "examples" / "squeezed_assistant.json"
MINIMAL = {"economics": {"price_per_user": 20, "requests_per_user": 200, "input_tokens": 1500, "output_tokens": 500,
                         "input_price_per_1m": 3, "output_price_per_1m": 15}}


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class ScenarioTests(unittest.TestCase):
    def test_minimal_scenario_hand_computed(self):
        a = analyze(Scenario.from_dict(MINIMAL))
        u = a.sections["unit"]
        self.assertAlmostEqual(u["cost_per_request"], 0.012)
        self.assertAlmostEqual(u["gross_margin"], 0.88)
        self.assertEqual(u["max_requests_per_user_at_target"], 500)  # target 0.7 leaves 20 x 0.3 = 6.00 dollars for inference, at 0.012 each
        self.assertTrue(a.ok)
        self.assertEqual(a.gates, [])

    def test_gates_pass_and_fail(self):
        data = dict(MINIMAL, gates={"min_margin": 0.9})
        a = analyze(Scenario.from_dict(data))
        self.assertFalse(a.ok)
        self.assertEqual((a.gates[0].name, a.gates[0].ok, round(a.gates[0].actual, 2)), ("min_margin", False, 0.88))
        self.assertTrue(analyze(Scenario.from_dict(dict(MINIMAL, gates={"min_margin": 0.85}))).ok)

    def test_healthy_example(self):
        a = analyze(Scenario.load(HEALTHY))
        self.assertTrue(a.ok)
        self.assertEqual({g.name for g in a.gates}, {"min_margin", "min_plan_margin", "max_loss_share", "min_squeeze_month", "min_ltv_cac"})
        self.assertAlmostEqual(a.sections["unit"]["cost_per_request"], 0.013398 * 1.04 * 1.08, places=9)   # worked by hand in the example's notes
        self.assertEqual(a.sections["projection"]["below_target_from_month"], 21)
        self.assertEqual(a.sections["unit"]["breakeven_users"], 979)

    def test_squeezed_example_misses_every_gate(self):
        a = analyze(Scenario.load(SQUEEZED))
        self.assertFalse(a.ok)
        self.assertEqual([g.ok for g in a.gates], [False] * 4)
        self.assertEqual(a.sections["projection"]["below_target_from_month"], 0)
        self.assertTrue(a.sections["projection"]["squeezing"])

    def test_typos_are_errors_not_silence(self):
        for bad, expected in [({"economic": {}}, "unknown key"), (dict(MINIMAL, gates={"min_marjin": 0.5}), "unknown gate"),
                              ({"economics": {"price_per_user": 1, "requests_per_user": 1, "input_tokens": 1, "output_tokens": 1,
                                              "input_price_per_1m": 1, "output_price_per_1m": 1, "cache_hit_ratio": 0.5}}, "cache_hit_ratio"),
                              (dict(MINIMAL, plans=[{"name": "x", "price": 1, "cap": 5}]), "plans[0]"),
                              (dict(MINIMAL, population={"kind": "zipf"}), "population kind"), ({}, "economics: required"),
                              (dict(MINIMAL, cohort={"monthly_churnn": 0.1}), "cohort")]:
            with self.assertRaisesRegex(ValueError, expected.replace("[", r"\[").replace("]", r"\]")):
                Scenario.from_dict(bad)

    def test_fingerprint_ignores_key_order_and_tracks_content(self):
        a = Scenario.from_dict(MINIMAL).fingerprint()
        shuffled = {"economics": dict(reversed(list(MINIMAL["economics"].items())))}
        self.assertEqual(Scenario.from_dict(shuffled).fingerprint(), a)
        changed = {"economics": dict(MINIMAL["economics"], requests_per_user=201)}
        self.assertNotEqual(Scenario.from_dict(changed).fingerprint(), a)
        self.assertRegex(a, r"^[0-9a-f]{64}$")

    def test_unreachable_price_target_is_reported_not_raised(self):
        data = {"economics": dict(MINIMAL["economics"], payment_fee_rate=0.4), "target_margin": 0.7}
        u = analyze(Scenario.from_dict(data)).sections["unit"]
        self.assertIsNone(u["price_for_target_margin"])
        self.assertIn("unreachable", u["price_note"])

    def test_text_and_markdown(self):
        a = analyze(Scenario.load(HEALTHY))
        text, md = to_text(a), to_markdown(a)
        self.assertIn("PASS", text)
        self.assertIn(a.fingerprint[:16], text)
        for part in ["## Plans", "## What moves the margin most", "## Gates", "PASS"]:
            self.assertIn(part, md)
        self.assertNotIn("\u2014", text + md)


class CliTests(unittest.TestCase):
    def test_analyze_exit_codes(self):
        self.assertEqual(run("analyze", str(HEALTHY))[0], 0)
        code, out, _ = run("analyze", str(SQUEEZED))
        self.assertEqual(code, 1)
        self.assertIn("FAIL", out)
        self.assertEqual(run("analyze", str(SQUEEZED), "--fail-on", "never")[0], 0)

    def test_json_and_markdown(self):
        data = json.loads(run("analyze", str(HEALTHY), "--format", "json")[1])
        self.assertTrue(data["ok"])
        self.assertEqual(data["tool"], "token.econ")
        self.assertIn("## Gates", run("analyze", str(HEALTHY), "--format", "markdown")[1])

    def test_project_tornado_sweep_fingerprint(self):
        code, out, _ = run("project", str(HEALTHY), "--months", "3")
        self.assertEqual((code, len(out.strip().splitlines())), (0, 4))
        self.assertIn("price_per_user", run("tornado", str(HEALTHY))[1])
        self.assertEqual(len(run("sweep", str(HEALTHY), "requests_per_user", "100", "200", "400")[1].strip().splitlines()), 3)
        self.assertRegex(run("fingerprint", str(HEALTHY))[1].strip(), r"^[0-9a-f]{64}$")

    def test_bad_input_is_exit_2(self):
        for argv in (["analyze", "/nonexistent.json"], ["sweep", str(HEALTHY), "no_such_field", "1"]):
            code, _, err = run(*argv)
            self.assertEqual(code, 2, argv)
            self.assertTrue(err.startswith("error:"))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp, "bad.json")
            p.write_text("{nope")
            self.assertEqual(run("analyze", str(p))[0], 2)

    def test_output_is_identical_across_processes_and_hash_seeds(self):
        outputs = set()
        for seed in ("1", "2", "77"):
            env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(ROOT))
            proc = subprocess.run([sys.executable, "-m", "token_econ", "analyze", str(HEALTHY), "--format", "json"], capture_output=True,
                                  text=True, env=env, cwd=ROOT)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            outputs.add(proc.stdout)
        self.assertEqual(len(outputs), 1)


if __name__ == "__main__":
    unittest.main()
