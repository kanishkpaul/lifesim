"""Phase 4 invariants: sweep parsing/behavior and the plot guard.

These are optional-tooling tests; they must not require matplotlib to run.
"""

import os
import tempfile

from lifesim.config import Config
from lifesim.plot import has_matplotlib, plot_bands
from lifesim.policy import DEFAULT_POLICY
from lifesim.simulate import simulate
from lifesim.sweep import parse_sweep, run_sweep


def test_parse_sweep_inclusive_endpoints():
    param, vals = parse_sweep("u=0:1:0.1")
    assert param == "u"
    assert vals[0] == 0.0 and vals[-1] == 1.0
    assert len(vals) == 11


def test_parse_sweep_rejects_unsweepable():
    try:
        parse_sweep("charisma=0:1:0.1")
    except ValueError:
        return
    raise AssertionError("non-sweepable param should raise")


def test_parse_sweep_rejects_bad_step():
    try:
        parse_sweep("u=0:1:0")
    except ValueError:
        return
    raise AssertionError("zero step should raise")


def test_u_sweep_n_grows_with_u():
    cfg = Config(domain="career", seed=1, horizon_months=12)
    rows = run_sweep(cfg, DEFAULT_POLICY, "u", [0.0, 0.5, 1.0])
    ns = [r["n"] for r in rows]
    assert ns == sorted(ns) and ns[0] < ns[-1], ns


def test_u_sweep_widens_cone():
    # Turning up u must (on average) widen the outcome cone — the honesty knob.
    cfg = Config(domain="career", seed=1, horizon_months=36)
    rows = run_sweep(cfg, DEFAULT_POLICY, "u", [0.1, 0.9])
    keys = rows[0]["keys"]
    lo = sum(rows[0]["outcomes"][k]["cone"] for k in keys) / len(keys)
    hi = sum(rows[1]["outcomes"][k]["cone"] for k in keys) / len(keys)
    assert hi > lo, (lo, hi)


def test_work_sweep_lifts_skill_median():
    # Sweeping the work weight up should raise skill's median monotone-ish.
    cfg = Config(domain="all", seed=1, uncertainty=0.2)
    rows = run_sweep(cfg, DEFAULT_POLICY, "work", [0.1, 0.9])
    assert rows[1]["outcomes"]["skill"]["p50"] > rows[0]["outcomes"]["skill"]["p50"]


def test_sweep_keeps_outcomes_separate():
    # No collapsed scalar: every row carries per-variable outcomes.
    cfg = Config(domain="all", seed=1)
    rows = run_sweep(cfg, DEFAULT_POLICY, "u", [0.2])
    assert set(rows[0]["outcomes"]) == set(rows[0]["keys"])


def test_plot_guard_is_clean():
    # plot.py must import on a bare interpreter; only calling it may fail, and
    # then with a clear RuntimeError, never a raw ImportError.
    res = simulate(Config(domain="career", seed=1, horizon_months=6))
    path = os.path.join(tempfile.mkdtemp(prefix="lifesim_"), "fan.png")
    if has_matplotlib():
        assert plot_bands(res, path) == path
        assert os.path.exists(path)
    else:
        try:
            plot_bands(res, path)
        except RuntimeError:
            return
        raise AssertionError("missing matplotlib should raise RuntimeError")
