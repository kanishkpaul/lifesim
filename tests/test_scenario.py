"""Scenario invariants: lossless round-trip, and --from-state skips jitter."""

import json
import os
import random
import tempfile

from lifesim.config import Config
from lifesim.scenario import (
    Scenario,
    load_scenario,
    load_state,
    save_scenario,
)
from lifesim.simulate import simulate
from lifesim.state import initial_state


def _tmp(name: str) -> str:
    return os.path.join(tempfile.mkdtemp(prefix="lifesim_"), name)


def test_round_trip_lossless():
    scen = Scenario(
        domain="career",
        horizon_months=36,
        uncertainty=0.42,
        overrides={"skill": 0.6, "savings": 12.0},
        policy={"work": 0.5, "love": 0.1, "health": 0.2, "explore": 0.2},
    )
    path = _tmp("s.json")
    save_scenario(scen, path)
    back = load_scenario(path)
    assert back == scen


def test_load_rejects_unknown_field():
    path = _tmp("bad.json")
    with open(path, "w") as fh:
        json.dump({"domain": "all", "banana": 1}, fh)
    try:
        load_scenario(path)
    except ValueError:
        return
    raise AssertionError("unknown field should raise")


def test_load_rejects_unknown_override_var():
    path = _tmp("bad2.json")
    with open(path, "w") as fh:
        json.dump({"overrides": {"charisma": 0.9}}, fh)
    try:
        load_scenario(path)
    except ValueError:
        return
    raise AssertionError("unknown override var should raise")


def test_from_state_skips_jitter():
    # With jitter off, the start is taken verbatim: identical across RNG states.
    ov = {"skill": 0.6, "connection": 0.3}
    a = initial_state("all", random.Random(1), overrides=ov, jitter=False)
    b = initial_state("all", random.Random(999), overrides=ov, jitter=False)
    assert a == b
    assert a["skill"] == 0.6 and a["connection"] == 0.3


def test_jitter_on_perturbs_start():
    ov = {"skill": 0.6}
    a = initial_state("all", random.Random(1), overrides=ov, jitter=True)
    assert a["skill"] != 0.6  # jitter moved it


def test_from_state_gives_zero_width_month0_band():
    # Every trajectory starts at the same observed point => no month-0 spread.
    cfg = Config(domain="all", uncertainty=0.3, seed=1, horizon_months=12)
    state = {"skill": 0.5, "reputation": 0.4, "network": 0.4, "savings": 5.0,
             "health": 0.6, "energy": 0.6, "mood": 0.5, "connection": 0.4,
             "luck": 0.5}
    res = simulate(cfg, overrides=state, jitter=False)
    for k in res.keys:
        b0 = res.bands[k][0]
        assert b0["p5"] == b0["p95"], (k, b0)


def test_load_state_accepts_wrapper():
    path = _tmp("st.json")
    with open(path, "w") as fh:
        json.dump({"state": {"skill": 0.7}}, fh)
    st = load_state(path)
    assert st == {"skill": 0.7}


def test_example_files_load():
    here = os.path.dirname(os.path.dirname(__file__))
    for fn in ("default.json", "scenario_template.json"):
        load_scenario(os.path.join(here, "examples", fn))
