"""Engine invariants: percentiles, band shape, ensemble size, ordering."""

from lifesim.config import Config
from lifesim.simulate import Result, pct, simulate
from lifesim.state import OUTCOME_KEYS


def test_pct_basic():
    xs = [0, 1, 2, 3, 4]
    assert pct(xs, 0) == 0
    assert pct(xs, 100) == 4
    assert pct(xs, 50) == 2


def test_pct_interpolates():
    assert abs(pct([0, 10], 25) - 2.5) < 1e-9


def test_pct_unsorted_input():
    assert pct([4, 0, 2, 1, 3], 50) == 2


def test_simulate_n_matches_config():
    cfg = Config(domain="career", uncertainty=0.3, seed=1)
    res = simulate(cfg)
    assert res.n == cfg.n_trajectories()
    assert len(res.finals) == res.n
    assert len(res.fired) == res.n


def test_bands_have_horizon_plus_one_months():
    cfg = Config(domain="love", uncertainty=0.2, seed=1, horizon_months=24)
    res = simulate(cfg)
    for k in OUTCOME_KEYS["love"]:
        assert len(res.bands[k]) == cfg.horizon_months + 1
        for entry in res.bands[k]:
            assert entry["p5"] <= entry["p50"] <= entry["p95"]


def test_result_tracks_domain_keys():
    res = simulate(Config(domain="all", uncertainty=0.1, seed=1))
    assert res.keys == OUTCOME_KEYS["all"]
    assert isinstance(res, Result)
