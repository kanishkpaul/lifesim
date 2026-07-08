"""Precision diagnostics: sampling error without fake certainty."""

from lifesim.config import Config
from lifesim.policy import DEFAULT_POLICY
from lifesim.precision import (
    convergence_check,
    monte_carlo_rows,
    precision_table,
    scaled_sample_config,
)
from lifesim.simulate import simulate


def _small_result():
    cfg = Config(
        domain="career",
        horizon_months=4,
        uncertainty=0.4,
        n_min=30,
        n_max=30,
        seed=7,
    )
    return simulate(cfg, DEFAULT_POLICY)


def test_monte_carlo_rows_cover_all_keys():
    res = _small_result()
    rows = monte_carlo_rows(res)
    assert set(rows) == set(res.keys)
    for row in rows.values():
        assert row["mean_se"] >= 0.0
        assert row["split_max_delta"] >= 0.0


def test_precision_table_names_sampling_error():
    table = precision_table(_small_result())
    assert "MONTE CARLO PRECISION" in table
    assert "MC SE" in table


def test_scaled_sample_config_preserves_u_and_scales_n():
    cfg = Config(uncertainty=0.5, n_min=10, n_max=20)
    bigger = scaled_sample_config(cfg, 3.0)
    assert bigger.u() == cfg.u()
    assert bigger.n_trajectories() == cfg.n_trajectories() * 3


def test_convergence_check_uses_larger_sample_budget():
    res = _small_result()
    check = convergence_check(res, DEFAULT_POLICY, factor=2.0)
    assert check["base_n"] == res.n
    assert check["bigger_n"] > check["base_n"]
    assert set(check["rows"]) == set(res.keys)
