"""Bell-curve rarity stays explicit, deterministic, and per-variable."""

from lifesim.config import Config
from lifesim.rarity import normal_cdf, rarity_for_value, rarity_rows, rarity_table
from lifesim.report import summarize
from lifesim.simulate import simulate


def _result():
    cfg = Config(
        domain="career",
        horizon_months=3,
        uncertainty=0.2,
        n_min=20,
        n_max=20,
        seed=5,
    )
    return simulate(cfg)


def test_normal_cdf_midpoint_is_half():
    assert abs(normal_cdf(10.0, mean=10.0, sd=2.0) - 0.5) < 1e-12


def test_normal_cdf_rejects_nonpositive_sd():
    try:
        normal_cdf(1.0, mean=0.0, sd=0.0)
    except ValueError:
        return
    raise AssertionError("sd=0 should be rejected")


def test_rarity_for_value_reports_percentages():
    row = rarity_for_value("skill", 0.45)
    assert abs(row["world_percentile"] - 50.0) < 1e-9
    assert abs(row["same_or_better"] - 50.0) < 1e-9
    assert abs(row["top_percent"] - row["same_or_better"]) < 1e-12
    assert row["top_percent_precise"] == "50"


def test_rarity_rows_cover_current_and_horizon_quantiles():
    res = _result()
    rows = rarity_rows(res)
    assert len(rows) == len(res.keys) * 4
    assert {r["case"] for r in rows} == {"current", "p5", "median", "p95"}
    assert {r["variable"] for r in rows} == set(res.keys)


def test_rarity_table_and_summary_include_caveat():
    res = _result()
    table = rarity_table(res)
    assert "BELL-CURVE WORLD RARITY" in table
    assert "top %" in table
    assert "no life score" in table
    assert "BELL-CURVE WORLD RARITY" in summarize(res)
