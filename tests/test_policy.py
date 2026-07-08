"""Policy invariants: allocations renormalize, compare is well-formed & honest."""

from lifesim.config import Config
from lifesim.policy import (
    POLICY_KEYS,
    PRESETS,
    compare,
    get_policy,
    normalize,
    parse_policy,
)
from lifesim.report import cvar_downside, stdev


def _sums_to_one(pol) -> bool:
    return abs(sum(pol.values()) - 1.0) < 1e-9


def test_normalize_sums_to_one():
    pol = normalize({"work": 2, "love": 1, "health": 1, "explore": 0})
    assert _sums_to_one(pol)
    assert set(pol) == set(POLICY_KEYS)


def test_parse_partial_spec_renormalizes():
    pol = parse_policy("work=1,love=1")
    assert _sums_to_one(pol)
    assert abs(pol["work"] - 0.5) < 1e-9
    assert pol["health"] == 0.0


def test_parse_rejects_unknown_key():
    try:
        parse_policy("wrok=0.5")
    except ValueError:
        return
    raise AssertionError("unknown key should raise")


def test_presets_all_normalize():
    for name in PRESETS:
        assert _sums_to_one(get_policy(name))


def test_normalize_rejects_all_zero():
    try:
        normalize({})
    except ValueError:
        return
    raise AssertionError("all-zero allocation should raise")


def test_compare_one_row_per_policy():
    cfg = Config(domain="all", uncertainty=0.3, seed=1)
    names = ["grind", "balanced", "relationship_first"]
    comp = compare(cfg, {n: get_policy(n) for n in names})
    assert comp["policies"] == names
    for k in comp["keys"]:
        for n in names:
            cell = comp["rows"][n][k]
            assert set(cell) == {"median", "spread", "cvar", "regret"}


def test_compare_regret_nonnegative_and_best_is_zero():
    cfg = Config(domain="all", uncertainty=0.3, seed=1)
    names = ["grind", "relationship_first"]
    comp = compare(cfg, {n: get_policy(n) for n in names})
    for k in comp["keys"]:
        regrets = [comp["rows"][n][k]["regret"] for n in names]
        assert all(r >= -1e-9 for r in regrets)
        assert min(regrets) < 1e-9  # some policy owns the best floor (regret 0)


def test_cvar_is_at_most_median_region():
    # Downside CVaR (worst 5% mean) must not exceed the overall mean.
    xs = list(range(100))
    assert cvar_downside(xs) < sum(xs) / len(xs)


def test_stdev_zero_for_constant():
    assert stdev([3.0] * 10) == 0.0


def test_compare_shows_tradeoff_not_dominance():
    # The whole point: work-favoring vs love-favoring must trade the best floor
    # on skill for the best floor on connection — no single winner.
    cfg = Config(domain="all", uncertainty=0.3, seed=1)
    comp = compare(cfg, {n: get_policy(n) for n in ("grind", "relationship_first")})
    skill_best = min(("grind", "relationship_first"),
                     key=lambda n: comp["rows"][n]["skill"]["regret"])
    conn_best = min(("grind", "relationship_first"),
                    key=lambda n: comp["rows"][n]["connection"]["regret"])
    assert skill_best == "grind"
    assert conn_best == "relationship_first"
