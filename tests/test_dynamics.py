"""Dynamics invariants: bounds, reproducibility, fan-widening, and coupling.

The coupling test is the load-bearing one: it proves work actually crowds out
connection, so the tradeoff the tool exists to reveal is mechanical, not cosmetic.
"""

import random
import statistics

from lifesim.config import Config
from lifesim.dynamics import step
from lifesim.state import BOUNDED_VARS, initial_state
from lifesim.simulate import simulate

_WORK_HEAVY = {"work": 0.8, "love": 0.05, "health": 0.1, "explore": 0.05}
_LOVE_HEAVY = {"work": 0.1, "love": 0.7, "health": 0.1, "explore": 0.1}


def test_bounded_vars_stay_in_range_over_long_roll():
    cfg = Config(horizon_months=240, uncertainty=1.0, domain="all", seed=7)
    rng = random.Random(7)
    state = initial_state("all", rng)
    policy = {"work": 0.4, "love": 0.2, "health": 0.2, "explore": 0.2}
    for _ in range(cfg.horizon_months):
        state = step(state, policy, cfg, rng)
        for k in BOUNDED_VARS:
            assert 0.0 <= state[k] <= 1.0, (k, state[k])


def test_fixed_seed_is_byte_identical():
    a = simulate(Config(domain="all", uncertainty=0.3, seed=1))
    b = simulate(Config(domain="all", uncertainty=0.3, seed=1))
    assert a.finals == b.finals
    assert a.bands == b.bands


def test_higher_u_widens_end_state_spread():
    lo = simulate(Config(domain="career", uncertainty=0.1, seed=3, horizon_months=60))
    hi = simulate(Config(domain="career", uncertainty=0.9, seed=3, horizon_months=60))

    def spread(res, key):
        xs = [f[key] for f in res.finals]
        return statistics.pstdev(xs)

    # Averaged across tracked vars, more uncertainty must fan wider.
    lo_spread = sum(spread(lo, k) for k in lo.keys) / len(lo.keys)
    hi_spread = sum(spread(hi, k) for k in hi.keys) / len(hi.keys)
    assert hi_spread > lo_spread, (lo_spread, hi_spread)


def test_work_crowds_out_connection():
    # Same seed, same everything except the allocation. Work-heavy must buy more
    # skill AND less connection than love-heavy: the career-vs-love tension.
    seed, u = 11, 0.3
    work = simulate(Config(domain="all", uncertainty=u, seed=seed), _WORK_HEAVY)
    love = simulate(Config(domain="all", uncertainty=u, seed=seed), _LOVE_HEAVY)

    def median(res, key):
        return statistics.median(f[key] for f in res.finals)

    assert median(work, "skill") > median(love, "skill")
    assert median(work, "connection") < median(love, "connection")


def test_step_does_not_mutate_input():
    cfg = Config(seed=5)
    rng = random.Random(5)
    state = initial_state("all", rng)
    snapshot = dict(state)
    step(state, {"work": 0.4, "love": 0.2, "health": 0.2, "explore": 0.2}, cfg, rng)
    assert state == snapshot, "step mutated its input state"
