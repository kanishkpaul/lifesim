"""Event invariants: rates rise with u, surprise index is a valid fraction."""

import random

from lifesim.config import Config
from lifesim.events import apply_events
from lifesim.report import mean_rare_events, surprise_index
from lifesim.simulate import simulate


def _event_count(u: float, trials: int = 4000) -> int:
    cfg = Config(uncertainty=u)
    rng = random.Random(99)
    policy = {"work": 0.4, "love": 0.2, "health": 0.2, "explore": 0.2}
    total = 0
    for _ in range(trials):
        state = {
            "skill": 0.4, "reputation": 0.3, "network": 0.3, "savings": 6.0,
            "health": 0.6, "energy": 0.6, "mood": 0.55, "connection": 0.4,
            "luck": 0.5,
        }
        total += len(apply_events(state, policy, cfg, rng))
    return total


def test_event_rate_rises_with_u():
    lo = _event_count(0.0)
    hi = _event_count(1.0)
    assert hi > lo, (lo, hi)


def test_surprise_index_in_unit_interval():
    for u in (0.0, 0.3, 1.0):
        res = simulate(Config(domain="all", uncertainty=u, seed=2))
        si = surprise_index(res)
        assert 0.0 <= si <= 1.0, (u, si)


def test_mean_rare_events_in_range_and_tracks_horizon():
    # Bounded to [0, 4] (four distinct rare types), and a longer horizon should
    # accumulate at least as many distinct rare types as a short one.
    short = simulate(Config(domain="all", uncertainty=0.5, seed=4, horizon_months=6))
    long = simulate(Config(domain="all", uncertainty=0.5, seed=4, horizon_months=120))
    for res in (short, long):
        assert 0.0 <= mean_rare_events(res) <= 4.0
    assert mean_rare_events(long) >= mean_rare_events(short)


def test_apply_events_returns_list_of_known_names():
    from lifesim.events import EVENTS
    names = {n for n, _, _ in EVENTS}
    cfg = Config(uncertainty=1.0)
    rng = random.Random(1)
    policy = {"work": 0.25, "love": 0.25, "health": 0.25, "explore": 0.25}
    for _ in range(200):
        state = {
            "skill": 0.5, "reputation": 0.5, "network": 0.5, "savings": 6.0,
            "health": 0.5, "energy": 0.5, "mood": 0.5, "connection": 0.5,
            "luck": 0.5,
        }
        fired = apply_events(state, policy, cfg, rng)
        assert set(fired) <= names
