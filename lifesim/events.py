"""Discrete stochastic events — the fat tails.

Per-step Gaussian noise alone gives a boring bell curve. Real lives are shaped by
rare discrete shocks: a breakthrough, a layoff, meeting someone, a health scare,
a windfall. Those are what put mass in the tails, and modeling them is the whole
reason this tool isn't a normal distribution with extra steps (CLAUDE.md #3).

This is the ONE module allowed to mutate state in place: `apply_events` edits the
freshly-built next-state dict that `dynamics.step` hands it. Base monthly event
rate scales with `u` (more uncertainty => more surprises); each event carries its
own multiplier keyed to state (skill, luck, connection) or policy (love).

Events are a list of `(name, prob_fn, effect_fn)`, so adding one is a one-line
append. `prob_fn(state, policy) -> multiplier` on top of the base rate;
`effect_fn(state, rng)` applies the shock. `RARE_EVENTS` marks which count toward
the report's surprise index.
"""

from __future__ import annotations

import random

from .state import clamp01


def _base_rate(u: float) -> float:
    """Monthly probability floor for any single event before its multiplier."""
    return 0.01 + 0.05 * u


# --- effect helpers: nudge a bounded var, keeping it in range immediately. ---
def _bump(state: dict, key: str, delta: float) -> None:
    state[key] = clamp01(state.get(key, 0.0) + delta)


def _add(state: dict, key: str, delta: float) -> None:
    state[key] = state.get(key, 0.0) + delta


# --- probability multipliers (keyed to state/policy). Return a scalar >= 0. ---
def _p_breakthrough(state: dict, policy: dict) -> float:
    # Skill and luck compound your odds of a break.
    return 1.0 + 3.0 * state["skill"] * state["luck"]


def _p_layoff(state: dict, policy: dict) -> float:
    # Bad luck (low luck) raises the odds; good luck suppresses them.
    return 1.0 + 2.5 * (1.0 - state["luck"])


def _p_meet_someone(state: dict, policy: dict) -> float:
    # You cannot meet someone you never make room for.
    if policy["love"] <= 0.05:
        return 0.0
    return 1.0 + 2.0 * policy["love"] * (1.0 - state["connection"])


def _p_breakup(state: dict, policy: dict) -> float:
    # Neglected connection drifts. High existing connection with attention is safe.
    return 0.5 + 2.0 * (1.0 - state["connection"]) * (1.0 - policy["love"])


def _p_health_shock(state: dict, policy: dict) -> float:
    # Low health and bad luck together are the danger zone.
    return 0.5 + 2.0 * (1.0 - state["health"]) * (1.0 - state["luck"])


def _p_windfall(state: dict, policy: dict) -> float:
    # Rare pure upside, mildly luck-sensitive. Kept small on purpose.
    return 0.4 * (0.5 + state["luck"])


# --- effects (mutate in place). ---
def _e_breakthrough(state: dict, rng: random.Random) -> None:
    _bump(state, "reputation", 0.12 + 0.05 * rng.random())
    _bump(state, "career_leverage", 0.08)
    _bump(state, "income_stability", 0.05)
    _add(state, "annual_income_usd", 5000.0 + 15000.0 * rng.random())
    _add(state, "cash_usd", 1000.0 + 3000.0 * rng.random())
    state["savings"] += 2.0 + 3.0 * rng.random()
    _bump(state, "mood", 0.10)


def _e_layoff(state: dict, rng: random.Random) -> None:
    _bump(state, "income_stability", -0.14)
    _bump(state, "career_leverage", -0.06)
    _bump(state, "stress", 0.18)
    _bump(state, "burnout_risk", 0.08)
    _add(state, "debt_usd", 1000.0 + 3000.0 * rng.random())
    _add(state, "cash_usd", -(1000.0 + 3000.0 * rng.random()))
    state["savings"] -= 2.0 + 2.0 * rng.random()
    _bump(state, "mood", -0.12)


def _e_meet_someone(state: dict, rng: random.Random) -> None:
    _bump(state, "connection", 0.15 + 0.10 * rng.random())
    _bump(state, "romantic_connection", 0.15 + 0.10 * rng.random())
    _bump(state, "friendship_depth", 0.05)
    _bump(state, "loneliness", -0.12)
    _bump(state, "mood", 0.10)


def _e_breakup(state: dict, rng: random.Random) -> None:
    _bump(state, "connection", -(0.15 + 0.15 * rng.random()))
    _bump(state, "romantic_connection", -(0.20 + 0.15 * rng.random()))
    _bump(state, "loneliness", 0.15)
    _bump(state, "stress", 0.10)
    _bump(state, "mood", -0.12)


def _e_health_shock(state: dict, rng: random.Random) -> None:
    _bump(state, "health", -(0.15 + 0.15 * rng.random()))
    _bump(state, "energy", -(0.10 + 0.10 * rng.random()))
    _bump(state, "fitness", -(0.10 + 0.10 * rng.random()))
    _bump(state, "sleep_quality", -0.08)
    _bump(state, "chronic_health_risk", 0.12)
    _bump(state, "burnout_risk", 0.08)


def _e_windfall(state: dict, rng: random.Random) -> None:
    cash = 4.0 + 5.0 * rng.random()
    state["savings"] += cash
    burn = state.get("monthly_burn_usd", 5000.0)
    _add(state, "cash_usd", cash * burn)
    _add(state, "investments_usd", 0.25 * cash * burn)
    _bump(state, "network", 0.08)
    _bump(state, "optionality", 0.08)


# (name, prob_fn, effect_fn). Append a one-liner to add an event.
EVENTS: list[tuple] = [
    ("breakthrough", _p_breakthrough, _e_breakthrough),
    ("layoff", _p_layoff, _e_layoff),
    ("meet_someone", _p_meet_someone, _e_meet_someone),
    ("breakup", _p_breakup, _e_breakup),
    ("health_shock", _p_health_shock, _e_health_shock),
    ("windfall", _p_windfall, _e_windfall),
]

# Events that count as "surprising" for the report's surprise index. Meeting
# someone / a routine drift are ordinary life; the rest are the fat-tail shocks.
RARE_EVENTS: frozenset[str] = frozenset(
    {"breakthrough", "layoff", "health_shock", "windfall"}
)


def apply_events(state: dict, policy: dict, cfg, rng: random.Random) -> list[str]:
    """Roll each event for this month, mutating `state`, and return names fired.

    Each event fires independently with probability `base_rate(u) * multiplier`,
    clamped to [0, 1]. Multiple events can fire in the same month. The returned
    list lets the trajectory accumulate a per-life fired-event record, which the
    report turns into the surprise index.
    """
    u = cfg.u()
    base = _base_rate(u) * cfg.event_rate_scale
    fired: list[str] = []
    for name, prob_fn, effect_fn in EVENTS:
        p = base * prob_fn(state, policy)
        if p <= 0.0:
            continue
        if p > 1.0:
            p = 1.0
        if rng.random() < p:
            effect_fn(state, rng)
            fired.append(name)
    return fired
