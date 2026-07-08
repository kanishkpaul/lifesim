"""The state vector: what a "life" is, numerically.

Continuous variables live in [0, 1]. `savings` is the one unbounded variable
(months of runway; may go negative = debt). `luck` is latent and internal — it
mean-reverts and only modulates event odds; the report never shows it directly.

Every trajectory starts from the same defaults plus small per-trajectory jitter,
so the fan has somewhere to open from — identical starts would understate near-
term spread. Scenario overrides replace defaults BEFORE jitter.
"""

from __future__ import annotations

import random

# Bounded variables are clamped to [0, 1]. savings is unbounded (documented).
BOUNDED_VARS = (
    "skill",
    "reputation",
    "network",
    "health",
    "energy",
    "mood",
    "connection",
    "luck",
)
UNBOUNDED_VARS = ("savings",)
ALL_VARS = BOUNDED_VARS + UNBOUNDED_VARS

# Generic starting person. Modest, mid-range; nobody starts saturated.
DEFAULTS: dict[str, float] = {
    "skill": 0.35,
    "reputation": 0.25,
    "network": 0.30,
    "savings": 6.0,       # six months of runway
    "health": 0.65,
    "energy": 0.60,
    "mood": 0.55,
    "connection": 0.40,
    "luck": 0.50,         # latent, mean-reverts to 0.5
}

# Which variables the report tracks per domain. Multi-objective: we keep the
# whole relevant vector, never a collapsed scalar (CLAUDE.md #5).
OUTCOME_KEYS: dict[str, list[str]] = {
    "career": ["skill", "reputation", "network", "savings"],
    "love": ["connection", "health", "energy", "mood"],
    "all": ["skill", "reputation", "network", "savings",
            "health", "energy", "mood", "connection"],
}

_JITTER_SD = 0.03


def clamp01(x: float) -> float:
    """Clamp a bounded variable into [0, 1]."""
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def clamp_state(state: dict[str, float]) -> dict[str, float]:
    """Clamp every bounded variable in-place-safe; leave unbounded ones alone.

    Returns the same dict for convenience. Called at the end of every dynamics
    step so no variable silently drifts out of range.
    """
    for k in BOUNDED_VARS:
        state[k] = clamp01(state[k])
    return state


def initial_state(
    domain: str,
    rng: random.Random,
    overrides: dict[str, float] | None = None,
    jitter: bool = True,
) -> dict[str, float]:
    """Build one trajectory's starting state.

    Args:
        domain: validated by config; used only for a friendly error here.
        rng: the single seeded RNG; all jitter is drawn from it.
        overrides: scenario values that REPLACE defaults before jitter.
        jitter: apply small N(0, 0.03) noise to bounded vars. Turned OFF for
            `--from-state` resumes, where the observed state is taken as given.

    Returns a fresh dict with every variable in ALL_VARS.
    """
    state = dict(DEFAULTS)
    if overrides:
        unknown = set(overrides) - set(ALL_VARS)
        if unknown:
            raise ValueError(f"unknown state variables in overrides: {sorted(unknown)}")
        state.update(overrides)

    if jitter:
        for k in BOUNDED_VARS:
            state[k] += rng.gauss(0.0, _JITTER_SD)
        # savings gets a proportionally larger, still-modest jitter (months).
        state["savings"] += rng.gauss(0.0, 0.5)

    return clamp_state(state)
