"""The state vector: what a "life" is, numerically.

Bounded variables live in [0, 1]. Money, hours, and runway are documented
non-bounded variables. `luck` is latent and internal — it mean-reverts and only
modulates event odds; the report never shows it directly.

Every trajectory starts from defaults plus small per-trajectory jitter, so the
fan has somewhere to open from. Scenario overrides replace defaults before
jitter. Derived diagnostics such as optionality are recomputed from the raw
state, not evolved as independent random walks.
"""

from __future__ import annotations

import random

BOUNDED_VARS = (
    "skill", "reputation", "network", "health", "energy", "mood",
    "connection", "luck", "income_stability", "career_leverage",
    "schedule_control", "recovery_time", "stress", "burnout_risk",
    "sleep_quality", "recovery_capacity", "romantic_connection",
    "friendship_depth", "family_support", "loneliness", "fitness",
    "chronic_health_risk", "energy_stability", "optionality",
    "luck_surface_area", "downside_fragility", "bottleneck_pressure",
)

# May go negative. `savings` is months of runway; `net_worth_usd` is assets-debt.
UNBOUNDED_VARS = ("savings", "net_worth_usd")

# Not capped above, but negative values are nonsensical.
NONNEGATIVE_VARS = (
    "cash_usd", "debt_usd", "investments_usd", "annual_income_usd",
    "monthly_burn_usd", "work_hours_per_week", "free_hours_per_week",
    "sleep_hours",
)
ALL_VARS = BOUNDED_VARS + UNBOUNDED_VARS + NONNEGATIVE_VARS

LOWER_IS_BETTER = frozenset(
    {
        "debt_usd", "monthly_burn_usd", "work_hours_per_week", "stress",
        "burnout_risk", "loneliness", "chronic_health_risk",
        "downside_fragility", "bottleneck_pressure",
    }
)
HIGHER_IS_BETTER: dict[str, bool] = {
    k: k not in LOWER_IS_BETTER for k in ALL_VARS if k != "luck"
}

DEFAULTS: dict[str, float] = {
    "skill": 0.35,
    "reputation": 0.25,
    "network": 0.30,
    "savings": 6.0,
    "health": 0.65,
    "energy": 0.60,
    "mood": 0.55,
    "connection": 0.40,
    "luck": 0.50,
    "cash_usd": 30000.0,
    "debt_usd": 5000.0,
    "investments_usd": 10000.0,
    "net_worth_usd": 35000.0,
    "annual_income_usd": 60000.0,
    "monthly_burn_usd": 5000.0,
    "income_stability": 0.45,
    "career_leverage": 0.35,
    "work_hours_per_week": 42.0,
    "free_hours_per_week": 38.0,
    "schedule_control": 0.35,
    "recovery_time": 0.45,
    "stress": 0.45,
    "burnout_risk": 0.35,
    "sleep_quality": 0.55,
    "recovery_capacity": 0.55,
    "romantic_connection": 0.30,
    "friendship_depth": 0.40,
    "family_support": 0.45,
    "loneliness": 0.45,
    "fitness": 0.50,
    "chronic_health_risk": 0.30,
    "energy_stability": 0.50,
    "sleep_hours": 7.0,
    "optionality": 0.40,
    "luck_surface_area": 0.35,
    "downside_fragility": 0.45,
    "bottleneck_pressure": 0.50,
}

OUTCOME_KEYS: dict[str, list[str]] = {
    "career": [
        "skill", "reputation", "network", "savings", "cash_usd", "debt_usd",
        "investments_usd", "net_worth_usd", "annual_income_usd",
        "income_stability", "career_leverage", "work_hours_per_week",
        "schedule_control", "optionality", "luck_surface_area",
    ],
    "love": [
        "connection", "romantic_connection", "friendship_depth",
        "family_support", "loneliness", "health", "energy", "mood",
        "free_hours_per_week", "stress", "burnout_risk", "sleep_quality",
        "recovery_time",
    ],
    "all": [
        "skill", "reputation", "network", "savings", "cash_usd", "debt_usd",
        "investments_usd", "net_worth_usd", "annual_income_usd",
        "monthly_burn_usd", "income_stability", "career_leverage",
        "work_hours_per_week", "free_hours_per_week", "schedule_control",
        "recovery_time", "stress", "burnout_risk", "sleep_quality",
        "recovery_capacity", "health", "fitness", "chronic_health_risk",
        "energy", "energy_stability", "sleep_hours", "mood", "connection",
        "romantic_connection", "friendship_depth", "family_support",
        "loneliness", "optionality", "luck_surface_area",
        "downside_fragility", "bottleneck_pressure",
    ],
}

_JITTER_SD = 0.03
_NONNEGATIVE_JITTER = {
    "cash_usd": 3000.0,
    "debt_usd": 1000.0,
    "investments_usd": 2500.0,
    "annual_income_usd": 5000.0,
    "monthly_burn_usd": 400.0,
    "work_hours_per_week": 3.0,
    "free_hours_per_week": 3.0,
    "sleep_hours": 0.35,
}


def clamp01(x: float) -> float:
    """Clamp a bounded variable into [0, 1]."""
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def clamp_state(state: dict[str, float]) -> dict[str, float]:
    """Clamp bounded vars and floor nonnegative vars; leave runway/net worth free."""
    for k in BOUNDED_VARS:
        state[k] = clamp01(state[k])
    for k in NONNEGATIVE_VARS:
        state[k] = max(0.0, state[k])
    return state


def refresh_derived_metrics(state: dict[str, float]) -> dict[str, float]:
    """Recompute coupled summary diagnostics from the raw state.

    These are not independent life forces. They summarize the rest of the vector
    so the report can talk about optionality, luck surface area, fragility, and
    bottlenecks without inventing a single aggregate life score.
    """
    burn = max(1.0, state["monthly_burn_usd"])
    state["savings"] = state["cash_usd"] / burn
    state["net_worth_usd"] = (
        state["cash_usd"] + state["investments_usd"] - state["debt_usd"]
    )
    runway = clamp01(state["savings"] / 24.0)
    debt_load = clamp01(state["debt_usd"] / 50000.0)
    free_time = clamp01(state["free_hours_per_week"] / 60.0)
    state["optionality"] = clamp01(
        0.25 * runway + 0.20 * state["income_stability"]
        + 0.18 * state["network"] + 0.14 * state["schedule_control"]
        + 0.13 * state["health"] + 0.10 * free_time
    )
    state["luck_surface_area"] = clamp01(
        0.30 * state["network"] + 0.25 * state["reputation"]
        + 0.15 * state["energy"] + 0.15 * state["schedule_control"]
        + 0.15 * state["career_leverage"]
    )
    state["downside_fragility"] = clamp01(
        0.22 * (1.0 - clamp01(state["savings"] / 6.0))
        + 0.20 * state["burnout_risk"] + 0.16 * state["loneliness"]
        + 0.16 * state["chronic_health_risk"]
        + 0.14 * (1.0 - state["income_stability"]) + 0.12 * debt_load
    )
    supports = (
        state["energy"], state["health"], state["connection"],
        state["income_stability"], 1.0 - state["downside_fragility"],
    )
    state["bottleneck_pressure"] = clamp01(1.0 - min(supports))
    return clamp_state(state)


def initial_state(
    domain: str,
    rng: random.Random,
    overrides: dict[str, float] | None = None,
    jitter: bool = True,
) -> dict[str, float]:
    """Build one trajectory's starting state.

    `domain` is validated by config; it is kept here for a friendly API. Turning
    jitter off is what makes `--from-state` an observed state rather than a guess.
    """
    state = dict(DEFAULTS)
    if overrides:
        unknown = set(overrides) - set(ALL_VARS)
        if unknown:
            raise ValueError(f"unknown state variables in overrides: {sorted(unknown)}")
        state.update(overrides)
        if "savings" in overrides and "cash_usd" not in overrides:
            state["cash_usd"] = max(0.0, state["savings"] * state["monthly_burn_usd"])
        if "cash_usd" in overrides and "savings" not in overrides:
            state["savings"] = state["cash_usd"] / max(1.0, state["monthly_burn_usd"])

    if jitter:
        for k in BOUNDED_VARS:
            state[k] += rng.gauss(0.0, _JITTER_SD)
        state["savings"] += rng.gauss(0.0, 0.5)
        for k, sd in _NONNEGATIVE_JITTER.items():
            state[k] += rng.gauss(0.0, sd)

    return refresh_derived_metrics(clamp_state(state))
