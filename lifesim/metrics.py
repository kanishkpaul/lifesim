"""Concrete life metrics layered on the core coupling graph.

These metrics make the simulator legible in human terms: money, income, time
freedom, burnout, relationship depth, fitness, optionality, and fragility. They
are still coupled to the core state, not independent score tracks.
"""

from __future__ import annotations

import random

from .state import clamp01

_NOISE_K = {
    "income_stability": 0.4,
    "career_leverage": 0.4,
    "schedule_control": 0.5,
    "recovery_time": 0.6,
    "stress": 0.7,
    "burnout_risk": 0.6,
    "sleep_quality": 0.5,
    "recovery_capacity": 0.5,
    "romantic_connection": 0.7,
    "friendship_depth": 0.5,
    "family_support": 0.4,
    "loneliness": 0.6,
    "fitness": 0.5,
    "chronic_health_risk": 0.4,
    "energy_stability": 0.5,
    "annual_income_usd": 45000.0,
    "monthly_burn_usd": 2500.0,
    "cash_usd": 25000.0,
    "debt_usd": 8000.0,
    "investments_usd": 18000.0,
    "work_hours_per_week": 18.0,
    "free_hours_per_week": 18.0,
    "sleep_hours": 1.5,
}


def _noise(rng: random.Random, scale: float, var: str) -> float:
    return rng.gauss(0.0, scale * _NOISE_K[var])


def _toward(old: float, target: float, rate: float, rng: random.Random,
            ns: float, key: str) -> float:
    return old + rate * (target - old) + _noise(rng, ns, key)


def update_life_metrics(s: dict, state: dict, policy: dict, ns: float,
                        rng: random.Random, career_capital: float) -> None:
    """Update concrete money, time, stress, relationship, and fitness metrics."""
    work = policy["work"]
    love = policy["love"]
    health_a = policy["health"]
    explore = policy["explore"]
    financial_stress = 1.0 if state["savings"] < 1.0 else 0.0

    leverage_target = clamp01(
        0.25 * state["skill"] + 0.30 * state["reputation"]
        + 0.25 * state["network"] + 0.20 * explore
    )
    s["career_leverage"] = _toward(
        state["career_leverage"], leverage_target, 0.16, rng, ns, "career_leverage"
    )
    stability_target = clamp01(
        0.20 + 0.30 * state["reputation"] + 0.20 * state["network"]
        + 0.20 * s["career_leverage"] + 0.10 * (1.0 - financial_stress)
    )
    s["income_stability"] = _toward(
        state["income_stability"], stability_target, 0.12, rng, ns, "income_stability"
    )

    target_income = 22000.0 + 210000.0 * (career_capital ** 1.35) \
        * (0.65 + 0.70 * s["career_leverage"]) * (0.75 + 0.50 * state["mood"])
    s["annual_income_usd"] = _toward(
        state["annual_income_usd"], target_income, 0.08, rng, ns, "annual_income_usd"
    )
    target_burn = 2600.0 + 2800.0 * (0.35 + 0.65 * state["stress"]) \
        + 1100.0 * (1.0 - health_a)
    s["monthly_burn_usd"] = _toward(
        state["monthly_burn_usd"], target_burn, 0.05, rng, ns, "monthly_burn_usd"
    )

    after_tax_monthly = 0.72 * s["annual_income_usd"] / 12.0
    surplus = after_tax_monthly - s["monthly_burn_usd"]
    s["cash_usd"] = state["cash_usd"] + surplus + _noise(rng, ns, "cash_usd")
    debt_payment = max(0.0, surplus) * 0.10
    new_debt = state["debt_usd"] - debt_payment + max(0.0, -s["cash_usd"]) * 0.35
    s["debt_usd"] = new_debt + _noise(rng, ns, "debt_usd")
    invest_add = max(0.0, s["cash_usd"] - 6.0 * s["monthly_burn_usd"]) * 0.04
    invest_growth = state["investments_usd"] * (0.002 + 0.004 * s["income_stability"])
    s["investments_usd"] = state["investments_usd"] + invest_add + invest_growth \
        + _noise(rng, ns, "investments_usd")

    work_target = 22.0 + 46.0 * work + 8.0 * financial_stress
    s["work_hours_per_week"] = _toward(
        state["work_hours_per_week"], work_target, 0.25, rng, ns, "work_hours_per_week"
    )
    free_target = 74.0 - 0.75 * s["work_hours_per_week"] + 12.0 * s["schedule_control"]
    s["free_hours_per_week"] = _toward(
        state["free_hours_per_week"], free_target, 0.20, rng, ns, "free_hours_per_week"
    )
    control_target = clamp01(
        0.20 + 0.35 * s["career_leverage"] + 0.20 * s["income_stability"]
        + 0.15 * explore - 0.25 * work
    )
    s["schedule_control"] = _toward(
        state["schedule_control"], control_target, 0.15, rng, ns, "schedule_control"
    )
    stress_target = clamp01(
        0.20 + 0.45 * work + 0.20 * financial_stress
        + 0.20 * (1.0 - state["schedule_control"]) - 0.25 * health_a
    )
    s["stress"] = _toward(state["stress"], stress_target, 0.22, rng, ns, "stress")

    sleepq_target = clamp01(0.65 + 0.25 * health_a - 0.35 * s["stress"] - 0.15 * work)
    s["sleep_quality"] = _toward(
        state["sleep_quality"], sleepq_target, 0.20, rng, ns, "sleep_quality"
    )
    s["sleep_hours"] = _toward(
        state["sleep_hours"], 5.4 + 2.6 * s["sleep_quality"] - 0.9 * s["stress"],
        0.18, rng, ns, "sleep_hours"
    )
    s["recovery_time"] = _toward(
        state["recovery_time"],
        clamp01(0.25 + 0.40 * health_a + 0.25 * s["sleep_quality"] + 0.10 * love),
        0.18, rng, ns, "recovery_time"
    )
    s["recovery_capacity"] = _toward(
        state["recovery_capacity"],
        clamp01(0.35 * state["health"] + 0.25 * s["fitness"]
                + 0.25 * s["sleep_quality"] + 0.15 * s["recovery_time"]),
        0.16, rng, ns, "recovery_capacity"
    )
    s["burnout_risk"] = _toward(
        state["burnout_risk"],
        clamp01(0.60 * s["stress"] + 0.25 * work + 0.20 * financial_stress
                - 0.45 * s["recovery_capacity"]),
        0.20, rng, ns, "burnout_risk"
    )

    s["romantic_connection"] = _toward(
        state["romantic_connection"],
        clamp01(0.65 * s["connection"] + 0.25 * love + 0.10 * state["mood"]),
        0.16, rng, ns, "romantic_connection"
    )
    s["friendship_depth"] = _toward(
        state["friendship_depth"],
        clamp01(0.35 * love + 0.25 * explore + 0.20 * state["mood"]
                + 0.20 * (1.0 - work)),
        0.14, rng, ns, "friendship_depth"
    )
    s["family_support"] = _toward(
        state["family_support"],
        clamp01(0.55 * state["family_support"] + 0.20 * love
                + 0.15 * state["mood"] + 0.10 * health_a),
        0.08, rng, ns, "family_support"
    )
    s["loneliness"] = _toward(
        state["loneliness"],
        clamp01(1.0 - (0.45 * s["connection"] + 0.25 * s["friendship_depth"]
                       + 0.20 * s["family_support"] + 0.10 * love)),
        0.18, rng, ns, "loneliness"
    )

    s["fitness"] = _toward(
        state["fitness"],
        clamp01(0.40 * state["health"] + 0.35 * health_a + 0.15 * s["sleep_quality"]
                - 0.15 * s["stress"]),
        0.14, rng, ns, "fitness"
    )
    s["chronic_health_risk"] = _toward(
        state["chronic_health_risk"],
        clamp01(0.50 * (1.0 - s["health"]) + 0.25 * s["stress"]
                + 0.20 * s["burnout_risk"] - 0.25 * s["fitness"]),
        0.10, rng, ns, "chronic_health_risk"
    )
    s["energy_stability"] = _toward(
        state["energy_stability"],
        clamp01(0.35 * s["energy"] + 0.25 * s["sleep_quality"]
                + 0.25 * s["health"] - 0.20 * s["stress"]),
        0.18, rng, ns, "energy_stability"
    )
