"""Bell-curve world-rarity reference.

This module answers a narrow question: "where would this outcome sit in a
simple bell-curve reference population?" The arithmetic is precise, but the
reference curves are intentionally coarse modeling assumptions, not census
facts. Rarity stays per outcome variable so the core never collapses a life into
one hidden score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .simulate import Result
from .state import HIGHER_IS_BETTER


@dataclass(frozen=True)
class WorldReference:
    """A normal reference curve for one outcome variable.

    `mean` and `sd` live on the same scale as the state variable. For bounded
    variables that is [0, 1]; for savings it is months of runway. Higher is
    better for the current state vector, but the flag keeps the table honest if
    a future variable flips direction.
    """

    mean: float
    sd: float
    higher_is_better: bool = True


# Coarse bell-curve references for the toy state variables. These are not world
# measurements; they are transparent priors so a "rarity" percentage is computed
# from visible assumptions rather than hidden vibes.
WORLD_REFERENCES: dict[str, WorldReference] = {
    "skill": WorldReference(mean=0.45, sd=0.18),
    "reputation": WorldReference(mean=0.35, sd=0.20),
    "network": WorldReference(mean=0.40, sd=0.20),
    "savings": WorldReference(mean=4.0, sd=8.0),
    "cash_usd": WorldReference(mean=18000.0, sd=35000.0),
    "debt_usd": WorldReference(mean=12000.0, sd=18000.0, higher_is_better=False),
    "investments_usd": WorldReference(mean=45000.0, sd=120000.0),
    "net_worth_usd": WorldReference(mean=60000.0, sd=180000.0),
    "annual_income_usd": WorldReference(mean=22000.0, sd=35000.0),
    "monthly_burn_usd": WorldReference(
        mean=1800.0, sd=1600.0, higher_is_better=False
    ),
    "income_stability": WorldReference(mean=0.45, sd=0.20),
    "career_leverage": WorldReference(mean=0.35, sd=0.20),
    "work_hours_per_week": WorldReference(
        mean=42.0, sd=15.0, higher_is_better=False
    ),
    "free_hours_per_week": WorldReference(mean=38.0, sd=15.0),
    "schedule_control": WorldReference(mean=0.40, sd=0.22),
    "recovery_time": WorldReference(mean=0.45, sd=0.20),
    "stress": WorldReference(mean=0.50, sd=0.20, higher_is_better=False),
    "burnout_risk": WorldReference(mean=0.35, sd=0.20, higher_is_better=False),
    "sleep_quality": WorldReference(mean=0.55, sd=0.18),
    "recovery_capacity": WorldReference(mean=0.55, sd=0.18),
    "health": WorldReference(mean=0.60, sd=0.18),
    "fitness": WorldReference(mean=0.50, sd=0.20),
    "chronic_health_risk": WorldReference(
        mean=0.30, sd=0.18, higher_is_better=False
    ),
    "energy": WorldReference(mean=0.55, sd=0.17),
    "energy_stability": WorldReference(mean=0.50, sd=0.18),
    "sleep_hours": WorldReference(mean=7.0, sd=1.2),
    "mood": WorldReference(mean=0.52, sd=0.16),
    "connection": WorldReference(mean=0.50, sd=0.20),
    "romantic_connection": WorldReference(mean=0.45, sd=0.22),
    "friendship_depth": WorldReference(mean=0.48, sd=0.20),
    "family_support": WorldReference(mean=0.50, sd=0.20),
    "loneliness": WorldReference(mean=0.42, sd=0.22, higher_is_better=False),
    "optionality": WorldReference(mean=0.42, sd=0.20),
    "luck_surface_area": WorldReference(mean=0.38, sd=0.20),
    "downside_fragility": WorldReference(
        mean=0.45, sd=0.20, higher_is_better=False
    ),
    "bottleneck_pressure": WorldReference(
        mean=0.50, sd=0.20, higher_is_better=False
    ),
}

for _key, _ref in WORLD_REFERENCES.items():
    expected = HIGHER_IS_BETTER.get(_key, True)
    if _ref.higher_is_better != expected:
        raise RuntimeError(f"rarity direction mismatch for {_key}")


def normal_cdf(x: float, mean: float, sd: float) -> float:
    """Normal CDF in [0, 1], using `erfc` so lower tails keep precision."""
    if sd <= 0:
        raise ValueError("standard deviation must be > 0")
    z = (x - mean) / sd
    p = 0.5 * math.erfc(-z / 2.0 ** 0.5)
    return 0.0 if p < 0.0 else 1.0 if p > 1.0 else p


def normal_sf(x: float, mean: float, sd: float) -> float:
    """Normal survival function, kept direct to avoid `1 - cdf` cancellation."""
    if sd <= 0:
        raise ValueError("standard deviation must be > 0")
    z = (x - mean) / sd
    p = 0.5 * math.erfc(z / 2.0 ** 0.5)
    return 0.0 if p < 0.0 else 1.0 if p > 1.0 else p


def _pct(p: float) -> float:
    return 100.0 * p


def _fmt_pct(p: float) -> str:
    return f"{p:0.2f}%"


def _precise_number(x: float) -> str:
    """Shortest decimal that preserves the computed float value."""
    return format(x, ".17g")


def _fmt_pct_precise(p: float) -> str:
    return f"{_precise_number(p)}%"


def rarity_for_value(key: str, value: float) -> dict[str, float | str]:
    """Return percentile and same-or-better share for one variable value.

    `world_percentile` means the percentage of the bell-curve reference
    population at or below the value. `same_or_better` is the percentage at least
    as good as that value for higher-is-better variables. `top_percent` is the
    same quantity under a clearer report/data label; the paired precise string
    keeps all meaningful float digits instead of rounding for display.
    """
    if key not in WORLD_REFERENCES:
        raise KeyError(f"no world rarity reference for {key!r}")
    ref = WORLD_REFERENCES[key]
    cdf = normal_cdf(value, ref.mean, ref.sd)
    same_or_better = normal_sf(value, ref.mean, ref.sd) if ref.higher_is_better else cdf
    top_percent = _pct(same_or_better)
    return {
        "value": value,
        "world_percentile": _pct(cdf),
        "same_or_better": top_percent,
        "top_percent": top_percent,
        "top_percent_precise": _precise_number(top_percent),
        "direction": "top" if ref.higher_is_better else "bottom",
    }


def rarity_rows(result: Result) -> list[dict[str, float | str]]:
    """Rows for current state and horizon p5/median/p95, per outcome variable."""
    rows: list[dict[str, float | str]] = []
    for key in result.keys:
        current = result.bands[key][0]["p50"]
        end = result.bands[key][-1]
        values = (
            ("current", current),
            ("p5", end["p5"]),
            ("median", end["p50"]),
            ("p95", end["p95"]),
        )
        for label, value in values:
            row = rarity_for_value(key, value)
            row["variable"] = key
            row["case"] = label
            rows.append(row)
    return rows


def rarity_table(result: Result) -> str:
    """Render bell-curve rarity percentages for the text report."""
    lines = [
        "BELL-CURVE WORLD RARITY",
        "  Percentages are model-implied shares of an illustrative world reference,",
        "  not empirical census precision. Kept separate per variable; no life score.",
        "  top % is the full-precision percentage of the reference population",
        "  at least as good as this value.",
        f"  {'variable':<12} {'case':<8} {'value':>8} {'world <=':>10} {'top %':>24}",
        f"  {'-'*12} {'-'*8} {'-'*8:>8} {'-'*10:>10} {'-'*24:>24}",
    ]
    for row in rarity_rows(result):
        lines.append(
            f"  {str(row['variable']):<12} {str(row['case']):<8} "
            f"{float(row['value']):>8.2f} "
            f"{_fmt_pct(float(row['world_percentile'])):>10} "
            f"{_fmt_pct_precise(float(row['top_percent'])):>24}"
        )
    return "\n".join(lines)
