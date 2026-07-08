"""Text report: outcome distribution, rarity reference, and honesty metrics.

The point of the honesty block is to make the tool's own uncertainty legible.
Two numbers carry it: the FAN RATIO (how many times wider the outcome cone is at
the horizon than near the start — the visible cost of looking far ahead) and the
SURPRISE INDEX (what fraction of simulated lives got hit by at least one rare
shock — a reminder that tails are not rare in aggregate). Plus one fixed line
saying the loudest variable is always the one nobody modeled. We never print a
single "life score": the whole outcome vector is shown so the user weighs the
tradeoffs themselves (CLAUDE.md #5).
"""

from __future__ import annotations

from .events import RARE_EVENTS
from .precision import precision_table
from .rarity import rarity_table
from .simulate import Result
from .state import HIGHER_IS_BETTER

_REMINDER = (
    "Reminder: past the near term, the outcome is dominated by variables nobody "
    "wrote down. This is structured brainstorming with error bars, not a forecast."
)


def _fmt(x: float) -> str:
    return f"{x:+.2f}" if x < 0 else f"{x:.2f}"


def stdev(xs: list[float]) -> float:
    """Population standard deviation of a sample. Spread of the outcome."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    return (sum((x - mean) ** 2 for x in xs) / n) ** 0.5


def cvar_downside(
    xs: list[float], frac: float = 0.05, higher_is_better: bool = True
) -> float:
    """Mean of the worst `frac` of outcomes — the number that matters for worst case.

    For higher-is-better variables the downside tail is low. For lower-is-better
    variables (debt, stress, loneliness, etc.) the downside tail is high. This is
    deliberately NOT the p5/p95 threshold but the mean beyond it, so a fat bad
    tail is punished.
    """
    if not xs:
        raise ValueError("cvar of empty sequence")
    s = sorted(xs)
    k = max(1, int(len(s) * frac))
    tail = s[:k] if higher_is_better else s[-k:]
    return sum(tail) / len(tail)


def surprise_index(result: Result) -> float:
    """Fraction of trajectories that hit >= 1 rare event. Always in [0, 1].

    Note this saturates toward 1 over long horizons — given enough months almost
    every life gets hit by *something*. That is honest but stops discriminating,
    which is why the report pairs it with `mean_rare_events` (below), a count that
    keeps moving after the fraction has pinned at 100%.
    """
    if not result.fired:
        return 0.0
    hit = sum(1 for s in result.fired if s & RARE_EVENTS)
    return hit / len(result.fired)


def mean_rare_events(result: Result) -> float:
    """Average number of DISTINCT rare event types per life.

    Because `fired` stores a set per trajectory, this counts distinct rare event
    *types* a life experienced (0–4), not repeat occurrences. It stays
    informative where the hit/no-hit fraction saturates: two horizons can both
    read 100% surprise while one averages 1.2 rare types and the other 3.1.
    """
    if not result.fired:
        return 0.0
    return sum(len(s & RARE_EVENTS) for s in result.fired) / len(result.fired)


def fan_ratio(result: Result) -> float:
    """Mean band width at the horizon divided by mean band width at month 1.

    Averaged across tracked variables. Uses month 1 (not the jittered month 0
    start) as the baseline so the ratio reflects widening driven by the dynamics
    and events, not the initial jitter. > 1 means the cone opened.
    """
    widths_start: list[float] = []
    widths_end: list[float] = []
    for k in result.keys:
        band = result.bands[k]
        base_m = 1 if len(band) > 1 else 0
        widths_start.append(band[base_m]["p95"] - band[base_m]["p5"])
        widths_end.append(band[-1]["p95"] - band[-1]["p5"])
    mean_start = sum(widths_start) / len(widths_start)
    mean_end = sum(widths_end) / len(widths_end)
    if mean_start <= 0:
        return float("inf")
    return mean_end / mean_start


def summarize(result: Result) -> str:
    """Build the full text report as a string (printing is the CLI's job)."""
    cfg = result.cfg
    lines: list[str] = []
    lines.append("=" * 66)
    lines.append(
        f"lifesim — domain={cfg.domain}  horizon={cfg.horizon_months}mo  "
        f"u={cfg.u():.2f}  N={result.n}  seed={cfg.seed}"
    )
    lines.append("=" * 66)
    lines.append("")
    lines.append(f"Outcome distribution at month {cfg.horizon_months} "
                 f"(each variable kept separate — no collapsed score):")
    lines.append("")
    lines.append(f"  {'variable':<12} {'p5':>8} {'p50':>8} {'p95':>8} {'cone(p95-p5)':>14}")
    lines.append(f"  {'-'*12} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*14:>14}")
    for k in result.keys:
        end = result.bands[k][-1]
        cone = end["p95"] - end["p5"]
        lines.append(
            f"  {k:<12} {_fmt(end['p5']):>8} {_fmt(end['p50']):>8} "
            f"{_fmt(end['p95']):>8} {_fmt(cone):>14}"
        )
    lines.append("")
    lines.append(rarity_table(result))
    lines.append("")
    lines.append(precision_table(result))
    lines.append("")

    # --- honesty block ---
    fr = fan_ratio(result)
    si = surprise_index(result)
    mre = mean_rare_events(result)
    lines.append("-" * 66)
    lines.append("HONESTY METRICS")
    lines.append("-" * 66)
    if fr == float("inf"):
        lines.append("  Fan ratio: undefined (zero spread near start).")
    else:
        lines.append(
            f"  Fan ratio: {fr:.1f}x  — the outcome cone grew about {fr:.1f} times "
            f"wider over {cfg.horizon_months} months."
        )
    lines.append(
        f"  Surprise index: {si:.0%}  — {si:.0%} of simulated lives hit at least "
        f"one rare event (breakthrough/layoff/health shock/windfall)."
    )
    lines.append(
        f"  Rare types per life: {mre:.2f}  — avg distinct rare events a life saw "
        f"(0-4). Keeps discriminating where the fraction above saturates."
    )
    lines.append("")
    lines.append("  " + _REMINDER)
    lines.append("-" * 66)
    return "\n".join(lines)


def comparison_table(comparison: dict) -> str:
    """Render `policy.compare` output: one block per outcome variable.

    Deliberately no aggregate score. Each variable gets its own table so the
    reader sees the tradeoff directly — the policy with the best median on skill
    may have the worst downside on connection, and that conflict is the answer.
    """
    keys = comparison["keys"]
    policies = comparison["policies"]
    rows = comparison["rows"]
    lines: list[str] = []
    lines.append("=" * 74)
    lines.append("POLICY COMPARISON — median / spread / downside-CVaR(5%) / regret")
    lines.append("  (regret = gap to the best policy's worst-case for that variable)")
    lines.append("  No column is a total. Pick the variable and risk you care about.")
    lines.append("=" * 74)
    for k in keys:
        lines.append("")
        lines.append(f"  {k}")
        lines.append(f"    {'policy':<20} {'median':>9} {'spread':>9} "
                     f"{'CVaR↓':>9} {'regret':>9}")
        lines.append(f"    {'-'*20} {'-'*9:>9} {'-'*9:>9} {'-'*9:>9} {'-'*9:>9}")
        # Order by safest bad-tail outcome, respecting metric direction.
        hib = HIGHER_IS_BETTER.get(k, True)
        ordered = sorted(
            policies, key=lambda p: rows[p][k]["cvar"], reverse=hib
        )
        for name in ordered:
            r = rows[name][k]
            marker = "  <- best floor" if r["regret"] == 0 else ""
            lines.append(
                f"    {name:<20} {_fmt(r['median']):>9} {_fmt(r['spread']):>9} "
                f"{_fmt(r['cvar']):>9} {_fmt(r['regret']):>9}{marker}"
            )
    lines.append("")
    lines.append("=" * 74)
    return "\n".join(lines)


def sweep_table(rows: list[dict], param: str) -> str:
    """Render a `sweep.run_sweep` result: one row per swept value.

    Shows each outcome's median with its cone width in parentheses, so movement
    in the central estimate and movement in the uncertainty are both legible. No
    aggregate — the reader watches the tradeoff shift as the single knob turns.
    """
    if not rows:
        return "(empty sweep)"
    keys = rows[0]["keys"]
    lines: list[str] = []
    lines.append("=" * (14 + 14 * len(keys)))
    lines.append(f"SENSITIVITY SWEEP over {param}  —  cell = median (cone width)")
    if param == "u":
        lines.append("  N grows with u: resolving a wider distribution costs samples.")
    lines.append("=" * (14 + 14 * len(keys)))
    header = f"  {param:>7} {'N':>6}  " + "".join(f"{k[:12]:>14}" for k in keys)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for r in rows:
        cells = "".join(
            f"{_fmt(r['outcomes'][k]['p50'])}({_fmt(r['outcomes'][k]['cone'])})".rjust(14)
            for k in keys
        )
        lines.append(f"  {r['value']:>7.3g} {r['n']:>6}  {cells}")
    lines.append("=" * (14 + 14 * len(keys)))
    return "\n".join(lines)
