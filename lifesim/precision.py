"""Monte Carlo precision diagnostics.

Outcome spread is the model's claim about life uncertainty. Monte Carlo error is
the smaller, more mechanical question: did we sample enough simulated lives to
estimate that spread stably? Keeping the two separate lets the report get more
statistically precise without pretending the future became knowable.
"""

from __future__ import annotations

from dataclasses import replace

from .simulate import Result, pct, simulate

QUANTILES = (5, 50, 95)


def _fmt(x: float) -> str:
    return f"{x:+.3f}" if x < 0 else f"{x:.3f}"


def _stdev(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    return (sum((x - mean) ** 2 for x in xs) / n) ** 0.5


def _split(xs: list[float]) -> tuple[list[float], list[float]]:
    return xs[::2], xs[1::2]


def monte_carlo_rows(result: Result) -> dict[str, dict[str, float]]:
    """Estimate sampling precision without drawing extra randomness.

    `mean_se` is the standard error of the sample mean. `split_max_delta` is the
    largest absolute p5/p50/p95 difference between even and odd trajectories, a
    deterministic stability check for the percentile band itself.
    """
    rows: dict[str, dict[str, float]] = {}
    for key in result.keys:
        xs = [float(final[key]) for final in result.finals]
        n = len(xs)
        mean = sum(xs) / n if n else 0.0
        mean_se = _stdev(xs) / (n ** 0.5) if n else 0.0
        even, odd = _split(xs)
        if even and odd:
            deltas = [abs(pct(even, q) - pct(odd, q)) for q in QUANTILES]
            split_max = max(deltas)
        else:
            split_max = 0.0
        rows[key] = {
            "mean": mean,
            "mean_se": mean_se,
            "split_max_delta": split_max,
        }
    return rows


def precision_table(result: Result) -> str:
    """Render compact sampling diagnostics for a normal report."""
    rows = monte_carlo_rows(result)
    lines = [
        "MONTE CARLO PRECISION",
        "  Sampling error is about simulation resolution, not life predictability.",
        f"  {'variable':<12} {'mean':>9} {'MC SE':>9} {'split Δmax':>12}",
        f"  {'-'*12} {'-'*9:>9} {'-'*9:>9} {'-'*12:>12}",
    ]
    for key in result.keys:
        row = rows[key]
        lines.append(
            f"  {key:<12} {_fmt(row['mean']):>9} {_fmt(row['mean_se']):>9} "
            f"{_fmt(row['split_max_delta']):>12}"
        )
    return "\n".join(lines)


def scaled_sample_config(cfg, factor: float):
    """Return a config with n_min/n_max scaled, preserving u->N proportionality."""
    if factor <= 1.0:
        raise ValueError("convergence factor must be > 1")
    n_min = max(1, round(cfg.n_min * factor))
    n_max = max(n_min, round(cfg.n_max * factor))
    return replace(cfg, n_min=n_min, n_max=n_max)


def convergence_check(
    base: Result,
    policy: dict,
    overrides=None,
    jitter: bool = True,
    factor: float = 2.0,
) -> dict:
    """Compare the current ensemble to a nested larger-N run.

    The larger run keeps the same seed, horizon, u, policy, and starting state;
    only sample counts change. This answers "are my percentiles stable if I pay
    for more trajectories?" without changing the modeled uncertainty.
    """
    bigger_cfg = scaled_sample_config(base.cfg, factor)
    bigger = simulate(bigger_cfg, policy, overrides=overrides, jitter=jitter)
    rows: dict[str, dict[str, float]] = {}
    for key in base.keys:
        small_end = base.bands[key][-1]
        big_end = bigger.bands[key][-1]
        deltas = {f"p{q}": big_end[f"p{q}"] - small_end[f"p{q}"] for q in QUANTILES}
        small_cone = small_end["p95"] - small_end["p5"]
        big_cone = big_end["p95"] - big_end["p5"]
        rows[key] = {
            **deltas,
            "cone": big_cone - small_cone,
            "max_abs": max(abs(v) for v in deltas.values()),
        }
    return {
        "factor": factor,
        "base_n": base.n,
        "bigger_n": bigger.n,
        "keys": base.keys,
        "rows": rows,
    }


def convergence_table(check: dict) -> str:
    """Render the larger-N percentile stability check."""
    lines = [
        "=" * 74,
        f"CONVERGENCE CHECK — N {check['base_n']} -> {check['bigger_n']} "
        f"({check['factor']:.2g}x sample budget)",
        "  Cells are larger-N minus current-N at the horizon; smaller is stabler.",
        "=" * 74,
        f"  {'variable':<12} {'Δp5':>9} {'Δp50':>9} {'Δp95':>9} "
        f"{'Δcone':>9} {'max |Δq|':>10}",
        f"  {'-'*12} {'-'*9:>9} {'-'*9:>9} {'-'*9:>9} "
        f"{'-'*9:>9} {'-'*10:>10}",
    ]
    for key in check["keys"]:
        row = check["rows"][key]
        lines.append(
            f"  {key:<12} {_fmt(row['p5']):>9} {_fmt(row['p50']):>9} "
            f"{_fmt(row['p95']):>9} {_fmt(row['cone']):>9} "
            f"{_fmt(row['max_abs']):>10}"
        )
    lines.append("=" * 74)
    return "\n".join(lines)
