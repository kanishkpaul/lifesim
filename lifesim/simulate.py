"""The Monte Carlo engine.

Runs `cfg.n_trajectories()` lives through one seeded RNG and collects both the
end-state distribution and the monthly percentile bands (p5/p50/p95) of each
tracked variable. The bands are what let the report show the cone widening over
the horizon — the honesty metric. Percentiles use a pure-stdlib interpolating
`pct`; no numpy in the core.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import dynamics
from .policy import DEFAULT_POLICY
from .state import OUTCOME_KEYS, initial_state


def pct(xs: list[float], p: float) -> float:
    """Percentile of `xs` at `p` in [0, 100] via linear interpolation.

    Pure stdlib. Uses the same "linear interpolation between closest ranks"
    convention as numpy's default so results are unsurprising. `xs` need not be
    sorted; a copy is sorted internally.
    """
    if not xs:
        raise ValueError("pct of empty sequence")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    rank = (p / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


@dataclass
class Result:
    """Everything the report needs, and nothing collapsed into a scalar.

    Attributes:
        cfg: the Config used (for horizon, domain, u).
        finals: end-state dict per trajectory.
        bands: bands[var] is a list over months; each entry is a dict with keys
            p5/p50/p95 for that tracked variable at that month.
        n: number of trajectories actually run.
        fired: per-trajectory set of event names that fired at least once.
        keys: the tracked outcome variables for this domain.
    """

    cfg: object
    finals: list[dict] = field(default_factory=list)
    bands: dict[str, list[dict]] = field(default_factory=dict)
    n: int = 0
    fired: list[set] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)


def run_trajectory(cfg, policy: dict, rng: random.Random, overrides=None,
                   jitter: bool = True):
    """Roll one life to the horizon.

    Returns `(final_state, path, fired_events)` where `path[var]` is the monthly
    series (length horizon_months + 1, including the starting point) for each
    tracked variable, and `fired_events` is the set of event names that fired at
    least once along the way.
    """
    keys = OUTCOME_KEYS[cfg.domain]
    state = initial_state(cfg.domain, rng, overrides=overrides, jitter=jitter)
    path: dict[str, list[float]] = {k: [state[k]] for k in keys}
    fired: set[str] = set()

    for _ in range(cfg.horizon_months):
        state = dynamics.step(state, policy, cfg, rng)
        fired.update(state.get("_fired", ()))  # accumulate this life's shocks
        for k in keys:
            path[k].append(state[k])
    return state, path, fired


def simulate(cfg, policy: dict | None = None, overrides=None,
             jitter: bool = True) -> Result:
    """Run the full Monte Carlo ensemble and assemble a `Result`.

    All `cfg.n_trajectories()` lives share ONE seeded RNG (from `cfg.seed`), so a
    fixed seed reproduces byte-identical output. Monthly percentile bands are
    computed across the ensemble per tracked variable, which is how the report
    later shows the outcome fan widening over the horizon.

    Args:
        cfg: Config (domain, horizon, u, seed).
        policy: attention allocation; defaults to DEFAULT_POLICY.
        overrides: starting-state overrides (e.g. from a scenario).
        jitter: whether to jitter each start (False for --from-state resumes).
    """
    if policy is None:
        policy = DEFAULT_POLICY
    rng = random.Random(cfg.seed)
    keys = OUTCOME_KEYS[cfg.domain]
    n = cfg.n_trajectories()
    months = cfg.horizon_months + 1  # includes the starting point

    finals: list[dict] = []
    fired: list[set] = []
    # series[var][month] accumulates every trajectory's value at that month.
    series: dict[str, list[list[float]]] = {k: [[] for _ in range(months)] for k in keys}

    for _ in range(n):
        final_state, pathi, fired_i = run_trajectory(
            cfg, policy, rng, overrides=overrides, jitter=jitter
        )
        finals.append(final_state)
        fired.append(fired_i)
        for k in keys:
            col = series[k]
            for m, v in enumerate(pathi[k]):
                col[m].append(v)

    bands: dict[str, list[dict]] = {}
    for k in keys:
        bands[k] = [
            {"p5": pct(col, 5), "p50": pct(col, 50), "p95": pct(col, 95)}
            for col in series[k]
        ]

    return Result(cfg=cfg, finals=finals, bands=bands, n=n, fired=fired, keys=keys)
