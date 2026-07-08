"""Policies: how finite attention is allocated across life-domains.

A policy is a dict over {"work","love","health","explore"} that sums to 1. This
module owns parsing and renormalization in Phase 1; the multi-policy `compare`
(median / spread / CVaR / regret) is added in Phase 2. Attention is finite, so
these four numbers competing for a budget of 1.0 is the whole point — spending on
work is literally not spending on love, which is how the crowd-out in dynamics.py
becomes mechanical rather than decorative.
"""

from __future__ import annotations

POLICY_KEYS = ("work", "love", "health", "explore")

Policy = dict  # a plain dict[str, float] over POLICY_KEYS, summing to 1.

DEFAULT_POLICY: dict[str, float] = {
    "work": 0.4,
    "love": 0.2,
    "health": 0.2,
    "explore": 0.2,
}

# Named presets to compare. Each is a distinct bet about where attention goes;
# the point of `compare` is to show what each bet costs in the worst case, not to
# crown a winner. All are renormalized on read via `get_policy`.
PRESETS: dict[str, dict[str, float]] = {
    "grind": {"work": 0.7, "love": 0.05, "health": 0.15, "explore": 0.1},
    "balanced": {"work": 0.4, "love": 0.2, "health": 0.2, "explore": 0.2},
    "relationship_first": {"work": 0.2, "love": 0.5, "health": 0.2, "explore": 0.1},
    "explore_heavy": {"work": 0.25, "love": 0.15, "health": 0.15, "explore": 0.45},
}


def get_policy(name: str) -> dict[str, float]:
    """Resolve a named preset to a normalized policy. Raises on unknown names."""
    if name not in PRESETS:
        raise ValueError(
            f"unknown policy preset {name!r}; known: {sorted(PRESETS)}"
        )
    return normalize(PRESETS[name])


def normalize(alloc: dict[str, float]) -> dict[str, float]:
    """Return a policy over POLICY_KEYS renormalized to sum to 1.

    Missing keys default to 0. Negative weights are rejected (an allocation of
    negative attention is meaningless). An all-zero allocation is an error.
    """
    out = {k: float(alloc.get(k, 0.0)) for k in POLICY_KEYS}
    for k, v in out.items():
        if v < 0:
            raise ValueError(f"policy weight {k}={v} must be >= 0")
    total = sum(out.values())
    if total <= 0:
        raise ValueError("policy weights sum to 0; nothing to allocate")
    return {k: v / total for k, v in out.items()}


def parse_policy(spec: str) -> dict[str, float]:
    """Parse a CLI policy string like 'work=0.4,love=0.2,health=0.2,explore=0.2'.

    Unspecified keys are 0; the result is renormalized so partial specs still sum
    to 1. Raises on unknown keys so typos fail loud rather than silently zeroing.
    """
    alloc: dict[str, float] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"bad policy term {chunk!r}; expected key=value")
        key, _, val = chunk.partition("=")
        key = key.strip()
        if key not in POLICY_KEYS:
            raise ValueError(f"unknown policy key {key!r}; expected one of {POLICY_KEYS}")
        alloc[key] = float(val)
    return normalize(alloc)


def compare(cfg, policies: dict[str, dict[str, float]],
            overrides=None, jitter: bool = True) -> dict:
    """Run each policy under the SAME cfg (and seed) and tabulate risk per outcome.

    Returns a plain dict — deliberately NOT a single ranking scalar. For every
    tracked outcome variable and every policy we report:
        median  — central outcome,
        spread  — std deviation (how uncertain),
        cvar    — mean of the worst 5% (the downside that actually stings),
        regret  — gap between this policy's worst-case (cvar) and the BEST
                  policy's worst-case for that same variable (>= 0). A policy
                  with 0 regret on a variable owns the best floor for it.

    Because regret is computed per variable, no policy is globally "best": one may
    own the skill floor while another owns the connection floor. Surfacing that
    conflict is the entire job — the user picks the risk criterion themselves.
    """
    from .report import cvar_downside, stdev  # local: avoid import cycle
    from .simulate import simulate
    from .state import HIGHER_IS_BETTER

    import statistics

    keys = None
    per_policy: dict[str, dict[str, dict[str, float]]] = {}
    for name, pol in policies.items():
        res = simulate(cfg, pol, overrides=overrides, jitter=jitter)
        keys = res.keys
        row: dict[str, dict[str, float]] = {}
        for k in res.keys:
            xs = [f[k] for f in res.finals]
            row[k] = {
                "median": statistics.median(xs),
                "spread": stdev(xs),
                "cvar": cvar_downside(xs, higher_is_better=HIGHER_IS_BETTER.get(k, True)),
            }
        per_policy[name] = row

    # Regret: per variable, gap to the safest bad-tail outcome across policies.
    for k in keys:
        hib = HIGHER_IS_BETTER.get(k, True)
        cvars = [per_policy[name][k]["cvar"] for name in policies]
        best_cvar = max(cvars) if hib else min(cvars)
        for name in policies:
            if hib:
                per_policy[name][k]["regret"] = best_cvar - per_policy[name][k]["cvar"]
            else:
                per_policy[name][k]["regret"] = per_policy[name][k]["cvar"] - best_cvar

    return {"keys": keys, "policies": list(policies), "rows": per_policy}
