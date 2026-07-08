"""Sensitivity sweep over a single knob — u or one policy weight.

A sweep answers "how much does the outcome move when I turn ONE dial?" without
pretending the rest of the world holds still. It reports, per swept value, every
tracked outcome's median AND cone width — never a collapsed score — so you can
see (say) that pushing `work` up keeps lifting skill long after it has started
eroding connection. For a `u` sweep it also prints N per row, making the "wider
distributions cost more samples" contract visible in the table itself.

This is Phase-4 optional tooling; the simulation core does not import it.
"""

from __future__ import annotations

from dataclasses import replace

from .config import Config
from .policy import POLICY_KEYS, normalize
from .simulate import pct, simulate

SWEEPABLE = ("u",) + POLICY_KEYS


def parse_sweep(spec: str) -> tuple[str, list[float]]:
    """Parse 'u=0:1:0.1' or 'work=0:1:0.2' into (param, inclusive value list).

    The stop endpoint is included when it lands on a step (within a small
    tolerance), matching the intuition that `0:1:0.1` covers both 0 and 1.
    """
    if "=" not in spec:
        raise ValueError("sweep must look like 'u=0:1:0.1'")
    param, _, rng = spec.partition("=")
    param = param.strip()
    if param not in SWEEPABLE:
        raise ValueError(f"cannot sweep {param!r}; sweepable: {SWEEPABLE}")
    parts = rng.split(":")
    if len(parts) != 3:
        raise ValueError("sweep range must be start:stop:step, e.g. 0:1:0.1")
    start, stop, step = (float(x) for x in parts)
    if step <= 0:
        raise ValueError("sweep step must be > 0")
    values: list[float] = []
    v = start
    # Integer step count avoids float drift accumulating over many additions.
    n_steps = int(round((stop - start) / step))
    for i in range(n_steps + 1):
        values.append(round(start + i * step, 10))
    return param, values


def _with_policy_weight(policy: dict, key: str, value: float) -> dict:
    """Set one policy weight to `value`, keep the others, renormalize to sum 1."""
    updated = dict(policy)
    updated[key] = value
    return normalize(updated)


def run_sweep(cfg: Config, policy: dict, param: str,
              values: list[float]) -> list[dict]:
    """Run one simulation per swept value; return per-outcome medians and cones.

    Returns a list of row dicts: `{"value", "n", "outcomes": {var: {p50, cone}}}`.
    For `param == "u"` the config's uncertainty (and hence N) changes each row;
    otherwise the named policy weight moves and the config is held fixed.
    """
    rows: list[dict] = []
    for v in values:
        if param == "u":
            run_cfg = replace(cfg, uncertainty=v)
            run_policy = policy
        else:
            run_cfg = cfg
            run_policy = _with_policy_weight(policy, param, v)
        res = simulate(run_cfg, run_policy)
        outcomes = {}
        for k in res.keys:
            xs = [f[k] for f in res.finals]
            outcomes[k] = {"p50": pct(xs, 50), "cone": pct(xs, 95) - pct(xs, 5)}
        rows.append({"value": v, "n": res.n, "outcomes": outcomes,
                     "keys": res.keys})
    return rows
