"""CLI entrypoint: `python3 -m lifesim`.

Phase 1 wires up the core loop end to end (domain, u, horizon, seed, policy).
Flags for later phases (--compare, --scenario, --from-state, --sweep, --plot) are
declared here but gently report "not until Phase N" so the surface is stable and
self-documenting while the phases land in order.
"""

from __future__ import annotations

import argparse
import sys

from .config import Config
from .policy import DEFAULT_POLICY, compare, get_policy, parse_policy
from .report import comparison_table, summarize
from .simulate import simulate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m lifesim",
        description="Monte Carlo simulator for a human life trajectory. A toy for "
                    "structured thinking with explicit uncertainty, not a predictor.",
    )
    p.add_argument("--domain", choices=("career", "love", "all"), default="all",
                   help="which slice of life to track (default: all)")
    p.add_argument("--u", type=float, default=0.3,
                   help="uncertainty 0..1: scales BOTH noise and trajectory count "
                        "(default: 0.3)")
    p.add_argument("--t", type=int, default=60,
                   help="horizon in months (default: 60)")
    p.add_argument("--seed", type=int, default=None,
                   help="fixed seed => reproducible output")
    p.add_argument("--policy", type=str, default=None,
                   help='attention allocation, e.g. '
                        '"work=0.4,love=0.2,health=0.2,explore=0.2" (renormalized)')
    # --- declared now, activated in later phases ---
    p.add_argument("--compare", type=str, default=None,
                   help="[Phase 2] rank named policies by median/spread/CVaR/regret")
    p.add_argument("--scenario", type=str, default=None,
                   help="[Phase 3] load a JSON scenario")
    p.add_argument("--from-state", type=str, default=None, dest="from_state",
                   help="[Phase 3] resume from an observed state (MPC loop)")
    p.add_argument("--sweep", type=str, default=None,
                   help="[Phase 4] sensitivity sweep, e.g. u=0:1:0.1")
    p.add_argument("--plot", nargs="?", const="lifesim_fan.png", default=None,
                   dest="plot_path",
                   help="[Phase 4] save a matplotlib percentile-fan plot to PATH "
                        "(default lifesim_fan.png; optional dependency)")
    return p


def _not_yet(name: str, phase: int) -> int:
    print(f"'{name}' arrives in Phase {phase}; not wired up yet.", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # --- resolve config, policy, and starting state (scenario / from-state) ---
    overrides: dict | None = None
    jitter = True
    policy = parse_policy(args.policy) if args.policy else DEFAULT_POLICY

    if args.scenario is not None:
        from .scenario import load_scenario
        try:
            scen = load_scenario(args.scenario)
        except (OSError, ValueError) as e:
            print(f"scenario error: {e}", file=sys.stderr)
            return 2
        cfg = scen.to_config(seed=args.seed)
        resolved = scen.resolved_policy()
        if resolved is not None and args.policy is None:
            policy = resolved            # scenario policy unless CLI overrode it
        overrides = scen.overrides or None
    else:
        cfg = Config(
            horizon_months=args.t,
            uncertainty=args.u,
            domain=args.domain,
            seed=args.seed,
        )

    if args.from_state is not None:
        from .scenario import load_state
        try:
            overrides = load_state(args.from_state)
        except (OSError, ValueError) as e:
            print(f"from-state error: {e}", file=sys.stderr)
            return 2
        jitter = False                   # observed state is data, not a guess

    if args.compare is not None:
        names = [n.strip() for n in args.compare.split(",") if n.strip()]
        try:
            policies = {n: get_policy(n) for n in names}
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2
        comp = compare(cfg, policies, overrides=overrides, jitter=jitter)
        print(comparison_table(comp))
        return 0

    if args.sweep is not None:
        from .sweep import parse_sweep, run_sweep
        from .report import sweep_table
        try:
            param, values = parse_sweep(args.sweep)
        except ValueError as e:
            print(f"sweep error: {e}", file=sys.stderr)
            return 2
        rows = run_sweep(cfg, policy, param, values)
        print(sweep_table(rows, param))
        return 0

    result = simulate(cfg, policy, overrides=overrides, jitter=jitter)
    print(summarize(result))

    if args.plot_path:
        from .plot import plot_bands
        try:
            out = plot_bands(result, args.plot_path)
        except RuntimeError as e:
            print(f"\nplot skipped: {e}", file=sys.stderr)
            return 0
        print(f"\nSaved percentile-fan plot to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
