"""CLI entrypoint: `python3 -m lifesim`.

The CLI keeps the distinction between modeled uncertainty and sampling precision:
`u` widens the life fan, while `--n-min`/`--n-max` and `--convergence` help check
whether the Monte Carlo estimate of that fan is stable.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

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
    p.add_argument("--n-min", type=int, default=None,
                   help="trajectory count at u=0 (default: 200)")
    p.add_argument("--n-max", type=int, default=None,
                   help="trajectory count at u=1 (default: 6000)")
    p.add_argument("--event-rate-scale", type=float, default=None,
                   help="positive multiplier for event base rates; calibrate "
                        "scenario shock rates without disabling fat tails")
    p.add_argument("--policy", type=str, default=None,
                   help='attention allocation, e.g. '
                        '"work=0.4,love=0.2,health=0.2,explore=0.2" (renormalized)')
    p.add_argument("--compare", type=str, default=None,
                   help="rank named policies by median/spread/CVaR/regret")
    p.add_argument("--scenario", type=str, default=None,
                   help="load a JSON scenario")
    p.add_argument("--from-state", type=str, default=None, dest="from_state",
                   help="resume from an observed state (MPC loop)")
    p.add_argument("--convergence", action="store_true",
                   help="after a single-policy run, compare horizon percentiles "
                        "against a larger-N run")
    p.add_argument("--convergence-factor", type=float, default=2.0,
                   help="sample multiplier for --convergence (default: 2.0)")
    p.add_argument("--sweep", type=str, default=None,
                   help="sensitivity sweep, e.g. u=0:1:0.1")
    p.add_argument("--plot", nargs="?", const="lifesim_fan.png", default=None,
                   dest="plot_path",
                   help="save a matplotlib percentile-fan plot to PATH "
                        "(default lifesim_fan.png; optional dependency)")
    return p


def _apply_precision_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    changes = {}
    if args.n_min is not None:
        changes["n_min"] = args.n_min
    if args.n_max is not None:
        changes["n_max"] = args.n_max
    if args.event_rate_scale is not None:
        changes["event_rate_scale"] = args.event_rate_scale
    return replace(cfg, **changes) if changes else cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.convergence and (args.compare is not None or args.sweep is not None):
        print("--convergence is for a single policy run; omit --compare/--sweep",
              file=sys.stderr)
        return 2

    # --- resolve config, policy, and starting state (scenario / from-state) ---
    overrides: dict | None = None
    jitter = True
    try:
        policy = parse_policy(args.policy) if args.policy else DEFAULT_POLICY
    except ValueError as e:
        print(f"policy error: {e}", file=sys.stderr)
        return 2

    if args.scenario is not None:
        from .scenario import load_scenario
        try:
            scen = load_scenario(args.scenario)
        except (OSError, ValueError) as e:
            print(f"scenario error: {e}", file=sys.stderr)
            return 2
        try:
            cfg = _apply_precision_overrides(scen.to_config(seed=args.seed), args)
        except ValueError as e:
            print(f"config error: {e}", file=sys.stderr)
            return 2
        resolved = scen.resolved_policy()
        if resolved is not None and args.policy is None:
            policy = resolved            # scenario policy unless CLI overrode it
        overrides = scen.overrides or None
    else:
        cfg_kwargs = {
            "horizon_months": args.t,
            "uncertainty": args.u,
            "domain": args.domain,
            "seed": args.seed,
        }
        if args.n_min is not None:
            cfg_kwargs["n_min"] = args.n_min
        if args.n_max is not None:
            cfg_kwargs["n_max"] = args.n_max
        if args.event_rate_scale is not None:
            cfg_kwargs["event_rate_scale"] = args.event_rate_scale
        try:
            cfg = Config(**cfg_kwargs)
        except ValueError as e:
            print(f"config error: {e}", file=sys.stderr)
            return 2

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

    if args.convergence:
        from .precision import convergence_check, convergence_table
        try:
            check = convergence_check(
                result,
                policy,
                overrides=overrides,
                jitter=jitter,
                factor=args.convergence_factor,
            )
        except ValueError as e:
            print(f"\nconvergence error: {e}", file=sys.stderr)
            return 2
        print()
        print(convergence_table(check))

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
