# CLAUDE.md — lifesim

A Monte Carlo simulator for a human life trajectory. Given initial conditions,
it rolls forward to horizon `t` under uncertainty `u`, sampling many trajectories
and reporting the distribution of outcomes — for a single domain (career, love,
health) or all of them coupled together.

This file is the constitution. `SPEC.md` is the detailed build. When they seem to
conflict, the non-negotiables below win.

## What this is (and isn't)

It is honest structured brainstorming with explicit uncertainty. It is **not** a
predictor. The dominant term in any real life is the set of variables nobody wrote
down. Every design decision below exists to stop the tool from lying about what it
can see. If a feature would make the output look more precise than the model
earns, it is a bug, not a feature.

## Non-negotiable design principles

1. **`u` scales two things at once.** The uncertainty knob `u ∈ [0,1]` must control
   both (a) the per-step aleatoric noise magnitude and (b) the number of sampled
   trajectories `N`. `N` scales linearly from `n_min` at u=0 to `n_max` at u=1, so
   **compute is roughly proportional to `u`**. You don't get to claim high
   uncertainty for free — resolving a wider distribution costs more samples.

2. **The cone never closes.** There is an irreducible `epistemic_floor` of noise
   that stays strictly positive even at u=0. No configuration may produce a point
   prediction. The report must surface how much the outcome fan widened over the
   horizon; that number is the honesty.

3. **Life is not Gaussian.** Per-step noise is only half of it. Discrete stochastic
   **events** (breakthrough, layoff, meeting someone, breakup, health shock,
   windfall) create the fat tails. Event base rates scale with `u`. Without events
   the distribution is a boring bell curve and the tool is worthless.

4. **Variables are coupled, not independent.** Trajectories must be more than `N`
   parallel random walks. The coupling graph in `SPEC.md` is mandatory: energy
   gates skill gains, work allocation crowds out connection, mood scales downstream
   effects, success compounds. Independent walks are a failure state.

5. **Multi-objective, never collapsed.** Track the whole outcome vector. Do **not**
   compute a single weighted "life score" anywhere in the core. The user compares
   tradeoffs across dimensions themselves. A scalar utility hides exactly the
   conflicts (career vs love, grind vs health) the tool exists to reveal.

6. **Policies are first-class.** A policy is an allocation of finite attention over
   life-domains that summs to 1. The engine must support comparing multiple
   policies on **variance and downside risk**, not just median outcome. "Which
   policy do I regret least in the worst case" is a supported query.

7. **Model-predictive, not open-loop.** The engine must be able to start a fresh
   simulation from an arbitrary observed state, so the user can re-plan every few
   weeks against what actually happened instead of trusting one long-horizon roll.

8. **Reproducible.** A fixed `seed` produces identical output. All randomness flows
   through a single seeded `random.Random` (or numpy Generator) instance passed
   explicitly — no module-level `random.*` calls, no hidden global state.

## Coding standards

- **Core is pure standard library.** Python 3.10+. No third-party imports in the
  simulation core. `matplotlib`, JSON/YAML scenario loaders, and any dashboard are
  optional and must be import-guarded so the core runs on a bare interpreter.
- Functional transition step: `step(state, policy, cfg, rng) -> new_state`. No
  in-place mutation of state except inside the events module, which is documented
  as the one exception.
- Everything clamps. Bounded variables live in `[0,1]`; unbounded ones (savings)
  are documented. No variable may silently drift out of range.
- Type hints throughout. Docstrings explain the *why* (the modeling choice), not
  just the *what*.
- Small modules, single responsibility. No file over ~250 lines.
- Tests are not optional (see `SPEC.md` §Testing). The honesty properties are
  encoded as invariants, not vibes.

## Build phases (do in order, test after each)

- **Phase 1 — Core loop.** `config`, `state`, `dynamics`, `events`, `simulate`,
  `report`, `cli`. One domain plus "all". Text output only. Ship when
  `python3 -m lifesim` runs and the honesty metrics print.
- **Phase 2 — Policies & risk.** `policy` module, `--compare` mode ranking policies
  by median / spread / CVaR downside. Regret table across policies.
- **Phase 3 — Scenarios & re-planning.** JSON scenario files for specific people;
  `--from-state` to resume from an observed state (the MPC loop).
- **Phase 4 — Optional extras.** Guarded `plot` module; a `--sweep` sensitivity
  analysis over `u` or a single policy weight. Nothing here may become a core dep.

Stop after each phase and show test output before proceeding.
