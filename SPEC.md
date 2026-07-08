# SPEC.md — lifesim build specification

Read `CLAUDE.md` first. This is the concrete build. Every non-negotiable there is
assumed here.

## Project layout

```
lifesim/
  CLAUDE.md
  SPEC.md
  README.md                # generated last: usage + the honesty caveats
  lifesim/
    __init__.py
    config.py              # Config, u -> (N, noise), epistemic floor
    state.py               # State vector, domains, initial conditions, clamps
    dynamics.py            # transition step: drift, mean-reversion, coupling
    events.py              # discrete stochastic shocks (fat tails)
    policy.py              # attention-allocation policies + comparison (Phase 2)
    simulate.py            # Monte Carlo engine: trajectories + percentile bands
    report.py              # text summary, honesty metrics, risk metrics
    scenario.py            # JSON scenario load/save, --from-state (Phase 3)
    plot.py                # optional matplotlib (Phase 4, import-guarded)
    __main__.py            # CLI (argparse); enables `python3 -m lifesim`
  tests/
    test_config.py
    test_dynamics.py
    test_events.py
    test_simulate.py
    test_policy.py         # Phase 2
    test_scenario.py       # Phase 3
  examples/
    default.json           # a generic starting person
    scenario_template.json # documented blank to copy
```

## The state vector

Continuous variables in `[0,1]` unless noted. `savings` is in months of runway
(unbounded, may go negative = debt).

| var          | meaning                                   | domains        |
|--------------|-------------------------------------------|----------------|
| `skill`      | domain competence                         | career, all    |
| `reputation` | visibility / credibility                  | career, all    |
| `network`    | reachable useful people                   | career, all    |
| `savings`    | months of runway (unbounded)              | career, all    |
| `health`     | physical baseline                         | love, all      |
| `energy`     | day-to-day capacity                       | love, all      |
| `mood`       | affective baseline; scales downstream     | all            |
| `connection` | quality of close relationship(s)          | love, all      |
| `luck`       | latent, mean-reverting; modulates events  | internal       |

`state.py` provides `initial_state(domain, rng, overrides=None)` returning a dict,
with small per-trajectory jitter (~N(0, 0.03)) on the starting point so trajectories
don't begin identical. `overrides` (from a scenario file) replace defaults before
jitter. Provide `OUTCOME_KEYS: dict[str, list[str]]` mapping each domain to the
variables the report tracks.

## config.py contracts (Phase 1, testable)

```python
@dataclass
class Config:
    horizon_months: int = 60
    uncertainty: float = 0.3        # u in [0,1], clamped
    domain: str = "all"             # "career" | "love" | "all"
    n_min: int = 200
    n_max: int = 6000
    epistemic_floor: float = 0.02   # irreducible; strictly > 0
    seed: int | None = None
    noise_gain: float = 0.10
    event_rate_scale: float = 1.0   # positive event calibration multiplier

    def n_trajectories(self) -> int   # round(n_min + u*(n_max-n_min))
    def noise_scale(self) -> float    # epistemic_floor + u*noise_gain
```

Invariants tests must assert:
- `n_trajectories()` is monotonically non-decreasing in `u`; equals `n_min` at u=0,
  `n_max` at u=1.
- `noise_scale()` is strictly positive for **all** `u`, including 0.
- `event_rate_scale` is strictly positive; scenarios may calibrate events but not
  silently turn off fat tails.
- Both are pure functions of the config (no RNG).

## dynamics.py — the transition step (Phase 1)

`step(state, policy, cfg, rng) -> new_state` advances one month. Signature is pure:
returns a new dict, mutates nothing except by calling `events.apply_events` on the
new dict at the end. Noise on each update is `rng.gauss(0, noise_scale * k)` for a
per-variable factor `k`.

**Mandatory coupling graph** (this is what makes it a life, not N walks):

- `energy` mean-reverts to a setpoint pulled **down** by work allocation and **up**
  by health allocation.
- `health` mean-reverts to a setpoint driven by health allocation, dragged by work.
- `skill` gain `= rate * work_alloc * energy * (1 - skill)` — diminishing returns,
  **gated by energy** (can't compound skill while depleted).
- `reputation` lags `skill` with a smoothing term, nudged by exploration allocation.
- `network` grows with work + exploration, slow natural decay.
- `savings += income - burn`, where `income` rises with career capital
  (`0.5*skill + 0.5*reputation`); `burn` roughly constant.
- `connection` grows with `love_alloc * mood`, **crowded out** by work allocation,
  decays under neglect. This is the career-vs-love tension, made mechanical.
- `mood` mean-reverts to a setpoint lifted by health and connection, depressed by
  financial stress (`savings < 1`). Mood scales downstream effects.
- `luck` mean-reverts to 0.5; only feeds event odds.

Keep coefficients modest so nothing saturates instantly. The prototype values
(work→skill rate ≈ 0.06, mean-reversion rates 0.15–0.30) are a sane starting point;
document any you change.

## events.py — fat tails (Phase 1)

`apply_events(state, policy, cfg, rng)` mutates `state` in place (the documented
exception). Monthly base rate `= (0.01 + 0.05*u) * event_rate_scale`; each event
has a multiplier on top, often keyed to state (skill, luck, connection) or policy
(love allocation).

Minimum event set:
- **Breakthrough/promotion** — likelier with high skill + luck: ↑reputation, ↑savings, ↑mood.
- **Layoff/rejection** — likelier with low luck: ↓savings, ↓mood.
- **Meet someone** — only if `policy.love > 0.05`: ↑connection, ↑mood.
- **Breakup/drift** — likelier when connection is neglected: ↓connection, ↓mood.
- **Health shock** — likelier with low health + bad luck: ↓health, ↓energy.
- **Windfall** — rare pure upside: ↑savings, ↑network.

Structure events as a list of `(name, prob_fn, effect_fn)` so adding one is a
one-line append. Record which events fired per trajectory so the report can compute
a **surprise index** (fraction of lives that hit at least one rare event).

## simulate.py — the engine (Phase 1)

- `run_trajectory(cfg, policy, rng) -> (final_state, path, fired_events)` where
  `path` tracks each `OUTCOME_KEYS[domain]` variable monthly.
- `simulate(cfg, policy) -> Result` runs `cfg.n_trajectories()` trajectories through
  one seeded RNG. `Result` holds: `finals` (list of end states), monthly percentile
  `bands` (p5/p50/p95 per tracked var), `n`, and per-trajectory fired-event sets.
- Percentiles via a pure-stdlib `pct(xs, p)` with linear interpolation. No numpy in core.

## report.py — output + honesty + risk (Phase 1 basic, Phase 2 risk)

Text summary prints, per tracked variable at horizon: p5, p50, p95, and cone width
(p95−p5). It also prints a bell-curve world rarity block:
- **Current**: the modeled starting-state median for each tracked variable.
- **Potential p5 / median / p95**: horizon outcomes compared against a simple
  normal world reference.
- **world <= / same-better**: precise percentages from the reference curve, with
  a caveat that the reference is illustrative rather than census truth.

It also prints a Monte Carlo precision block:
- **MC SE**: standard error of the sample mean, separating sampling precision from
  outcome uncertainty.
- **Split Δmax**: largest p5/p50/p95 difference between even and odd trajectories,
  a deterministic percentile stability check.

Then the honesty block:
- **Fan ratio**: mean band width at month `t` ÷ mean band width at month 1. Print it
  in plain words ("uncertainty grew 10× over the horizon").
- **Surprise index**: fraction of trajectories that hit ≥1 rare event.
- A fixed one-line reminder that undescribed variables dominate past the near term.

Phase 2 adds risk metrics per outcome: standard deviation and **CVaR** on the
downside (mean of the worst 5% of outcomes) — the number that actually matters for
"worst case".

## precision.py — sampling precision without fake certainty

- `monte_carlo_rows(result)` reports per-variable mean, MC standard error, and a
  split-half percentile drift.
- `convergence_check(base_result, policy, factor=2.0)` reruns the same seed and
  scenario with scaled `n_min`/`n_max`, comparing horizon p5/p50/p95. This is an
  opt-in cost because bigger uncertainty fans should cost more samples.

## rarity.py — bell-curve rarity without a life score

- `normal_cdf(x, mean, sd)` computes a pure-stdlib normal CDF.
- `WORLD_REFERENCES` holds transparent per-variable bell-curve assumptions. These
  are model references, not empirical world-population claims.
- `rarity_rows(result)` reports current plus horizon p5/median/p95 for each
  tracked variable. It must not collapse variables into a single rarity score.

## policy.py — policies & comparison (Phase 2)

- A policy is `{"work","love","health","explore"}` summing to 1; `parse_policy` and
  a renormalizer live here. Ship 3–4 named presets (e.g. `grind`, `balanced`,
  `relationship_first`, `explore_heavy`).
- `compare(cfg, policies) -> ranking` runs each policy and produces a table: median,
  spread, and downside CVaR per outcome, plus a **regret** column (for each policy,
  its worst-case gap to the best policy's worst case). Do not reduce to one score —
  present the table and let the user pick their risk criterion.

## scenario.py — specific people & re-planning (Phase 3)

- JSON scenarios: `{ "domain": ..., "horizon_months": ..., "uncertainty": ...,
  "n_min": ..., "n_max": ..., "event_rate_scale": ..., "overrides": {
  "skill": 0.6, ... }, "policy": {...} }`. Load/validate/save.
- `--from-state state.json` starts the sim from an observed state (no jitter), for
  the model-predictive loop: run 3 months for real, record where you actually are,
  re-simulate forward. Document this workflow in the README.

## CLI (`python3 -m lifesim`)

```
--domain {career,love,all}     default all
--u FLOAT                       uncertainty 0..1, default 0.3
--t INT                         horizon months, default 60
--seed INT
--n-min INT                     trajectory count at u=0
--n-max INT                     trajectory count at u=1
--event-rate-scale FLOAT        positive event-rate calibration multiplier
--policy "work=0.4,love=0.2,health=0.2,explore=0.2"
--scenario PATH                 load a JSON scenario (Phase 3)
--from-state PATH               resume from observed state (Phase 3)
--convergence                   compare horizon percentiles to a larger-N run
--convergence-factor FLOAT      sample multiplier for --convergence
--compare grind,balanced,...    rank named policies (Phase 2)
--sweep u=0:1:0.1               sensitivity sweep (Phase 4)
--plot                          save matplotlib plot (Phase 4, guarded)
```

## Testing (required, not optional)

Encode the honesty principles as invariants:
- **config**: N monotonic in u; noise strictly positive at u=0; endpoints exact.
- **dynamics**: all bounded vars stay in `[0,1]` over a long roll; a fixed seed
  reproduces byte-identical output; higher `u` yields strictly wider end-state
  spread (fan widens with uncertainty).
- **coupling**: a work-heavy policy produces higher median `skill` **and** lower
  median `connection` than a love-heavy policy at the same seed (proves crowd-out
  is real, not decorative).
- **events**: event rate rises with `u`; event-rate calibration affects base
  rates; surprise index in `[0,1]`.
- **policy**: allocations renormalize to sum 1; `compare` returns one row per policy.
- **scenario**: round-trip load/save is lossless; `--from-state` skips jitter.
- **precision**: MC precision rows cover all outcomes; convergence uses a larger
  sample budget while preserving `u`.
- **rarity**: bell-curve percentages are deterministic; rows cover current and
  horizon p5/median/p95 per outcome variable; no single life score is emitted.

Use `pytest` if available, else a plain `python3 -m tests.run` harness — but the
test *logic* stays stdlib so it runs anywhere.

## README.md (generate last)

Usage examples, the knob semantics, and — prominently — the caveats: this is a toy
for structured thinking, `u` buys resolution not truth, the cone never closes, and
the biggest variable is always the one you didn't model.
