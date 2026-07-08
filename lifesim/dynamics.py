"""The transition step: one month of a life.

`step` is pure — it reads `state` and returns a NEW dict, mutating nothing. The
only sanctioned mutation in the whole engine happens when `step` hands the fresh
dict to `events.apply_events` at the very end (documented exception, CLAUDE.md).

The coupling graph below is mandatory. It is what makes a trajectory a life
instead of N independent random walks: energy gates skill, work crowds out
connection, mood scales downstream effects, career capital compounds into money.
If you deleted the coupling and kept only the noise, the tool would be a bell
curve and worthless. Coefficients are kept modest so nothing saturates instantly.
"""

from __future__ import annotations

import random

from . import events
from .state import clamp_state

# --- Coupling coefficients (prototype values; tuned to not saturate). ---
_SKILL_RATE = 0.06          # work->skill learning rate, gated by energy
_REP_LAG = 0.10             # how fast reputation chases skill
_REP_EXPLORE = 0.03         # exploration nudges visibility
_NET_GROWTH = 0.05          # work+explore build network
_NET_DECAY = 0.01           # relationships/contacts fade if untended
_ENERGY_REV = 0.25          # energy mean-reversion rate
_HEALTH_REV = 0.15          # health mean-reversion rate
_MOOD_REV = 0.20            # mood mean-reversion rate
_LUCK_REV = 0.30            # luck snaps back to 0.5 fairly quickly
_CONN_GROWTH = 0.08         # love*mood builds connection
_CONN_WORK_CROWD = 0.06     # work allocation actively erodes connection
_CONN_DECAY = 0.02          # connection decays under neglect
_BURN = 0.5                 # ~constant monthly burn, in months-of-runway units
_INCOME_GAIN = 0.9          # career capital -> income scale

# Per-variable noise factors k: multiplied by cfg.noise_scale(). savings noise is
# in months-of-runway units, so it carries a larger factor than bounded vars.
_NOISE_K: dict[str, float] = {
    "skill": 0.5,
    "reputation": 0.5,
    "network": 0.6,
    "savings": 3.0,
    "health": 0.6,
    "energy": 1.0,
    "mood": 1.0,
    "connection": 0.8,
    "luck": 1.2,
}


def _noise(rng: random.Random, scale: float, var: str) -> float:
    return rng.gauss(0.0, scale * _NOISE_K[var])


def step(state: dict, policy: dict, cfg, rng: random.Random) -> dict:
    """Advance the state by one month and return a new dict.

    Args:
        state: current state vector (not mutated).
        policy: attention allocation over work/love/health/explore, sums to 1.
        cfg: Config, read for noise_scale().
        rng: the single seeded RNG threading all randomness.

    The order is: read old state -> compute coupled updates into a fresh dict ->
    add per-variable Gaussian noise -> clamp bounded vars -> apply discrete
    events (the one mutation) -> clamp again. Returns the new state.
    """
    ns = cfg.noise_scale()
    work = policy["work"]
    love = policy["love"]
    health_a = policy["health"]
    explore = policy["explore"]

    s = dict(state)  # fresh copy; we never touch the caller's dict

    # --- energy: mean-reverts to a setpoint pulled DOWN by work, UP by health. ---
    energy_set = 0.55 - 0.30 * work + 0.35 * health_a
    s["energy"] = state["energy"] + _ENERGY_REV * (energy_set - state["energy"]) \
        + _noise(rng, ns, "energy")

    # --- health: setpoint driven by health allocation, dragged by grind. ---
    health_set = 0.55 + 0.40 * health_a - 0.20 * work
    s["health"] = state["health"] + _HEALTH_REV * (health_set - state["health"]) \
        + _noise(rng, ns, "health")

    # --- skill: diminishing-returns gain, GATED by (old) energy. Can't compound
    #     competence while depleted, no matter how much you allocate to work. ---
    skill_gain = _SKILL_RATE * work * state["energy"] * (1.0 - state["skill"])
    s["skill"] = state["skill"] + skill_gain + _noise(rng, ns, "skill")

    # --- reputation: lags skill (smoothing), nudged by exploration/visibility. ---
    s["reputation"] = state["reputation"] \
        + _REP_LAG * (state["skill"] - state["reputation"]) \
        + _REP_EXPLORE * explore + _noise(rng, ns, "reputation")

    # --- network: grows with work+explore, slow natural decay. ---
    s["network"] = state["network"] \
        + _NET_GROWTH * (0.5 * work + explore) * (1.0 - state["network"]) \
        - _NET_DECAY * state["network"] + _noise(rng, ns, "network")

    # --- savings: += income - burn. Income rises with career capital; mood
    #     scales it (a flat affect underperforms its resume). Burn ~constant. ---
    career_capital = 0.5 * state["skill"] + 0.5 * state["reputation"]
    mood_mult = 0.7 + 0.6 * state["mood"]   # mood scales downstream effects
    income = _INCOME_GAIN * career_capital * mood_mult
    s["savings"] = state["savings"] + income - _BURN + _noise(rng, ns, "savings")

    # --- connection: grows with love*mood, actively CROWDED OUT by work, decays
    #     under neglect. This is the career-vs-love tension, made mechanical. ---
    conn_gain = _CONN_GROWTH * love * state["mood"] * (1.0 - state["connection"])
    conn_loss = _CONN_WORK_CROWD * work + _CONN_DECAY * (1.0 - love)
    s["connection"] = state["connection"] + conn_gain \
        - conn_loss * state["connection"] + _noise(rng, ns, "connection")

    # --- mood: mean-reverts to a setpoint lifted by health & connection,
    #     depressed by financial stress (runway under one month). ---
    stress = 0.15 if state["savings"] < 1.0 else 0.0
    mood_set = 0.40 + 0.25 * state["health"] + 0.25 * state["connection"] - stress
    s["mood"] = state["mood"] + _MOOD_REV * (mood_set - state["mood"]) \
        + _noise(rng, ns, "mood")

    # --- luck: latent, mean-reverts to 0.5, feeds ONLY event odds. ---
    s["luck"] = state["luck"] + _LUCK_REV * (0.5 - state["luck"]) \
        + _noise(rng, ns, "luck")

    clamp_state(s)                          # no bounded var escapes [0, 1]
    fired = events.apply_events(s, policy, cfg, rng)  # the one sanctioned mutation
    clamp_state(s)                          # events may push out of range; re-clamp
    # Stash the month's fired events under a reserved, non-variable key so the
    # engine can build a per-life surprise record without breaking step's
    # new-state return contract. Nothing in the coupling or reports reads it as a
    # state variable (it isn't in ALL_VARS), so it rides along harmlessly.
    s["_fired"] = fired
    return s
