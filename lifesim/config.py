"""Configuration and the uncertainty knob.

The single most important modeling decision lives here: `u` (uncertainty) is not
a free lunch. It scales BOTH the per-step noise magnitude AND the number of
sampled trajectories `N`. Resolving a wider distribution costs more compute, so
you cannot claim high uncertainty without paying for it (CLAUDE.md non-negotiable
#1). And there is an irreducible `epistemic_floor`: noise is strictly positive
even at u=0, so no configuration can ever emit a point prediction (#2).
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_DOMAINS = ("career", "love", "all")


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class Config:
    """Immutable-ish run configuration. `n_trajectories`/`noise_scale` are pure.

    Attributes:
        horizon_months: how far forward to roll each life.
        uncertainty: `u in [0, 1]`, clamped on read via `u()`. Drives N and noise.
        domain: which slice of life to track ("career" | "love" | "all").
        n_min: trajectory count at u=0.
        n_max: trajectory count at u=1. Compute is ~linear in u between these.
        epistemic_floor: irreducible noise floor, strictly > 0. The honesty guard.
        seed: fixed seed => byte-identical output. None => nondeterministic.
        noise_gain: how much extra noise the full uncertainty knob buys on top of
            the floor. Documented so the floor stays meaningful relative to it.
        event_rate_scale: person/scenario calibration for event base rates. Kept
            strictly positive so fat-tail events can never be disabled silently.
    """

    horizon_months: int = 60
    uncertainty: float = 0.3
    domain: str = "all"
    n_min: int = 200
    n_max: int = 6000
    epistemic_floor: float = 0.02
    seed: int | None = None
    noise_gain: float = 0.10
    event_rate_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.domain not in VALID_DOMAINS:
            raise ValueError(
                f"domain must be one of {VALID_DOMAINS}, got {self.domain!r}"
            )
        if self.n_min < 1 or self.n_max < self.n_min:
            raise ValueError("require 1 <= n_min <= n_max")
        if self.epistemic_floor <= 0:
            raise ValueError("epistemic_floor must be strictly > 0 (the cone never closes)")
        if self.horizon_months < 1:
            raise ValueError("horizon_months must be >= 1")
        if self.event_rate_scale <= 0:
            raise ValueError("event_rate_scale must be strictly > 0")

    def u(self) -> float:
        """The uncertainty knob, clamped to [0, 1]."""
        return _clamp(self.uncertainty, 0.0, 1.0)

    def n_trajectories(self) -> int:
        """Number of Monte Carlo trajectories. Linear in u: n_min at 0, n_max at 1.

        Pure function of the config; touches no RNG. This is the "compute is
        proportional to u" contract — wider distributions cost more samples.
        """
        return round(self.n_min + self.u() * (self.n_max - self.n_min))

    def noise_scale(self) -> float:
        """Per-step aleatoric noise scale. Strictly positive for ALL u.

        `epistemic_floor + u * noise_gain`. At u=0 this equals the floor (> 0),
        which is why a point prediction is impossible by construction.
        """
        return self.epistemic_floor + self.u() * self.noise_gain
