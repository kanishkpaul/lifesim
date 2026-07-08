"""Scenarios and model-predictive re-planning.

Two jobs, both optional (the core never imports this module):

1. A JSON *scenario* describes a specific person's starting point and run knobs —
   domain, horizon, uncertainty, starting-state overrides, and a policy. Load /
   validate / save, round-trippable without loss.

2. `--from-state`: the model-predictive loop. Instead of trusting one 60-month
   roll, you live a few months for real, write down where you ACTUALLY landed,
   and re-simulate forward from there. Resuming from an observed state takes it as
   given — no starting jitter — because it isn't a guess, it's data.

JSON is stdlib, so this loads on a bare interpreter; only YAML (not used here)
would need guarding. Nothing here feeds back into the simulation core.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .config import VALID_DOMAINS, Config
from .policy import normalize
from .state import ALL_VARS

SCHEMA_VERSION = 1


@dataclass
class Scenario:
    """A specific person + run configuration, serializable to/from JSON.

    Attributes:
        domain: "career" | "love" | "all".
        horizon_months: how far to roll.
        uncertainty: u in [0, 1].
        n_min: trajectory count at u=0.
        n_max: trajectory count at u=1.
        event_rate_scale: positive calibration multiplier for event base rates.
        overrides: starting-state values replacing defaults (before jitter).
        policy: attention allocation; renormalized on use. None => default.
        schema_version: bumped if the on-disk format changes.
    """

    domain: str = "all"
    horizon_months: int = 60
    uncertainty: float = 0.3
    n_min: int = 200
    n_max: int = 6000
    event_rate_scale: float = 1.0
    overrides: dict = field(default_factory=dict)
    policy: dict | None = None
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> "Scenario":
        if self.domain not in VALID_DOMAINS:
            raise ValueError(f"domain must be one of {VALID_DOMAINS}, got {self.domain!r}")
        if not (0 <= self.uncertainty <= 1):
            # Config clamps, but a scenario file out of range is a human error.
            raise ValueError(f"uncertainty {self.uncertainty} outside [0, 1]")
        if self.horizon_months < 1:
            raise ValueError("horizon_months must be >= 1")
        if self.n_min < 1 or self.n_max < self.n_min:
            raise ValueError("require 1 <= n_min <= n_max")
        if self.event_rate_scale <= 0:
            raise ValueError("event_rate_scale must be strictly > 0")
        _check_state_keys(self.overrides, "overrides")
        if self.policy is not None:
            normalize(self.policy)  # raises on bad/empty/negative allocations
        return self

    def to_config(self, seed: int | None = None) -> Config:
        """Build a Config from this scenario. `seed` comes from the CLI, not JSON."""
        return Config(
            horizon_months=self.horizon_months,
            uncertainty=self.uncertainty,
            domain=self.domain,
            n_min=self.n_min,
            n_max=self.n_max,
            event_rate_scale=self.event_rate_scale,
            seed=seed,
        )

    def resolved_policy(self) -> dict | None:
        """Normalized policy, or None to let the caller use its default."""
        return normalize(self.policy) if self.policy is not None else None


def _check_state_keys(d: dict, where: str) -> None:
    unknown = set(d) - set(ALL_VARS)
    if unknown:
        raise ValueError(f"unknown state variables in {where}: {sorted(unknown)}")


def load_scenario(path: str) -> Scenario:
    """Load and validate a scenario JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("scenario file must be a JSON object")
    known = {f for f in Scenario().__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown scenario fields: {sorted(unknown)}")
    return Scenario(**data).validate()


def save_scenario(scenario: Scenario, path: str) -> None:
    """Serialize a validated scenario to JSON (round-trips via load_scenario)."""
    scenario.validate()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(scenario), fh, indent=2, sort_keys=True)
        fh.write("\n")


def load_state(path: str) -> dict:
    """Load an observed-state JSON file for `--from-state` (no jitter on resume).

    Accepts either a bare state dict `{"skill": 0.6, ...}` or a wrapper
    `{"state": {...}}` so a snapshot saved from other tooling still loads. Only
    known state variables are allowed; partial states fall back to defaults for
    the rest.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "state" in data and isinstance(data["state"], dict):
        data = data["state"]
    if not isinstance(data, dict):
        raise ValueError("state file must be a JSON object of variable->value")
    _check_state_keys(data, "state")
    return {k: float(v) for k, v in data.items()}
