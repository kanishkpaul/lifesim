"""lifesim — a Monte Carlo simulator for a human life trajectory.

The core (config, state, dynamics, events, simulate, report) is pure standard
library. Optional pieces (matplotlib plotting, scenario loaders) are import-
guarded elsewhere so the core runs on a bare interpreter.

This is honest structured brainstorming with explicit uncertainty, NOT a
predictor. See CLAUDE.md and README.md for the honesty caveats.
"""

__version__ = "0.1.0"

SEED_DOC = "All randomness flows through one explicitly-passed RNG. See config.Config."
