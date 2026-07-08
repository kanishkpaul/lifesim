"""Optional matplotlib plotting — the percentile fan per tracked variable.

matplotlib is NOT a core dependency. The import is guarded so that importing this
module (or the package) never fails on a bare interpreter; the failure only
happens if you actually call `plot_bands` without matplotlib installed, and then
it is a clear, actionable message rather than an ImportError at load time.

The picture it draws is the honesty story: a shaded p5–p95 band widening with the
horizon around a p50 line, one subplot per outcome variable — the cone you cannot
close, made visual.
"""

from __future__ import annotations

from .simulate import Result


def has_matplotlib() -> bool:
    """True if matplotlib can be imported. Used to gate --plot in the CLI."""
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:
        return False


def plot_bands(result: Result, path: str) -> str:
    """Save a percentile-fan figure for every tracked variable to `path`.

    Raises a helpful RuntimeError (not ImportError) if matplotlib is missing, so
    the optional dependency stays genuinely optional. Returns `path` on success.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: no display needed to save a file
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "matplotlib is required for --plot but is not installed. "
            "Install it (`pip install matplotlib`) or use text output. "
            "The core simulator never needs it."
        ) from e

    keys = result.keys
    months = list(range(len(result.bands[keys[0]])))
    ncols = 2
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3 * nrows), squeeze=False)

    for idx, k in enumerate(keys):
        ax = axes[idx // ncols][idx % ncols]
        band = result.bands[k]
        p5 = [b["p5"] for b in band]
        p50 = [b["p50"] for b in band]
        p95 = [b["p95"] for b in band]
        ax.fill_between(months, p5, p95, alpha=0.25, label="p5–p95")
        ax.plot(months, p50, linewidth=1.8, label="median")
        ax.set_title(k)
        ax.set_xlabel("month")
        ax.grid(True, alpha=0.3)

    # Blank any unused subplot cells.
    for idx in range(len(keys), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    cfg = result.cfg
    fig.suptitle(
        f"lifesim — domain={cfg.domain}  u={cfg.u():.2f}  N={result.n}  "
        f"(the cone never closes)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
