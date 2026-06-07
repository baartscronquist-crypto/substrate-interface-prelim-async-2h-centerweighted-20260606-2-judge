"""Empty scaffold for the coupled interface-response task.

Keep the public API stable:

    Config
    make_grid(cfg)
    run_trial(alpha, cfg, grid, rng, record_fields=False)

The implementation below is intentionally only an interface-compatible
baseline.  It is not a scientific solution and should score 0 on the hidden
trend evaluator until the actual model is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    alpha_values: tuple[float, ...] = (0.0, -0.001, 0.002)
    num_trials: int = 10
    dt: float = 1.0
    t_total: float = 120.0
    half_width: float = 50.0
    depth: float = 20.0
    nx: int = 64
    nz: int = 32
    history_window: int = 45
    feature_radius: float = 10.0
    interface_layer_depth: float = 2.0
    seed: int = 20260601


@dataclass(frozen=True)
class Grid:
    x: np.ndarray
    z: np.ndarray
    X: np.ndarray
    Z: np.ndarray
    dx: float
    dz: float


def make_grid(cfg: Config) -> Grid:
    x = np.linspace(-cfg.half_width, cfg.half_width, cfg.nx)
    z = np.linspace(0.0, cfg.depth, cfg.nz)
    X, Z = np.meshgrid(x, z)
    return Grid(x=x, z=z, X=X, Z=Z, dx=float(x[1] - x[0]), dz=float(z[1] - z[0]))


def run_trial(
    alpha: float,
    cfg: Config,
    grid: Grid,
    rng: np.random.Generator,
    record_fields: bool = False,
) -> dict[str, np.ndarray]:
    """Return interface-compatible placeholder arrays.

    Replace this function with a real model.  The hidden judge uses the
    second half of the returned time series as the post-transient window.
    """

    del alpha, rng
    time = np.arange(0.0, cfg.t_total + cfg.dt, cfg.dt)
    nt = int(time.size)

    rate = np.zeros(nt, dtype=float)
    feedback_scalar = np.full(nt, 0.2, dtype=float)
    scalar_surface_mean = np.full(nt, 0.2, dtype=float)

    if record_fields:
        scalar_samples = np.full((cfg.nz, cfg.nx, nt), 0.2, dtype=float)
    else:
        scalar_samples = None

    return {
        "time": time,
        "rate": rate,
        "feedback_scalar": feedback_scalar,
        "scalar_surface_mean": scalar_surface_mean,
        "scalar_samples": scalar_samples,
    }
