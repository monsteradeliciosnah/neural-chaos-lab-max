from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "logistic_map_step",
    "henon_step",
    "lorenz_step",
]

def _as_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x)

def logistic_map_step(x: ArrayLike, r: float = 3.8) -> np.ndarray:
    X = _as_array(x)
    return r * X * (1.0 - X)

def henon_step(state: ArrayLike, a: float = 1.4, b: float = 0.3) -> np.ndarray:
    S = _as_array(state)
    if S.shape[-1] != 2:
        raise ValueError("henon_step expects last dimension == 2 (x, y).")
    x, y = S[..., 0], S[..., 1]
    x_next = 1.0 - a * x * x + y
    y_next = b * x
    return np.stack([x_next, y_next], axis=-1)

def lorenz_step(state: ArrayLike, dt: float = 0.01,
                sigma: float = 10.0, beta: float = 8.0/3.0, rho: float = 28.0) -> np.ndarray:
    S = _as_array(state)
    if S.shape[-1] != 3:
        raise ValueError("lorenz_step expects last dimension == 3 (x, y, z).")
    x, y, z = S[..., 0], S[..., 1], S[..., 2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.stack([x + dt * dx, y + dt * dy, z + dt * dz], axis=-1)
