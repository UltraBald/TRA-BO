"""Mixed Sobol search spaces for DLinear and PatchTST."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc


def _choice(unit: float, values: Sequence[Any]) -> Any:
    index = min(int(np.floor(float(unit) * len(values))), len(values) - 1)
    return values[index]


def _linear(unit: float, low: float, high: float) -> float:
    return float(low + float(unit) * (high - low))


def _log_uniform(unit: float, low: float, high: float) -> float:
    return float(np.exp(np.log(low) + float(unit) * (np.log(high) - np.log(low))))


def _decode_dlinear(point: NDArray[np.float64]) -> dict[str, Any]:
    return {
        "seq_len": int(_choice(point[0], [96, 192, 336, 512, 720])),
        "moving_avg": int(_choice(point[1], [7, 13, 25, 49])),
        "individual": bool(_choice(point[2], [False, True])),
        "learning_rate": _log_uniform(point[3], 1e-5, 1e-2),
        "weight_decay": _log_uniform(point[4], 1e-7, 1e-2),
        "batch_size": int(_choice(point[5], [16, 32, 64, 128])),
    }


def _decode_patchtst(point: NDArray[np.float64]) -> dict[str, Any]:
    patch_len = int(_choice(point[1], [8, 16, 24, 32, 48]))
    strides = [value for value in [4, 8, 12, 16, 24, 32] if value <= patch_len]
    d_model = int(_choice(point[3], [64, 128, 256, 512]))
    heads = [value for value in [2, 4, 8, 16] if d_model % value == 0]
    return {
        "seq_len": int(_choice(point[0], [96, 192, 336, 512, 720])),
        "patch_len": patch_len,
        "stride": int(_choice(point[2], strides)),
        "d_model": d_model,
        "n_heads": int(_choice(point[4], heads)),
        "e_layers": int(_choice(point[5], [1, 2, 3, 4])),
        "dropout": _linear(point[6], 0.0, 0.5),
        "learning_rate": _log_uniform(point[7], 1e-5, 3e-3),
    }


_SPACES: dict[str, tuple[int, Callable[[NDArray[np.float64]], dict[str, Any]]]] = {
    "dlinear": (6, _decode_dlinear),
    "patchtst": (8, _decode_patchtst),
}


def space_dimension(forecaster: str) -> int:
    """Return the latent Sobol dimension of a supported search space."""

    key = forecaster.lower()
    if key not in _SPACES:
        raise ValueError(f"unknown forecaster {forecaster!r}; choose from {sorted(_SPACES)}")
    return _SPACES[key][0]


def decode_configuration(
    forecaster: str, point: Sequence[float]
) -> dict[str, Any]:
    """Decode one unit-cube point into a constraint-valid configuration."""

    key = forecaster.lower()
    dimension = space_dimension(key)
    vector = np.asarray(point, dtype=np.float64)
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise ValueError(f"point must be a finite vector of length {dimension}")
    if np.any(vector < 0.0) or np.any(vector >= 1.0):
        raise ValueError("Sobol coordinates must lie in [0, 1)")
    return _SPACES[key][1](vector)


def generate_sobol_pool(
    forecaster: str, count: int = 256, seed: int = 0
) -> tuple[NDArray[np.float64], list[dict[str, Any]]]:
    """Generate a latent Sobol pool and its decoded configurations."""

    if count < 1:
        raise ValueError("count must be positive")
    dimension = space_dimension(forecaster)
    exponent = int(np.ceil(np.log2(max(count, 2))))
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
    latent = sampler.random_base2(exponent)[:count]
    decoded = [decode_configuration(forecaster, point) for point in latent]
    return latent, decoded
