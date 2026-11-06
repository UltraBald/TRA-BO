"""Acquisition functions for minimization Bayesian optimization."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from .surrogate import MaternGaussianProcess


ACQUISITIONS = ("PI", "EI", "UCB", "TS")


def acquisition_scores(
    name: str,
    surrogate: MaternGaussianProcess,
    candidates: ArrayLike,
    observed_targets: ArrayLike,
    rng: np.random.Generator,
    xi: float = 0.01,
    beta: float = 2.0,
) -> NDArray[np.float64]:
    """Return maximized utilities for one acquisition function."""

    action = name.upper()
    if action not in ACQUISITIONS:
        raise ValueError(f"unknown acquisition function: {name}")

    x = np.asarray(candidates, dtype=np.float64)
    y = np.asarray(observed_targets, dtype=np.float64)
    if action == "TS":
        return -surrogate.sample(x, rng)

    mean, std = surrogate.predict(x, return_std=True)
    std = np.maximum(np.asarray(std, dtype=np.float64), 1e-12)
    mean = np.asarray(mean, dtype=np.float64)
    incumbent = float(np.min(y))
    improvement = incumbent - mean - xi
    z = improvement / std

    if action == "PI":
        return norm.cdf(z)
    if action == "EI":
        return improvement * norm.cdf(z) + std * norm.pdf(z)
    return -mean + beta * std
