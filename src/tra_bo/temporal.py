"""Temporal Rank Agreement and reliability-calibrated objectives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import kendalltau


@dataclass(frozen=True)
class WindowSummary:
    """Loss summaries for chronologically ordered validation windows."""

    early: NDArray[np.float64]
    recent: NDArray[np.float64]
    full_history: NDArray[np.float64]


@dataclass(frozen=True)
class TRAEstimate:
    """Tie-adjusted Temporal Rank Agreement estimate."""

    value: float
    kendall_tau_b: float
    valid: bool
    n_configurations: int


def _as_loss_matrix(losses: ArrayLike) -> NDArray[np.float64]:
    matrix = np.asarray(losses, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("losses must have shape [n_configurations, n_windows]")
    if matrix.shape[0] < 2:
        raise ValueError("at least two evaluated configurations are required")
    if matrix.shape[1] < 2:
        raise ValueError("at least two validation windows are required")
    if not np.isfinite(matrix).all():
        raise ValueError("losses must contain only finite values")
    return matrix


def summarize_windows(losses: ArrayLike, recent_count: int) -> WindowSummary:
    """Compute early, recent, and full-history validation losses."""

    matrix = _as_loss_matrix(losses)
    n_windows = matrix.shape[1]
    if not 1 <= recent_count < n_windows:
        raise ValueError("recent_count must satisfy 1 <= recent_count < n_windows")
    split = n_windows - recent_count
    return WindowSummary(
        early=matrix[:, :split].mean(axis=1),
        recent=matrix[:, split:].mean(axis=1),
        full_history=matrix.mean(axis=1),
    )


def compute_tra(losses: ArrayLike, recent_count: int) -> TRAEstimate:
    """Compute TRA as a rescaled Kendall tau-b rank agreement."""

    summary = summarize_windows(losses, recent_count)
    if np.ptp(summary.early) == 0.0 or np.ptp(summary.recent) == 0.0:
        return TRAEstimate(0.5, 0.0, False, len(summary.early))

    result = kendalltau(summary.early, summary.recent, variant="b")
    tau = float(result.statistic)
    if not np.isfinite(tau):
        return TRAEstimate(0.5, 0.0, False, len(summary.early))
    q = float(np.clip((1.0 + tau) / 2.0, 0.0, 1.0))
    return TRAEstimate(q, tau, True, len(summary.early))


def temporal_target(
    losses: ArrayLike,
    recent_count: int,
    q: float,
    eta: float = 1.0 / 8.0,
) -> NDArray[np.float64]:
    """Calibrate full-history losses toward recent losses using TRA."""

    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must lie in [0, 1]")
    summary = summarize_windows(losses, recent_count)
    weight = eta * (1.0 - q)
    return (1.0 - weight) * summary.full_history + weight * summary.recent
