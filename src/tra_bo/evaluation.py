"""Future-regret measures for temporal HPO."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def incumbent_future_loss_trajectory(
    incumbent_indices: ArrayLike,
    candidate_future_losses: ArrayLike,
) -> np.ndarray:
    """Map an incumbent-index trajectory to held-out future losses."""

    indices = np.asarray(incumbent_indices)
    losses = np.asarray(candidate_future_losses, dtype=np.float64)
    if indices.ndim != 1 or not len(indices):
        raise ValueError("incumbent_indices must be a non-empty vector")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("incumbent_indices must contain integers")
    if losses.ndim != 1 or not len(losses) or not np.isfinite(losses).all():
        raise ValueError("candidate_future_losses must be a finite vector")
    if np.any(indices < 0) or np.any(indices >= len(losses)):
        raise ValueError("incumbent_indices contain an out-of-range index")
    return losses[indices.astype(int)]


def normalized_future_regret_auc(
    incumbent_future_losses: ArrayLike,
    selectable_future_losses: ArrayLike,
) -> float:
    """Average normalized future regret along an incumbent trajectory."""

    trajectory = np.asarray(incumbent_future_losses, dtype=np.float64)
    pool = np.asarray(selectable_future_losses, dtype=np.float64)
    if trajectory.ndim != 1 or pool.ndim != 1 or not len(trajectory) or not len(pool):
        raise ValueError("future losses must be non-empty vectors")
    if not np.isfinite(trajectory).all() or not np.isfinite(pool).all():
        raise ValueError("future losses must be finite")
    lower = float(pool.min())
    scale = float(pool.max() - lower)
    if scale <= 0.0:
        raise ValueError("selectable future losses must have non-zero range")
    return float(np.mean((trajectory - lower) / scale))


def normalized_final_regret(
    incumbent_future_loss: float,
    selectable_future_losses: ArrayLike,
) -> float:
    """Normalized regret of the final returned configuration."""

    pool = np.asarray(selectable_future_losses, dtype=np.float64)
    if pool.ndim != 1 or not len(pool) or not np.isfinite(pool).all():
        raise ValueError("selectable future losses must be a finite vector")
    lower = float(pool.min())
    scale = float(pool.max() - lower)
    if scale <= 0.0:
        raise ValueError("selectable future losses must have non-zero range")
    return float((float(incumbent_future_loss) - lower) / scale)
