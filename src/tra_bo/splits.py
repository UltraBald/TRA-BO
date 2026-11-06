"""Leakage-free chronological windows for temporal HPO."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastWindow:
    """Training prefix and its disjoint target interval."""

    name: str
    train_start: int
    train_end: int
    target_start: int
    target_end: int


@dataclass(frozen=True)
class OrderedTemporalSplit:
    """Chronological validation windows followed by one future window."""

    validation: tuple[ForecastWindow, ...]
    future: ForecastWindow
    outer_end: int
    horizon: int


def make_ordered_split(
    length: int,
    horizon: int,
    validation_windows: int = 4,
    outer_fraction: float = 1.0,
    minimum_training_points: int | None = None,
) -> OrderedTemporalSplit:
    """Create expanding-prefix windows V1 < ... < VK < V+."""

    if length < 1 or horizon < 1 or validation_windows < 2:
        raise ValueError("length and horizon must be positive; use at least two windows")
    if not 0.0 < outer_fraction <= 1.0:
        raise ValueError("outer_fraction must lie in (0, 1]")
    outer_end = int(length * outer_fraction)
    minimum = max(2 * horizon, 96) if minimum_training_points is None else int(
        minimum_training_points
    )
    first_target = outer_end - (validation_windows + 1) * horizon
    if first_target < minimum:
        raise ValueError("the series is too short for the requested temporal split")
    validation = tuple(
        ForecastWindow(
            name=f"V{index + 1}",
            train_start=0,
            train_end=first_target + index * horizon,
            target_start=first_target + index * horizon,
            target_end=first_target + (index + 1) * horizon,
        )
        for index in range(validation_windows)
    )
    future_start = outer_end - horizon
    future = ForecastWindow("V+", 0, future_start, future_start, outer_end)
    return OrderedTemporalSplit(validation, future, outer_end, horizon)
