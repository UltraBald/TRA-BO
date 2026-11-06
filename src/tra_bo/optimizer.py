"""Core dual-timescale TRA-BO optimization loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .acquisition import acquisition_scores
from .llm import AcquisitionSelector, BOState, LLMAcquisitionSelector
from .surrogate import MaternGaussianProcess
from .temporal import compute_tra, temporal_target


WindowObjective = Callable[[NDArray[np.float64]], Sequence[float]]


@dataclass(frozen=True)
class TRABOConfig:
    """TRA-BO optimization settings."""

    budget: int = 50
    n_initial: int = 16
    recent_count: int = 2
    block_size: int = 5
    eta: float = 1.0 / 8.0
    xi: float = 0.01
    beta: float = 2.0
    gp_noise: float = 1e-6
    gp_restarts: int = 2
    random_state: int = 0

    def __post_init__(self) -> None:
        if self.budget < 2:
            raise ValueError("budget must be at least 2")
        if not 2 <= self.n_initial <= self.budget:
            raise ValueError("require 2 <= n_initial <= budget")
        if self.recent_count < 1:
            raise ValueError("recent_count must be positive")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("eta must lie in [0, 1]")
        if self.xi < 0.0:
            raise ValueError("xi must be non-negative")
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative")
        if self.gp_noise <= 0.0:
            raise ValueError("gp_noise must be positive")
        if self.gp_restarts < 0:
            raise ValueError("gp_restarts must be non-negative")


@dataclass(frozen=True)
class TrialRecord:
    """One evaluated configuration and the state used to select it."""

    evaluation: int
    candidate_index: int
    candidate: tuple[float, ...]
    window_losses: tuple[float, ...]
    target: float
    decision_tra: float
    updated_tra: float
    acquisition: str | None
    acquisition_score: float | None
    improved: bool | None


@dataclass(frozen=True)
class OptimizationResult:
    """Final incumbent, optimization trace, and TRA trajectory."""

    best_index: int
    best_candidate: tuple[float, ...]
    best_target: float
    final_tra: float
    observed_indices: tuple[int, ...]
    incumbent_indices: tuple[int, ...]
    tra_history: tuple[tuple[int, float], ...]
    final_targets: tuple[float, ...]
    trials: tuple[TrialRecord, ...]
    llm_calls: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trials"] = [asdict(record) for record in self.trials]
        return payload

    def save_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _normalize_candidates(candidates: NDArray[np.float64]) -> NDArray[np.float64]:
    lower = candidates.min(axis=0)
    scale = candidates.max(axis=0) - lower
    scale[scale == 0.0] = 1.0
    return (candidates - lower) / scale


def _latest_min_distance(features: NDArray[np.float64], indices: list[int]) -> float:
    if len(indices) < 2:
        return 0.0
    latest = features[indices[-1]]
    previous = features[np.asarray(indices[:-1], dtype=int)]
    return float(np.linalg.norm(previous - latest, axis=1).min())


class TRABO:
    """Bayesian optimization with blockwise Temporal Rank Agreement updates."""

    def __init__(
        self,
        config: TRABOConfig | None = None,
        selector: AcquisitionSelector | None = None,
    ) -> None:
        self.config = config or TRABOConfig()
        self.selector = selector if selector is not None else LLMAcquisitionSelector()

    def optimize(
        self,
        candidates: ArrayLike,
        objective: WindowObjective,
        initial_indices: Sequence[int] | None = None,
    ) -> OptimizationResult:
        """Optimize a finite pool using only ordered validation-window losses."""

        x = np.asarray(candidates, dtype=np.float64)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("candidates must be a finite matrix with shape [n, d]")
        cfg = self.config
        if cfg.budget > len(x):
            raise ValueError("budget cannot exceed the number of candidates")

        if initial_indices is None:
            initial = list(range(cfg.n_initial))
        else:
            initial = [int(index) for index in initial_indices]
            if len(initial) != cfg.n_initial or len(set(initial)) != len(initial):
                raise ValueError("initial_indices must contain n_initial unique indices")
            if min(initial) < 0 or max(initial) >= len(x):
                raise ValueError("initial_indices contain an out-of-range index")

        features = _normalize_candidates(x)
        observed = list(initial)
        remaining = set(range(len(x))) - set(observed)
        losses = [self._evaluate(objective, x[index]) for index in observed]
        n_windows = len(losses[0])
        if any(len(row) != n_windows for row in losses[1:]):
            raise ValueError("objective returned an inconsistent number of windows")
        if not 1 <= cfg.recent_count < n_windows:
            raise ValueError("recent_count must be smaller than the number of windows")

        loss_matrix = np.asarray(losses, dtype=np.float64)
        tra = compute_tra(loss_matrix, cfg.recent_count)
        tra_history: list[tuple[int, float]] = [(len(observed), tra.value)]
        initial_targets = temporal_target(
            loss_matrix, cfg.recent_count, tra.value, cfg.eta
        )
        trials: list[TrialRecord] = [
            TrialRecord(
                evaluation=position + 1,
                candidate_index=index,
                candidate=tuple(float(value) for value in x[index]),
                window_losses=tuple(float(value) for value in loss_matrix[position]),
                target=float(initial_targets[position]),
                decision_tra=tra.value,
                updated_tra=tra.value,
                acquisition=None,
                acquisition_score=None,
                improved=None,
            )
            for position, index in enumerate(observed)
        ]

        rng = np.random.default_rng(cfg.random_state)
        acquisition_history: list[str] = []
        improvement_history: list[bool] = []
        incumbent_history: list[float] = [float(initial_targets.min())]
        incumbent_indices: list[int] = [observed[int(np.argmin(initial_targets))]]

        while len(observed) < cfg.budget:
            targets = temporal_target(
                np.asarray(losses), cfg.recent_count, tra.value, cfg.eta
            )
            surrogate = MaternGaussianProcess(
                dimension=x.shape[1],
                random_state=cfg.random_state + len(observed),
                noise=cfg.gp_noise,
                n_restarts=cfg.gp_restarts,
            ).fit(features[np.asarray(observed)], targets)
            diagnostics = surrogate.diagnostics
            available = np.asarray(sorted(remaining), dtype=int)
            _, candidate_std = surrogate.predict(
                features[available], return_std=True
            )
            candidate_std = np.asarray(candidate_std, dtype=np.float64)
            state = BOState(
                tra=tra.value,
                tra_valid=tra.valid,
                evaluations=len(observed),
                initial_evaluations=cfg.n_initial,
                remaining_budget=cfg.budget - len(observed),
                total_budget=cfg.budget,
                dimension=x.shape[1],
                incumbent=float(targets.min()),
                incumbent_history=tuple(incumbent_history),
                target_mean=float(targets.mean()),
                target_std=float(targets.std()),
                target_range=float(np.ptp(targets)),
                latest_min_distance=_latest_min_distance(features, observed),
                gp_uncertainty_mean=float(candidate_std.mean()),
                gp_uncertainty_max=float(candidate_std.max()),
                gp_lengthscale_min=diagnostics.lengthscale_min,
                gp_lengthscale_mean=diagnostics.lengthscale_mean,
                gp_lengthscale_max=diagnostics.lengthscale_max,
                gp_output_scale=diagnostics.output_scale,
                previous_acquisition=(acquisition_history[-1] if acquisition_history else None),
                acquisition_history=tuple(acquisition_history),
                improvement_history=tuple(improvement_history),
            )
            acquisition = self.selector.select(state).upper()
            scores = acquisition_scores(
                acquisition,
                surrogate,
                features[available],
                targets,
                rng,
                xi=cfg.xi,
                beta=cfg.beta,
            )
            local_index = int(np.argmax(scores))
            selected = int(available[local_index])
            decision_tra = tra.value

            new_losses = self._evaluate(objective, x[selected])
            if len(new_losses) != n_windows:
                raise ValueError("objective returned an inconsistent number of windows")
            observed.append(selected)
            remaining.remove(selected)
            losses.append(new_losses)

            completed_queries = len(observed) - cfg.n_initial
            boundary = completed_queries % cfg.block_size == 0
            if boundary or len(observed) == cfg.budget:
                tra = compute_tra(np.asarray(losses), cfg.recent_count)
                tra_history.append((len(observed), tra.value))

            updated_targets = temporal_target(
                np.asarray(losses), cfg.recent_count, tra.value, cfg.eta
            )
            improved = bool(updated_targets[-1] < updated_targets[:-1].min())
            acquisition_history.append(acquisition)
            improvement_history.append(improved)
            incumbent_history.append(float(updated_targets.min()))
            incumbent_indices.append(observed[int(np.argmin(updated_targets))])
            trials.append(
                TrialRecord(
                    evaluation=len(observed),
                    candidate_index=selected,
                    candidate=tuple(float(value) for value in x[selected]),
                    window_losses=tuple(float(value) for value in new_losses),
                    target=float(updated_targets[-1]),
                    decision_tra=decision_tra,
                    updated_tra=tra.value,
                    acquisition=acquisition,
                    acquisition_score=float(scores[local_index]),
                    improved=improved,
                )
            )

        final_targets = temporal_target(
            np.asarray(losses), cfg.recent_count, tra.value, cfg.eta
        )
        best_local = int(np.argmin(final_targets))
        best_index = observed[best_local]
        calls = tuple(getattr(self.selector, "calls", ()))
        return OptimizationResult(
            best_index=best_index,
            best_candidate=tuple(float(value) for value in x[best_index]),
            best_target=float(final_targets[best_local]),
            final_tra=tra.value,
            observed_indices=tuple(observed),
            incumbent_indices=tuple(incumbent_indices),
            tra_history=tuple(tra_history),
            final_targets=tuple(float(value) for value in final_targets),
            trials=tuple(trials),
            llm_calls=calls,
        )

    def optimize_precomputed(
        self,
        candidates: ArrayLike,
        validation_losses: ArrayLike,
        initial_indices: Sequence[int] | None = None,
    ) -> OptimizationResult:
        """Optimize a candidate pool with precomputed visible-window losses.

        Rows in ``validation_losses`` must align with rows in ``candidates``.
        Future-window losses must not be included.
        """

        x = np.asarray(candidates, dtype=np.float64)
        losses = np.asarray(validation_losses, dtype=np.float64)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("candidates must be a finite matrix with shape [n, d]")
        if losses.ndim != 2 or not np.isfinite(losses).all():
            raise ValueError(
                "validation_losses must be a finite matrix with shape [n, K]"
            )
        if len(x) != len(losses):
            raise ValueError("candidates and validation_losses must have matching rows")

        keys = [tuple(float(value) for value in row) for row in x]
        if len(set(keys)) != len(keys):
            raise ValueError("precomputed optimization requires unique candidate rows")
        row_by_candidate = {key: index for index, key in enumerate(keys)}

        def objective(candidate: NDArray[np.float64]) -> NDArray[np.float64]:
            key = tuple(float(value) for value in candidate)
            return losses[row_by_candidate[key]]

        return self.optimize(x, objective, initial_indices=initial_indices)

    @staticmethod
    def _evaluate(
        objective: WindowObjective, candidate: NDArray[np.float64]
    ) -> tuple[float, ...]:
        values = np.asarray(objective(candidate.copy()), dtype=np.float64)
        if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("objective must return at least two finite window losses")
        return tuple(float(value) for value in values)
