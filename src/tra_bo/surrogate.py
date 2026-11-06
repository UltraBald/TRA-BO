"""Gaussian-process surrogate used by TRA-BO."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


@dataclass(frozen=True)
class GPDiagnostics:
    """Compact GP state supplied to the acquisition selector."""

    lengthscale_min: float
    lengthscale_mean: float
    lengthscale_max: float
    output_scale: float


class MaternGaussianProcess:
    """ARD Matérn-5/2 GP for a minimization objective."""

    def __init__(
        self,
        dimension: int,
        random_state: int,
        noise: float = 1e-6,
        n_restarts: int = 2,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(dimension),
            length_scale_bounds=(1e-3, 1e3),
            nu=2.5,
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=noise,
            normalize_y=True,
            n_restarts_optimizer=n_restarts,
            random_state=random_state,
        )
        self._fitted = False

    def fit(self, x: ArrayLike, y: ArrayLike) -> "MaternGaussianProcess":
        features = np.asarray(x, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        if features.ndim != 2 or target.ndim != 1:
            raise ValueError("x and y must have shapes [n, d] and [n]")
        if len(features) != len(target):
            raise ValueError("x and y must contain the same number of rows")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(features, target)
        self._fitted = True
        return self

    def predict(
        self, x: ArrayLike, return_std: bool = False
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        if not self._fitted:
            raise RuntimeError("the GP must be fitted before prediction")
        features = np.asarray(x, dtype=np.float64)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            prediction = self.model.predict(features, return_std=return_std)
        arrays = prediction if isinstance(prediction, tuple) else (prediction,)
        if not all(np.isfinite(values).all() for values in arrays):
            raise FloatingPointError("the GP produced a non-finite prediction")
        return prediction

    def sample(self, x: ArrayLike, rng: np.random.Generator) -> NDArray[np.float64]:
        """Draw one joint posterior function sample."""

        if not self._fitted:
            raise RuntimeError("the GP must be fitted before sampling")
        features = np.asarray(x, dtype=np.float64)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            mean, covariance = self.model.predict(features, return_cov=True)
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            raise FloatingPointError("the GP produced a non-finite posterior")
        covariance = np.asarray(covariance, dtype=np.float64)
        covariance.flat[:: len(covariance) + 1] += 1e-10
        return rng.multivariate_normal(np.asarray(mean), covariance)

    @property
    def diagnostics(self) -> GPDiagnostics:
        if not self._fitted:
            raise RuntimeError("the GP must be fitted before reading diagnostics")
        kernel = self.model.kernel_
        lengthscales = np.atleast_1d(np.asarray(kernel.k2.length_scale, dtype=float))
        return GPDiagnostics(
            lengthscale_min=float(lengthscales.min()),
            lengthscale_mean=float(lengthscales.mean()),
            lengthscale_max=float(lengthscales.max()),
            output_scale=float(kernel.k1.constant_value),
        )
