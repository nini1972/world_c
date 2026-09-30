"""
Gaussian Process Surrogate & Uncertainty Quantification
Enables active learning, surrogate modeling, and boundary localization for heavy simulations.
"""
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

class GaussianProcessSurrogate:
    """
    Gaussian Process Regressor wrapper for physical simulation emulation.
    Fits continuous surrogate landscapes over sparse parameter grids,
    computes analytical uncertainty intervals (+/- 2 sigma), and suggests
    optimal exploration points using Bayesian acquisition (Maximum Uncertainty).
    """
    def __init__(
        self,
        kernel_type: str = "matern",
        length_scale: float = 1.0,
        nu: float = 2.5,
        noise_level: float = 1e-5
    ):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
        
        self.kernel_type = kernel_type
        if kernel_type == "matern":
            base_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2), nu=nu)
        else:
            base_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2))
            
        kernel = base_kernel + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-8, 1e-1))
        self.gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        self.X_train = None
        self.y_train = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessSurrogate":
        """Fits the GP model to parameter observations X and target values y."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.X_train = X
        self.y_train = y
        self.gpr.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X_new: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predicts mean and standard deviation at query points."""
        if not self.is_fitted:
            raise RuntimeError("Surrogate model must be fitted before predict.")
        X_new = np.asarray(X_new, dtype=np.float64)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        if return_std:
            mean, std = self.gpr.predict(X_new, return_std=True)
            return mean, std
        return self.gpr.predict(X_new, return_std=False), None

    def suggest_next_samples(self, candidate_pool: np.ndarray, n_samples: int = 3) -> np.ndarray:
        """
        Suggests the most informative parameter locations to sample next.
        strategy: Samples where epistemic variance sigma^2 is highest.
        """
        if not self.is_fitted:
            raise RuntimeError("Surrogate model must be fitted first.")
        candidate_pool = np.asarray(candidate_pool, dtype=np.float64)
        if candidate_pool.ndim == 1:
            candidate_pool = candidate_pool.reshape(-1, 1)
        _, std = self.predict(candidate_pool, return_std=True)
        top_indices = np.argsort(std)[::-1][:n_samples]
        return candidate_pool[top_indices]

    def find_critical_boundary(self, X_grid: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Locates the critical boundary contour where predicted mean crosses threshold.
        Returns predicted boundary points with confidence bounds.
        """
        mean, std = self.predict(X_grid, return_std=True)
        diff = np.abs(mean - threshold)
        closest_idx = np.argsort(diff)[:max(1, len(diff) // 20)]
        return {
            "boundary_points": X_grid[closest_idx].tolist(),
            "predicted_mean": mean[closest_idx].tolist(),
            "uncertainty_std": std[closest_idx].tolist(),
            "threshold": threshold
        }
