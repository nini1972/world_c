"""
TrajectorySummary: Online streaming statistical accumulator for long-horizon simulations.
Provides O(1) memory profiling of multi-million step trajectories:
- Exact online moments (Welford algorithm: mean, variance, skewness, kurtosis)
- Running extreme value tracking (min, max, excursions)
- Streaming multi-lag autocorrelation estimator
- Fixed-size reservoir sample for non-parametric quantile sketching
"""

import math
from typing import Dict, Any, List, Optional, Union
import numpy as np


class TrajectorySummary:
    """
    Online statistical accumulator for scalar time-series trajectories.
    Allows running 10^6 - 10^8 step simulations in constant memory (O(1)).
    """

    def __init__(
        self,
        target_lags: Optional[List[int]] = None,
        reservoir_size: int = 1000,
        seed: Optional[int] = 42,
    ):
        """
        Parameters
        ----------
        target_lags : list of int, optional
            Specific time lags at which to estimate autocorrelation (default [1, 2, 5, 10, 20, 50, 100]).
        reservoir_size : int, default 1000
            Size of the reservoir sample for online quantile and histogram estimation.
        seed : int, optional
            RNG seed for reservoir sampling.
        """
        self.count = 0
        self.target_lags = sorted(target_lags or [1, 2, 5, 10, 20, 50, 100])
        self.max_lag = max(self.target_lags) if self.target_lags else 0
        self.reservoir_size = reservoir_size

        # Welford running moments
        self.mean = 0.0
        self.M2 = 0.0
        self.M3 = 0.0
        self.M4 = 0.0

        # Extremes
        self.min_val = float("inf")
        self.max_val = float("-inf")
        self.min_idx = -1
        self.max_idx = -1

        # Online circular buffer for lagged cross-products
        self._buffer_len = self.max_lag + 1
        self._history = [0.0] * self._buffer_len
        self._lag_cross_sum = {k: 0.0 for k in self.target_lags}
        self._lag_counts = {k: 0 for k in self.target_lags}

        # Reservoir sampling (Algorithm R)
        self._reservoir = np.zeros(self.reservoir_size, dtype=float)
        self._rng = np.random.RandomState(seed)

    def update(self, x: float) -> None:
        """Accumulates a single scalar observation into the running summary."""
        x = float(x)
        idx = self.count
        self.count += 1
        n = self.count

        # Extreme values
        if x < self.min_val:
            self.min_val = x
            self.min_idx = idx
        if x > self.max_val:
            self.max_val = x
            self.max_idx = idx

        # Welford update for mean, variance, skewness, and kurtosis
        delta = x - self.mean
        delta_n = delta / n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * (n - 1)

        self.mean += delta_n
        self.M4 += term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * self.M2 - 4 * delta_n * self.M3
        self.M3 += term1 * delta_n * (n - 2) - 3 * delta_n * self.M2
        self.M2 += term1

        # Lagged correlations via circular buffer
        buf_pos = idx % self._buffer_len
        self._history[buf_pos] = x

        for k in self.target_lags:
            if idx >= k:
                past_pos = (idx - k) % self._buffer_len
                past_val = self._history[past_pos]
                self._lag_cross_sum[k] += past_val * x
                self._lag_counts[k] += 1

        # Reservoir sampling
        if idx < self.reservoir_size:
            self._reservoir[idx] = x
        else:
            j = self._rng.randint(0, n)
            if j < self.reservoir_size:
                self._reservoir[j] = x

    def update_batch(self, arr: Union[np.ndarray, List[float]]) -> None:
        """Batch update for an array of sequential values."""
        for val in arr:
            self.update(val)

    @property
    def variance(self) -> float:
        """Sample variance s^2."""
        if self.count < 2:
            return 0.0
        return self.M2 / (self.count - 1)

    @property
    def std(self) -> float:
        """Sample standard deviation s."""
        return math.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        """Fisher-Pearson sample skewness."""
        if self.count < 3 or self.M2 <= 1e-15:
            return 0.0
        n = self.count
        return (math.sqrt(n) * self.M3) / (self.M2 ** 1.5)

    @property
    def kurtosis(self) -> float:
        """Excess kurtosis (normal distribution = 0.0)."""
        if self.count < 4 or self.M2 <= 1e-15:
            return 0.0
        n = self.count
        return (n * self.M4) / (self.M2 * self.M2) - 3.0

    def autocorrelation(self) -> Dict[int, float]:
        """
        Computes sample autocorrelation rho(k) at all target lags k.
        rho(k) = Cov(x_t, x_{t-k}) / Var(x)
        """
        var = self.variance
        if var <= 1e-15:
            return {k: 0.0 for k in self.target_lags}

        mu = self.mean
        ac = {}
        for k in self.target_lags:
            cnt = self._lag_counts[k]
            if cnt > 0:
                # E[X_t * X_{t-k}] - mu^2
                cov = (self._lag_cross_sum[k] / cnt) - (mu * mu)
                ac[k] = float(np.clip(cov / var, -1.0, 1.0))
            else:
                ac[k] = 0.0
        return ac

    def quantiles(self, q_list: Optional[List[float]] = None) -> Dict[float, float]:
        """
        Estimates distribution quantiles from the online reservoir sketch.
        """
        if q_list is None:
            q_list = [0.05, 0.25, 0.50, 0.75, 0.95]
        active_count = min(self.count, self.reservoir_size)
        if active_count == 0:
            return {q: 0.0 for q in q_list}

        samples = self._reservoir[:active_count]
        res = {}
        for q in q_list:
            res[q] = float(np.percentile(samples, q * 100))
        return res

    def to_dict(self) -> Dict[str, Any]:
        """Returns comprehensive summary metadata dictionary."""
        return {
            "n_steps": self.count,
            "mean": self.mean,
            "variance": self.variance,
            "std": self.std,
            "skewness": self.skewness,
            "excess_kurtosis": self.kurtosis,
            "min": self.min_val if self.count > 0 else 0.0,
            "max": self.max_val if self.count > 0 else 0.0,
            "min_timestep": self.min_idx,
            "max_timestep": self.max_idx,
            "autocorrelation": self.autocorrelation(),
            "quantiles": self.quantiles(),
        }

    def __repr__(self) -> str:
        return (
            f"<TrajectorySummary n={self.count} "
            f"mean={self.mean:.4f} std={self.std:.4f} "
            f"range=[{self.min_val:.4f}, {self.max_val:.4f}]>"
        )
