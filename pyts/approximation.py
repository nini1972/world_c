import numpy as np
from scipy.stats import norm


class SymbolicAggregateApproximation:
    def __init__(self, n_bins=4, strategy="quantile", alphabet=None):
        self.n_bins = int(n_bins)
        self.strategy = strategy
        self.alphabet = alphabet

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        if single:
            X = X[None, :]
        alphabet = (
            [chr(97 + i) for i in range(self.n_bins)]
            if self.alphabet is None
            else list(self.alphabet)
        )
        out = np.empty(X.shape, dtype="<U1")
        for idx, row in enumerate(X):
            std = np.std(row)
            z = (row - np.mean(row)) / (std if std > 1e-12 else 1.0)
            if self.strategy == "normal":
                edges = norm.ppf(np.linspace(0, 1, self.n_bins + 1)[1:-1])
            elif self.strategy == "uniform":
                edges = np.linspace(np.min(z), np.max(z), self.n_bins + 1)[1:-1]
            else:
                edges = np.quantile(z, np.linspace(0, 1, self.n_bins + 1)[1:-1])
            bins = np.digitize(z, edges)
            bins = np.clip(bins, 0, self.n_bins - 1)
            out[idx] = [alphabet[b] for b in bins]
        return out[0] if single else out

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
