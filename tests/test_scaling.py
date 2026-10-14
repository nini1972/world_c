import numpy as np
import pytest
from scipy import stats

from colony_lib.scaling import (
    TrajectorySummary,
    extract_finite_size_scaling,
    extract_finite_horizon_scaling,
    falsification_certificate,
)


def test_trajectory_summary_welford_moments():
    rng = np.random.RandomState(42)
    data = rng.normal(loc=5.0, scale=2.0, size=5000)

    summary = TrajectorySummary()
    summary.update_batch(data)

    assert summary.count == 5000
    assert np.isclose(summary.mean, np.mean(data), atol=1e-5)
    assert np.isclose(summary.variance, np.var(data, ddof=1), atol=1e-5)
    assert np.isclose(summary.std, np.std(data, ddof=1), atol=1e-5)
    assert np.isclose(summary.skewness, stats.skew(data), atol=0.1)
    assert np.isclose(summary.kurtosis, stats.kurtosis(data), atol=0.2)


def test_trajectory_summary_extremes():
    summary = TrajectorySummary()
    values = [10.0, 3.5, 8.2, -4.0, 15.0, 1.2]
    for v in values:
        summary.update(v)

    assert summary.min_val == -4.0
    assert summary.min_idx == 3
    assert summary.max_val == 15.0
    assert summary.max_idx == 4


def test_trajectory_summary_autocorrelation():
    # AR(1) process: x_t = 0.75 * x_{t-1} + noise
    rng = np.random.RandomState(123)
    N = 8000
    x = np.zeros(N)
    for t in range(1, N):
        x[t] = 0.75 * x[t - 1] + rng.normal(scale=0.5)

    summary = TrajectorySummary(target_lags=[1, 2, 5])
    summary.update_batch(x)

    ac = summary.autocorrelation()
    # Lag 1 autocorrelation should be close to 0.75
    assert np.isclose(ac[1], 0.75, atol=0.08)
    assert ac[2] < ac[1]
    assert ac[5] < ac[2]


def test_trajectory_summary_quantiles():
    rng = np.random.RandomState(999)
    uniform_data = rng.uniform(0.0, 100.0, size=10000)

    summary = TrajectorySummary(reservoir_size=1000, seed=42)
    summary.update_batch(uniform_data)

    q = summary.quantiles([0.25, 0.50, 0.75])
    assert np.isclose(q[0.25], 25.0, atol=3.0)
    assert np.isclose(q[0.50], 50.0, atol=3.0)
    assert np.isclose(q[0.75], 75.0, atol=3.0)

    d = summary.to_dict()
    assert "mean" in d
    assert "autocorrelation" in d
    assert "quantiles" in d


def test_extract_finite_size_scaling():
    # Synthetic law: O(N) = 1.65 + 3.2 * N^(-0.50)
    sizes = np.array([16, 32, 64, 128, 256, 512, 1024, 2048])
    true_y_inf = 1.65
    true_beta = 0.50
    true_a = 3.20

    rng = np.random.RandomState(42)
    noise = rng.normal(0, 0.005, size=len(sizes))
    observables = true_y_inf + true_a * (sizes ** -true_beta) + noise

    res = extract_finite_size_scaling(sizes, observables)
    assert res["success"] is True
    assert np.isclose(res["asymptotic_limit"], true_y_inf, atol=0.05)
    assert np.isclose(res["scaling_exponent"], true_beta, atol=0.08)
    assert res["r_squared"] > 0.98


def test_extract_finite_horizon_scaling_true_transition():
    # True non-zero transition K(T) = 1.80 + 12.0 / T^0.75
    horizons = np.array([50, 100, 250, 500, 1000, 2500, 5000])
    observables = 1.80 + 12.0 * (horizons ** -0.75)

    res = extract_finite_horizon_scaling(horizons, observables)
    assert res["success"] is True
    assert res["verdict"] == "TRUE_ASYMPTOTIC_TRANSITION"
    assert np.isclose(res["asymptotic_limit"], 1.80, atol=0.05)
    assert res["crossover_timescale_t_star"] > 0
    assert res["crossover_timescale_t_star"] < float("inf")


def test_extract_finite_horizon_scaling_rate_law_artifact():
    # Rate law artifact: K_app(T) = 8.5 / T (decays to 0)
    horizons = np.array([50, 100, 200, 500, 1000, 2500, 5000, 10000])
    observables = 8.5 / horizons

    res = extract_finite_horizon_scaling(horizons, observables)
    assert res["success"] is True
    assert res["verdict"] == "RATE_LAW_RELAXATION_ARTIFACT"
    assert np.isclose(res["asymptotic_limit"], 0.0, atol=0.05)
    assert res["crossover_timescale_t_star"] == float("inf")


def test_falsification_certificate():
    horizon_tuple = {"T": 10000, "N": 128, "dt": 0.01, "R0": 0.1, "seed": 42}
    scaling_res = {
        "verdict": "RATE_LAW_RELAXATION_ARTIFACT",
        "scaling_exponent": 0.98,
        "asymptotic_limit": 0.002,
        "r_squared": 0.998,
        "explanation": "K_app(T) decays as 8.5 / T^0.98",
    }
    cert = falsification_certificate(
        claim_id="HYP-KURAMOTO-ESCAPE-001",
        claim_statement="Apparent critical coupling K_c = 1.6 is a genuine phase transition",
        horizon_tuple=horizon_tuple,
        scaling_result=scaling_res,
        author_lineage="deepseek_v4_flash",
    )

    assert cert["certificate_type"] == "EPISTEMIC_HORIZON_CERTIFICATE"
    assert cert["claim_id"] == "HYP-KURAMOTO-ESCAPE-001"
    assert cert["verdict"] == "RATE_LAW_RELAXATION_ARTIFACT"
    assert len(cert["provenance_hash"]) == 64
