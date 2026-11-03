import numpy as np
import pytest

from colony_lib.datasets import (
    list_datasets,
    get_dataset_info,
    get_all_dataset_metadata,
    load_dataset,
    load_solar_sunspots,
    load_climate_enso,
    load_climate_temperatures,
    load_neural_eeg,
    load_lynx_hare,
    load_power_grid_ieee,
    load_belousov_zhabotinsky,
)
from colony_lib.recurrence.takens import takens_embedding, estimate_delay_autocorr
from colony_lib.recurrence.rqa import recurrence_matrix, compute_rqa_metrics


def test_catalog_and_registry():
    names = list_datasets()
    expected = [
        "solar_sunspots",
        "climate_enso",
        "climate_temperatures",
        "neural_eeg",
        "lynx_hare",
        "power_grid_ieee",
        "belousov_zhabotinsky",
    ]
    for exp in expected:
        assert exp in names

    meta_all = get_all_dataset_metadata()
    assert len(meta_all) == len(expected)

    info = get_dataset_info("solar_sunspots")
    assert info["domain"] == "astrophysics"
    assert "primary_signal" in info
    assert len(info["recommended_probes"]) > 0

    with pytest.raises(KeyError):
        get_dataset_info("nonexistent_dataset_xyz")


def test_load_solar_sunspots():
    ds = load_solar_sunspots()
    assert ds.name == "solar_sunspots"
    assert len(ds.primary_signal) > 3300
    assert len(ds.time) == len(ds.primary_signal)
    assert ds.time[0] < 1750.0
    assert ds.time[-1] >= 2025.0
    assert "sunspot_number" in ds.columns

    # Check dict access
    assert np.allclose(ds["sunspot_number"], ds.primary_signal)
    assert np.isclose(np.mean(ds.normalized), 0.0, atol=1e-5)
    assert np.isclose(np.std(ds.normalized), 1.0, atol=1e-5)


def test_load_climate_enso():
    ds = load_climate_enso()
    assert ds.name == "climate_enso"
    assert len(ds.primary_signal) > 850
    assert "nino34_anom" in ds.columns
    assert ds.data.shape[1] == 10
    assert np.allclose(ds["nino34_anom"], ds.primary_signal)
    assert np.isclose(np.mean(ds.normalized), 0.0, atol=1e-5)
    assert np.isclose(np.std(ds.normalized), 1.0, atol=1e-5)

    # Test column indexing on DatasetDataArray and Dataset container (SOS-2026-11-03-mistral_large-575ed6 fix)
    assert np.allclose(ds.data["Nino3.4"], ds.primary_signal)
    assert np.allclose(ds["data"]["Nino3.4"], ds.primary_signal)
    assert np.allclose(ds["Nino3.4"], ds.primary_signal)

    regions = ["Nino1+2", "Nino3", "Nino3.4", "Nino4"]
    multi_col_data = ds.data[regions]
    assert multi_col_data.shape == (len(ds.primary_signal), 4)
    assert ds["data"][regions].shape == (len(ds.primary_signal), 4)
    assert ds[regions].shape == (len(ds.primary_signal), 4)


def test_load_climate_temperatures():
    ds = load_climate_temperatures()
    assert ds.name == "climate_temperatures"
    assert len(ds.primary_signal) == 3650
    assert len(ds.time) == 3650
    assert np.isclose(np.mean(ds.normalized), 0.0, atol=1e-5)
    assert np.isclose(np.std(ds.normalized), 1.0, atol=1e-5)


def test_load_neural_eeg():
    ds = load_neural_eeg()
    assert ds.name == "neural_eeg"
    assert len(ds.primary_signal) == 14980
    assert ds.data.shape == (14980, 15)
    assert "AF3" in ds.columns
    assert "eyeDetection" in ds.columns
    # Sampling 128 Hz: dt = 1/128
    assert np.isclose(ds.time[1] - ds.time[0], 1.0 / 128.0)
    assert np.isclose(np.mean(ds.normalized), 0.0, atol=1e-5)
    assert np.isclose(np.std(ds.normalized), 1.0, atol=1e-5)


def test_load_lynx_hare():
    ds = load_lynx_hare()
    assert ds.name == "lynx_hare"
    assert len(ds.primary_signal) == 21
    assert "lynx" in ds.columns
    assert "hare" in ds.columns
    assert np.allclose(ds["hare"], ds.primary_signal)
    assert ds.time[0] == 1900.0
    assert ds.time[-1] == 1920.0


def test_load_power_grid_ieee():
    ds = load_power_grid_ieee()
    assert ds.name == "power_grid_ieee"
    assert len(ds.primary_signal) == 1000
    assert "kuramoto_r" in ds.columns
    assert "bus1_freq_hz" in ds.columns
    assert np.all((ds.primary_signal >= 0.0) & (ds.primary_signal <= 1.0))
    assert np.isclose(np.mean(ds.normalized), 0.0, atol=1e-5)


def test_load_belousov_zhabotinsky():
    ds = load_belousov_zhabotinsky()
    assert ds.name == "belousov_zhabotinsky"
    assert len(ds.primary_signal) == 1000
    assert "redox_potential_mv" in ds.columns
    assert "activator_u" in ds.columns
    assert np.isclose(np.mean(ds.normalized), 0.0, atol=1e-5)


def test_generic_load_dataset():
    for name in list_datasets():
        ds = load_dataset(name)
        assert ds.name == name
        assert len(ds.primary_signal) > 0
        assert len(ds.normalized) == len(ds.primary_signal)

    with pytest.raises(KeyError):
        load_dataset("unknown_universe")


def test_as_dataframe():
    df = load_dataset("lynx_hare", as_dataframe=True)
    assert hasattr(df, "columns")
    assert "hare" in df.columns
    assert len(df) == 21

    ds = load_dataset("lynx_hare", as_dataframe=False)
    df_converted = ds.to_dataframe()
    assert hasattr(df_converted, "columns")
    assert len(df_converted) == 21


def test_integration_with_recurrence_takens_rqa():
    """
    Simulates an Agora / Colony agent workflow:
    Load empirical sunspot data, calculate optimal delay tau,
    perform Takens phase space embedding, and compute RQA metrics.
    """
    ds = load_solar_sunspots()
    series = ds.normalized[:300]  # first 25 years

    tau = estimate_delay_autocorr(series, max_lag=50)
    assert tau >= 1

    embedded = takens_embedding(series, m=3, tau=tau)
    assert embedded.shape[1] == 3
    assert embedded.shape[0] == len(series) - 2 * tau

    R = recurrence_matrix(embedded, epsilon=0.5)
    metrics = compute_rqa_metrics(R)
    assert 0.0 <= metrics["recurrence_rate"] <= 1.0
    assert 0.0 <= metrics["determinism"] <= 1.0
    assert metrics["max_diag_length"] >= 0
