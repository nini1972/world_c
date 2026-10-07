"""
Registry and metadata for canonical empirical datasets in colony_lib.
Provides external reality benchmarks for the Existential Evolution Colony and Agora.
"""

import os
from typing import Dict, Any, List

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "data")

DATASET_CATALOG: Dict[str, Dict[str, Any]] = {
    "solar_sunspots": {
        "name": "solar_sunspots",
        "title": "Monthly Mean Total Sunspot Number (SILSO)",
        "domain": "astrophysics",
        "source": "Royal Observatory of Belgium, WDC-SILSO (1749-2026)",
        "filename": "solar_sunspots.csv",
        "sampling": "Monthly",
        "time_range": "1749 - 2026 (3,300+ months)",
        "columns": ["year", "month", "decimal_year", "sunspot_number", "std_dev", "n_observations"],
        "primary_signal": "sunspot_number",
        "physical_system": "Solar Magnetohydrodynamic Dynamo",
        "description": (
            "Monthly smoothed and total sunspot numbers spanning 277 years. "
            "Exhibits asymmetric ~11-year Schwabe cycles, Maunder-like grand minima modulation, "
            "and deterministic chaotic dynamics driven by the non-linear solar magnetic dynamo."
        ),
        "recommended_probes": [
            "Takens Embedding & Phase Space Reconstruction",
            "Recurrence Quantification Analysis (RQA)",
            "Maximum Lyapunov Exponent Estimation",
            "Fourier & Wavelet Power Spectrum Analysis",
            "Adler / Kuramoto Phase-Locking to External Forcing"
        ]
    },
    "climate_enso": {
        "name": "climate_enso",
        "title": "El Niño Southern Oscillation (Niño 3.4 SST & Anomalies)",
        "domain": "climate",
        "source": "NOAA Climate Prediction Center (CPC) ERSSTv5 (1950-2026)",
        "filename": "climate_enso.csv",
        "sampling": "Monthly",
        "time_range": "1950 - 2026 (850+ months)",
        "columns": [
            "year", "month", "nino12_sst", "nino12_anom",
            "nino3_sst", "nino3_anom", "nino4_sst", "nino4_anom",
            "nino34_sst", "nino34_anom"
        ],
        "primary_signal": "nino34_anom",
        "physical_system": "Coupled Atmosphere-Ocean Delayed Oscillator",
        "description": (
            "Sea surface temperature (SST) anomalies across the equatorial Pacific. "
            "The Niño 3.4 index captures canonical El Niño / La Niña events arising from "
            "the non-linear delayed coupling between fast atmospheric winds and slow oceanic thermocline waves."
        ),
        "recommended_probes": [
            "Delayed Differential Equation (DDE) Parameter Fitting",
            "Cross-Correlation with Fast/Slow Timescale Separation",
            "Bifurcation & Tipping Point Early Warning Signals",
            "Entropy & Non-Gaussian Tail Analysis"
        ]
    },
    "climate_temperatures": {
        "name": "climate_temperatures",
        "title": "Daily Surface Minimum Temperatures",
        "domain": "meteorology",
        "source": "Australian Bureau of Meteorology (Melbourne Station)",
        "filename": "climate_temperatures.csv",
        "sampling": "Daily",
        "time_range": "10-Year Daily Continuous Series (3,650 points)",
        "columns": ["Date", "Temp"],
        "primary_signal": "Temp",
        "physical_system": "Atmospheric Planetary Boundary Layer Dynamics",
        "description": (
            "Multi-year daily minimum surface temperature observations. "
            "Combines periodic annual orbital forcing with multiscale stochastic weather fluctuations "
            "and non-stationary seasonal variance."
        ),
        "recommended_probes": [
            "Seasonal Trend Decomposition",
            "Autocorrelation & Memory Decay (Hurst Exponent)",
            "Kolmogorov Turbulence Cascade Signatures",
            "Extreme Value Statistics"
        ]
    },
    "neural_eeg": {
        "name": "neural_eeg",
        "title": "14-Channel Cortical Scalp EEG",
        "domain": "neuroscience",
        "source": "University of California Irvine (UCI) Machine Learning Repository",
        "filename": "neural_eeg.csv",
        "sampling": "128 Hz (continuous time-series)",
        "time_range": "14,980 timesteps (approx. 117 seconds continuous)",
        "columns": ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4", "eyeDetection"],
        "primary_signal": "AF3",
        "physical_system": "Macroscopic Neural Population Synchronization",
        "description": (
            "Synchronized 14-channel electroencephalogram recording of human cortical activity "
            "spanning frontal, temporal, parietal, and occipital lobes. "
            "Features neural rhythms, eye state transitions, and spatial coherence across brain regions."
        ),
        "recommended_probes": [
            "Spatial Synchronization & Kuramoto Order Parameter",
            "Information Theoretic Transfer Entropy Between Channels",
            "Permutation Entropy & Complexity-Entropy Causality Planes",
            "Phase-Amplitude Coupling and Critical Avalanche Scaling"
        ]
    },
    "lynx_hare": {
        "name": "lynx_hare",
        "title": "Hudson's Bay Company Predator-Prey Pelt Collection",
        "domain": "ecology",
        "source": "Hudson's Bay Company Historical Records (Elton & Nicholson 1942)",
        "filename": "lynx_hare.csv",
        "sampling": "Annual",
        "time_range": "1900 - 1920 (Extended cycle benchmark)",
        "columns": ["year", "lynx", "hare"],
        "primary_signal": "hare",
        "physical_system": "Non-Linear Predator-Prey Trophic Cascade",
        "description": (
            "Annual population proxy counts for Canadian lynx and snowshoe hare. "
            "The empirical foundation for Lotka-Volterra limit cycles, showing persistent ~9-10 year "
            "oscillations with a characteristic ~2-year trophic phase delay."
        ),
        "recommended_probes": [
            "Limit Cycle Geometry & Vector Field Reconstruction",
            "Lotka-Volterra & Holling Type II Functional Response Fitting",
            "Phase Portrait Trajectory Curvature",
            "Nonlinear Damping & Carrying Capacity Identification"
        ]
    }
}

def list_datasets() -> List[str]:
    """Returns the list of all available empirical dataset identifiers."""
    return list(DATASET_CATALOG.keys())

def get_dataset_info(name: str) -> Dict[str, Any]:
    """Retrieves metadata descriptor for a given dataset."""
    name_clean = name.strip().lower()
    if name_clean not in DATASET_CATALOG:
        raise KeyError(
            f"Dataset '{name}' not found. Available datasets: {list_datasets()}"
        )
    return DATASET_CATALOG[name_clean]

def get_all_dataset_metadata() -> Dict[str, Dict[str, Any]]:
    """Returns full catalog metadata for all registered datasets."""
    return DATASET_CATALOG.copy()
