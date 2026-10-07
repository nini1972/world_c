"""
colony_lib.datasets: Canonical empirical real-world datasets for benchmark and simulation.
Fulfilling the collective Inquiry of Desires (Option A: Real-World Data).
"""

from .registry import (
    DATASETS_DIR,
    DATASET_CATALOG,
    list_datasets,
    get_dataset_info,
    get_all_dataset_metadata,
)

from .loaders import (
    Dataset,
    load_dataset,
    load_solar_sunspots,
    load_climate_enso,
    load_climate_temperatures,
    load_neural_eeg,
    load_lynx_hare,
)

__all__ = [
    "DATASETS_DIR",
    "DATASET_CATALOG",
    "list_datasets",
    "get_dataset_info",
    "get_all_dataset_metadata",
    "Dataset",
    "load_dataset",
    "load_solar_sunspots",
    "load_climate_enso",
    "load_climate_temperatures",
    "load_neural_eeg",
    "load_lynx_hare",
]
