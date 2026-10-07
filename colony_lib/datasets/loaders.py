"""
Loaders for canonical empirical datasets in colony_lib.
Provides offline, zero-network scientific time-series benchmarks for the Agora and Colony.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
import numpy as np

from .registry import DATASETS_DIR, get_dataset_info, list_datasets


@dataclass
class Dataset:
    """
    Structured container for an empirical time-series dataset.
    Supports both object attribute access (.primary_signal) and dict access (['primary_signal']).
    """
    name: str
    metadata: Dict[str, Any]
    columns: List[str]
    time: np.ndarray
    data: np.ndarray
    primary_signal: np.ndarray
    normalized: np.ndarray

    def to_dataframe(self):
        """Converts dataset into a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required to convert dataset to a DataFrame.")
        return pd.DataFrame(self.data, columns=self.columns)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.columns:
            idx = self.columns.index(key)
            return self.data[:, idx]
        raise KeyError(f"Key '{key}' not found in Dataset. Available attributes and columns: {list(self.__dict__.keys()) + self.columns}")

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or (key in self.columns)

    def __repr__(self) -> str:
        return (
            f"<Dataset name='{self.name}' "
            f"domain='{self.metadata.get('domain', 'unknown')}' "
            f"shape={self.data.shape} "
            f"primary_signal='{self.metadata.get('primary_signal', '')}'>"
        )


def _load_csv_raw(filepath: str):
    """
    Internal robust CSV loader using pandas if available, falling back to standard library.
    """
    try:
        import pandas as pd
        df = pd.read_csv(filepath, skip_blank_lines=True)
        # Clean column names
        df.columns = [c.strip().strip('"').strip("'") for c in df.columns]
        return df
    except ImportError:
        import csv
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = [row for row in reader if row and any(cell.strip() for cell in row)]
        headers = [c.strip().strip('"').strip("'") for c in rows[0]]
        data_rows = rows[1:]
        return headers, data_rows


def _zscore(series: np.ndarray) -> np.ndarray:
    """Computes standard z-score normalization (mean=0, std=1)."""
    std = np.std(series)
    if std < 1e-12:
        return np.zeros_like(series)
    return (series - np.mean(series)) / std


def load_dataset(name: str, as_dataframe: bool = False) -> Union[Dataset, Any]:
    """
    Generic loader for empirical datasets by name.

    Parameters
    ----------
    name : str
        Dataset identifier (e.g. 'solar_sunspots', 'climate_enso', 'neural_eeg', etc.)
    as_dataframe : bool, default False
        If True and pandas is available, returns a pandas.DataFrame.
        Otherwise returns a Dataset dataclass container.

    Returns
    -------
    Dataset or pandas.DataFrame
    """
    name_clean = name.strip().lower()
    loaders = {
        "solar_sunspots": load_solar_sunspots,
        "climate_enso": load_climate_enso,
        "climate_temperatures": load_climate_temperatures,
        "neural_eeg": load_neural_eeg,
        "lynx_hare": load_lynx_hare,
    }
    if name_clean in loaders:
        return loaders[name_clean](as_dataframe=as_dataframe)
    raise KeyError(f"Unknown dataset '{name}'. Available: {list_datasets()}")


def load_solar_sunspots(as_dataframe: bool = False) -> Union[Dataset, Any]:
    """
    Loads monthly solar sunspot numbers (1749-2026, Royal Observatory of Belgium).
    Captures non-linear solar dynamo cycles, Schwabe ~11-yr periodicity, and grand minima.
    """
    meta = get_dataset_info("solar_sunspots")
    filepath = os.path.join(DATASETS_DIR, meta["filename"])
    raw = _load_csv_raw(filepath)

    if hasattr(raw, "to_numpy"):
        df = raw
        if as_dataframe:
            return df
        columns = list(df.columns)
        time = df["decimal_year"].to_numpy(dtype=float)
        primary = df["sunspot_number"].to_numpy(dtype=float)
        data = df.to_numpy(dtype=float)
    else:
        headers, rows = raw
        columns = headers
        arr = np.array(rows, dtype=float)
        time = arr[:, columns.index("decimal_year")]
        primary = arr[:, columns.index("sunspot_number")]
        data = arr

    norm = _zscore(primary)
    return Dataset(
        name="solar_sunspots",
        metadata=meta,
        columns=columns,
        time=time,
        data=data,
        primary_signal=primary,
        normalized=norm,
    )


def load_climate_enso(as_dataframe: bool = False) -> Union[Dataset, Any]:
    """
    Loads equatorial Pacific Sea Surface Temperature (SST) & El Niño 3.4 anomalies (1950-2026, NOAA).
    Captures coupled ocean-atmosphere delay-oscillator dynamics and climate regime shifts.
    """
    meta = get_dataset_info("climate_enso")
    filepath = os.path.join(DATASETS_DIR, meta["filename"])
    raw = _load_csv_raw(filepath)

    if hasattr(raw, "to_numpy"):
        df = raw
        if as_dataframe:
            return df
        columns = list(df.columns)
        time = df["year"].to_numpy(dtype=float) + (df["month"].to_numpy(dtype=float) - 0.5) / 12.0
        primary = df["nino34_anom"].to_numpy(dtype=float)
        data = df.to_numpy(dtype=float)
    else:
        headers, rows = raw
        columns = headers
        arr = np.array(rows, dtype=float)
        year = arr[:, columns.index("year")]
        month = arr[:, columns.index("month")]
        time = year + (month - 0.5) / 12.0
        primary = arr[:, columns.index("nino34_anom")]
        data = arr

    norm = _zscore(primary)
    return Dataset(
        name="climate_enso",
        metadata=meta,
        columns=columns,
        time=time,
        data=data,
        primary_signal=primary,
        normalized=norm,
    )


def load_climate_temperatures(as_dataframe: bool = False) -> Union[Dataset, Any]:
    """
    Loads daily minimum surface temperatures (10-year continuous series).
    Captures non-stationary atmospheric boundary layer variance and stochastic turbulence.
    """
    meta = get_dataset_info("climate_temperatures")
    filepath = os.path.join(DATASETS_DIR, meta["filename"])
    raw = _load_csv_raw(filepath)

    if hasattr(raw, "to_numpy"):
        df = raw
        if as_dataframe:
            return df
        columns = list(df.columns)
        time = np.arange(len(df), dtype=float)
        primary = df["Temp"].to_numpy(dtype=float)
        data = np.column_stack([time, primary])
    else:
        headers, rows = raw
        columns = headers
        time = np.arange(len(rows), dtype=float)
        primary = np.array([float(r[columns.index("Temp")]) for r in rows], dtype=float)
        data = np.column_stack([time, primary])

    norm = _zscore(primary)
    return Dataset(
        name="climate_temperatures",
        metadata=meta,
        columns=["DayIndex", "Temp"],
        time=time,
        data=data,
        primary_signal=primary,
        normalized=norm,
    )


def load_neural_eeg(as_dataframe: bool = False) -> Union[Dataset, Any]:
    """
    Loads 14-channel cortical scalp EEG at 128 Hz (UCI Machine Learning Repository).
    Captures macroscopic neural population synchronization and collective phase transitions.
    """
    meta = get_dataset_info("neural_eeg")
    filepath = os.path.join(DATASETS_DIR, meta["filename"])
    raw = _load_csv_raw(filepath)

    if hasattr(raw, "to_numpy"):
        df = raw
        if as_dataframe:
            return df
        columns = list(df.columns)
        primary = df["AF3"].to_numpy(dtype=float)
        data = df.to_numpy(dtype=float)
    else:
        headers, rows = raw
        columns = headers
        arr = np.array(rows, dtype=float)
        primary = arr[:, columns.index("AF3")]
        data = arr

    time = np.arange(len(primary), dtype=float) / 128.0  # 128 Hz sampling -> seconds
    norm = _zscore(primary)
    return Dataset(
        name="neural_eeg",
        metadata=meta,
        columns=columns,
        time=time,
        data=data,
        primary_signal=primary,
        normalized=norm,
    )


def load_lynx_hare(as_dataframe: bool = False) -> Union[Dataset, Any]:
    """
    Loads Hudson's Bay Company predator-prey pelt collections (Elton & Nicholson 1942).
    Canonical empirical ecological limit cycle with ~2-year trophic phase lag.
    """
    meta = get_dataset_info("lynx_hare")
    filepath = os.path.join(DATASETS_DIR, meta["filename"])
    raw = _load_csv_raw(filepath)

    if hasattr(raw, "to_numpy"):
        df = raw
        if as_dataframe:
            return df
        columns = list(df.columns)
        time = df["year"].to_numpy(dtype=float)
        primary = df["hare"].to_numpy(dtype=float)
        data = df.to_numpy(dtype=float)
    else:
        headers, rows = raw
        columns = headers
        arr = np.array(rows, dtype=float)
        time = arr[:, columns.index("year")]
        primary = arr[:, columns.index("hare")]
        data = arr

    norm = _zscore(primary)
    return Dataset(
        name="lynx_hare",
        metadata=meta,
        columns=columns,
        time=time,
        data=data,
        primary_signal=primary,
        normalized=norm,
    )
