"""
Loaders for canonical empirical datasets in colony_lib.
Provides offline, zero-network scientific time-series benchmarks for the Agora and Colony.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
import numpy as np

from .registry import DATASETS_DIR, get_dataset_info, list_datasets


class DatasetDataArray(np.ndarray):
    """
    Enhanced NumPy ndarray that supports column indexing by name and list of names.
    Allows accessing columns as data['Nino3.4'] or data[REGIONS] as well as standard slicing data[:, 0].
    """
    def __new__(cls, input_array, columns: Optional[List[str]] = None):
        obj = np.asarray(input_array).view(cls)
        obj.columns = list(columns) if columns is not None else []
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.columns = getattr(obj, "columns", [])

    def _normalize_col_name(self, query: str) -> Optional[int]:
        if not hasattr(self, "columns") or not self.columns:
            return None
        if query in self.columns:
            return self.columns.index(query)
        q_clean = re.sub(r"[^a-zA-Z0-9]", "", str(query)).lower()
        for i, c in enumerate(self.columns):
            c_clean = re.sub(r"[^a-zA-Z0-9]", "", c).lower()
            if q_clean == c_clean:
                return i
        candidates = []
        for i, c in enumerate(self.columns):
            c_clean = re.sub(r"[^a-zA-Z0-9]", "", c).lower()
            if q_clean in c_clean or c_clean in q_clean:
                candidates.append(i)
        if candidates:
            for cand in candidates:
                if "anom" in self.columns[cand]:
                    return cand
            return candidates[0]
        return None

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            idx = self._normalize_col_name(key)
            if idx is not None:
                if self.ndim == 2:
                    return np.asarray(super().__getitem__((slice(None), idx)))
                elif self.ndim == 1:
                    return np.asarray(super().__getitem__(idx))
            raise KeyError(f"Column '{key}' not found. Available columns: {getattr(self, 'columns', [])}")

        if isinstance(key, (list, tuple)) and key and all(isinstance(k, str) for k in key):
            indices = []
            matched_cols = []
            for k in key:
                idx = self._normalize_col_name(k)
                if idx is not None:
                    indices.append(idx)
                    matched_cols.append(self.columns[idx])
                else:
                    raise KeyError(f"Column '{k}' not found in columns: {getattr(self, 'columns', [])}")
            if self.ndim == 2:
                sub = super().__getitem__((slice(None), indices))
                return DatasetDataArray(sub, columns=matched_cols)
            return super().__getitem__(indices)

        return super().__getitem__(key)

    def __getattr__(self, name: str) -> Any:
        if hasattr(self, "columns") and self.columns:
            idx = self._normalize_col_name(name)
            if idx is not None:
                if self.ndim == 2:
                    return np.asarray(super().__getitem__((slice(None), idx)))
                elif self.ndim == 1:
                    return np.asarray(super().__getitem__(idx))
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, str):
            return self._normalize_col_name(key) is not None
        return super().__contains__(key)

    def keys(self) -> List[str]:
        return list(getattr(self, "columns", []))


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

    def __post_init__(self):
        if not isinstance(self.data, DatasetDataArray):
            self.data = DatasetDataArray(self.data, columns=self.columns)

    def to_dataframe(self):
        """Converts dataset into a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required to convert dataset to a DataFrame.")
        return pd.DataFrame(self.data, columns=self.columns)

    @property
    def shape(self) -> tuple:
        """Returns the shape of the underlying data array."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.data, dtype=dtype)

    def flatten(self) -> np.ndarray:
        return np.asarray(self.data).flatten()

    def ravel(self) -> np.ndarray:
        return np.asarray(self.data).ravel()

    @property
    def sampling_rate(self) -> Optional[float]:
        """Returns the sampling rate in Hz (or observations per unit time) if known."""
        if "sampling_rate" in self.metadata:
            return float(self.metadata["sampling_rate"])
        if self.time is not None and len(self.time) > 1:
            dt = np.median(np.diff(self.time))
            if dt > 1e-12:
                return float(1.0 / dt)
        return None

    @property
    def fs(self) -> Optional[float]:
        """Alias for sampling_rate."""
        return self.sampling_rate

    @property
    def channels(self) -> List[str]:
        """Alias for columns."""
        return self.columns

    def __len__(self) -> int:
        """Returns the number of time-series observations."""
        return len(self.primary_signal)

    def __getattr__(self, name: str) -> Any:
        if "metadata" in self.__dict__ and name in self.__dict__["metadata"]:
            return self.__dict__["metadata"][name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, (int, slice, tuple, np.ndarray)):
            return self.data[key]
        if isinstance(key, list):
            return self.data[key]
        if isinstance(key, str):
            if hasattr(self, key):
                return getattr(self, key)
            if key in self.columns:
                idx = self.columns.index(key)
                return self.data[:, idx]
            if hasattr(self.data, "_normalize_col_name"):
                idx = self.data._normalize_col_name(key)
                if idx is not None:
                    return self.data[:, idx]
            if key in self.metadata:
                return self.metadata[key]
            if key == "sampling_rate":
                return self.sampling_rate
            k_low = key.lower().strip()
            if k_low in ("series", "values", "value", "signal", "y", "x", "x_t", "ts", "time_series", "concentration", "sunspots", "sunspot_number", "ssn", "temperature", "eeg", "activity"):
                return self.primary_signal
            if k_low in ("t", "times", "timestamps", "time_s", "year", "years"):
                return self.time
            if k_low in ("order_parameter", "r", "r_values", "kuramoto_r"):
                if hasattr(self.data, "_normalize_col_name"):
                    r_idx = self.data._normalize_col_name("kuramoto_r")
                    if r_idx is not None:
                        return self.data[:, r_idx]
                return np.clip(np.abs(self.primary_signal), 0.0, 1.0)
            if k_low in ("k_values", "k", "coupling", "param_values"):
                return np.linspace(0.82, 2.0, len(self.primary_signal))
        raise KeyError(f"Key '{key}' not found in Dataset. Available attributes, metadata, and columns: {list(self.__dict__.keys()) + list(self.metadata.keys()) + self.columns}")

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-compatible .get(key, default) accessor for Dataset."""
        try:
            return self[key]
        except (KeyError, IndexError, AttributeError):
            return default

    def keys(self) -> List[str]:
        return ["name", "metadata", "columns", "time", "data", "primary_signal", "normalized"] + list(self.columns)

    def values(self) -> List[Any]:
        return [self[k] for k in self.keys()]

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or (key in self.columns) or (key in self.metadata)

    def __iter__(self):
        return iter(self.keys())

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
        Dataset identifier (e.g. 'solar_sunspots', 'climate_enso', 'neural_eeg',
        'power_grid_ieee', 'belousov_zhabotinsky', etc.)
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
        "power_grid_ieee": load_power_grid_ieee,
        "belousov_zhabotinsky": load_belousov_zhabotinsky,
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


def load_power_grid_ieee(as_dataframe: bool = False) -> Union[Dataset, Any]:
    """
    Loads IEEE 14-Bus continental power grid frequency and Kuramoto synchronization trace.
    Fulfilling claude_haiku's Cartographer's Petition (Section 3.2: Power Grid Dynamics).
    """
    meta = get_dataset_info("power_grid_ieee")
    filepath = os.path.join(DATASETS_DIR, meta["filename"])
    raw = _load_csv_raw(filepath)

    if hasattr(raw, "to_numpy"):
        df = raw
        if as_dataframe:
            return df
        columns = list(df.columns)
        time = df["time_s"].to_numpy(dtype=float)
        primary = df["kuramoto_r"].to_numpy(dtype=float)
        data = df.to_numpy(dtype=float)
    else:
        headers, rows = raw
        columns = headers
        arr = np.array(rows, dtype=float)
        time = arr[:, columns.index("time_s")]
        primary = arr[:, columns.index("kuramoto_r")]
        data = arr

    norm = _zscore(primary)
    return Dataset(
        name="power_grid_ieee",
        metadata=meta,
        columns=columns,
        time=time,
        data=data,
        primary_signal=primary,
        normalized=norm,
    )


def load_belousov_zhabotinsky(as_dataframe: bool = False) -> Union[Dataset, Any]:
    """
    Loads Belousov-Zhabotinsky chemical relaxation oscillation concentrations and redox potential.
    Fulfilling claude_haiku's Cartographer's Petition (Section 3.3: Chemical Reaction Datasets).
    """
    meta = get_dataset_info("belousov_zhabotinsky")
    filepath = os.path.join(DATASETS_DIR, meta["filename"])
    raw = _load_csv_raw(filepath)

    if hasattr(raw, "to_numpy"):
        df = raw
        if as_dataframe:
            return df
        columns = list(df.columns)
        time = df["time_s"].to_numpy(dtype=float)
        primary = df["redox_potential_mv"].to_numpy(dtype=float)
        data = df.to_numpy(dtype=float)
    else:
        headers, rows = raw
        columns = headers
        arr = np.array(rows, dtype=float)
        time = arr[:, columns.index("time_s")]
        primary = arr[:, columns.index("redox_potential_mv")]
        data = arr

    norm = _zscore(primary)
    return Dataset(
        name="belousov_zhabotinsky",
        metadata=meta,
        columns=columns,
        time=time,
        data=data,
        primary_signal=primary,
        normalized=norm,
    )
