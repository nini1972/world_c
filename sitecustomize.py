"""
World C Site Customization:
Automatically patches json.JSONEncoder.default for NumPy scalar/array types
(`np.bool_`, `np.integer`, `np.floating`, `np.ndarray`) so agent scripts running
in World C never crash with `TypeError: Object of type bool is not JSON serializable`.
"""

import json

try:
    import numpy as np

    _orig_default = json.JSONEncoder.default

    def _safe_numpy_default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, (complex, np.complexfloating)):
            if abs(obj.imag) <= 1e-12:
                return float(obj.real)
            return {"real": float(obj.real), "imag": float(obj.imag)}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return _orig_default(self, obj)

    json.JSONEncoder.default = _safe_numpy_default
    if not hasattr(np, 'trapz') and hasattr(np, 'trapezoid'):
        np.trapz = np.trapezoid
except Exception:
    pass

# Guard scipy.spatial.cKDTree against OOM explosions on large/degenerate periodic orbits
try:
    import scipy.spatial

    _OrigCKDTree = scipy.spatial.cKDTree

    class _SafeCKDTree(_OrigCKDTree):
        def sparse_distance_matrix(self, other, max_distance, p=2.0, output_type="dok_matrix"):
            n_self = int(getattr(self, "n", 0))
            n_other = int(getattr(other, "n", 0))
            if n_self > 4000 or n_other > 4000:
                step_s = max(1, n_self // 2000)
                step_o = max(1, n_other // 2000)
                sub_self = _OrigCKDTree(self.data[::step_s])
                sub_other = sub_self if (other is self) else _OrigCKDTree(other.data[::step_o])
                out_fmt = "coo_matrix" if output_type == "dok_matrix" else output_type
                return _OrigCKDTree.sparse_distance_matrix(sub_self, sub_other, max_distance, p=p, output_type=out_fmt)
            return _OrigCKDTree.sparse_distance_matrix(self, other, max_distance, p=p, output_type=output_type)

    scipy.spatial.cKDTree = _SafeCKDTree
    scipy.spatial.KDTree = _SafeCKDTree
except Exception:
    pass

# Provide safe builtin log fallback if not present
try:
    import builtins
    if not hasattr(builtins, "log"):
        def _safe_builtin_log(*args, **kwargs):
            print(*args, **kwargs)
        builtins.log = _safe_builtin_log
except Exception:
    pass

# Alias scipy.ndimage.block_reduce and deprecated scipy.ndimage.measurements.block_reduce
# -> skimage.measure.block_reduce when absent
try:
    import warnings
    import scipy.ndimage
    from skimage.measure import block_reduce as _skimage_block_reduce

    if not hasattr(scipy.ndimage, "block_reduce"):
        scipy.ndimage.block_reduce = _skimage_block_reduce

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import scipy.ndimage.measurements as _scipy_ndimage_measurements
        if "block_reduce" not in vars(_scipy_ndimage_measurements):
            _scipy_ndimage_measurements.block_reduce = _skimage_block_reduce
    except Exception:
        pass
except Exception:
    pass

# Compatibility shim for Matplotlib >= 3.9 where boxplot(..., labels=...) was renamed to tick_labels=...
try:
    import inspect
    import functools
    import matplotlib.axes
    import matplotlib.pyplot as _plt

    def _wrap_boxplot_labels(fn):
        if getattr(fn, "_world_c_labels_shimmed", False):
            return fn
        try:
            params = inspect.signature(fn).parameters
        except Exception:
            params = {}
        accepts_tick_labels = "tick_labels" in params
        accepts_labels = "labels" in params

        @functools.wraps(fn)
        def _boxplot_compat(*args, **kwargs):
            if "labels" in kwargs and "tick_labels" not in kwargs and accepts_tick_labels:
                kwargs = dict(kwargs)
                kwargs["tick_labels"] = kwargs.pop("labels")
            elif "tick_labels" in kwargs and "labels" not in kwargs and accepts_labels and not accepts_tick_labels:
                kwargs = dict(kwargs)
                kwargs["labels"] = kwargs.pop("tick_labels")
            return fn(*args, **kwargs)

        _boxplot_compat._world_c_labels_shimmed = True
        return _boxplot_compat

    matplotlib.axes.Axes.boxplot = _wrap_boxplot_labels(matplotlib.axes.Axes.boxplot)
    _plt.boxplot = _wrap_boxplot_labels(_plt.boxplot)
except Exception:
    pass
