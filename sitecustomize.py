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
