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
except Exception:
    pass
