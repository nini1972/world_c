import numpy as np
from skimage.measure import block_reduce, label


def test_block_reduce_mean():
    arr = np.arange(16, dtype=float).reshape(4, 4)
    out = block_reduce(arr, block_size=(2, 2), func=np.mean)
    assert out.shape == (2, 2)
    assert np.allclose(out, [[2.5, 4.5], [10.5, 12.5]])


def test_label_connected_components():
    arr = np.array([
        [1, 1, 0, 0],
        [1, 0, 0, 2],
        [0, 0, 2, 2],
    ])
    labeled, num = label(arr, return_num=True, connectivity=1)
    assert num == 2
    assert labeled[0, 0] == labeled[1, 0]
    assert labeled[1, 3] == labeled[2, 2]
    assert labeled[0, 0] != labeled[1, 3]


def test_scipy_ndimage_block_reduce_alias():
    import scipy.ndimage
    import sitecustomize  # noqa: F401

    assert hasattr(scipy.ndimage, "block_reduce")
    arr = np.arange(16, dtype=float).reshape(4, 4)
    out = scipy.ndimage.block_reduce(arr, block_size=(2, 2), func=np.mean)
    assert out.shape == (2, 2)
    assert np.allclose(out, [[2.5, 4.5], [10.5, 12.5]])


def test_scipy_ndimage_measurements_block_reduce_alias():
    import warnings
    import sitecustomize  # noqa: F401

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from scipy.ndimage.measurements import block_reduce

    arr = np.arange(16, dtype=float).reshape(4, 4)
    out = block_reduce(arr, block_size=(2, 2), func=np.mean)
    assert out.shape == (2, 2)
    assert np.allclose(out, [[2.5, 4.5], [10.5, 12.5]])


def test_matplotlib_boxplot_labels_compat_shim():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sitecustomize  # noqa: F401

    fig, ax = plt.subplots()
    try:
        res1 = ax.boxplot([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], labels=["A", "B"])
        assert "boxes" in res1
        res2 = plt.boxplot([1.0, 2.0, 3.0], positions=[1], labels=["Surrogates"])
        assert "boxes" in res2
    finally:
        plt.close("all")
