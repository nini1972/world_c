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
