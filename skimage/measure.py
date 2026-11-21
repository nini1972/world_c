import numpy as np


def block_reduce(image, block_size=2, func=np.sum, cval=0, func_kwargs=None):
    """Lightweight pure-NumPy drop-in for skimage.measure.block_reduce."""
    arr = np.asarray(image)
    if np.isscalar(block_size):
        block_size = (int(block_size),) * arr.ndim
    else:
        block_size = tuple(int(b) for b in block_size)
    if len(block_size) != arr.ndim:
        raise ValueError(f"block_size length {len(block_size)} must match image ndim {arr.ndim}")

    pad_width = []
    for dim, b in zip(arr.shape, block_size):
        rem = dim % b
        pad_width.append((0, (b - rem) % b))
    if any(p[1] > 0 for p in pad_width):
        arr = np.pad(arr, pad_width, mode="constant", constant_values=cval)

    new_shape = []
    for dim, b in zip(arr.shape, block_size):
        new_shape.extend([dim // b, b])
    reshaped = arr.reshape(new_shape)
    axes = tuple(range(1, 2 * arr.ndim, 2))
    kwargs = func_kwargs or {}
    return func(reshaped, axis=axes, **kwargs)
