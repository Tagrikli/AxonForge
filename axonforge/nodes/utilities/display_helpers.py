"""Display-oriented helper utilities for node implementations."""

import numpy as np


def to_display_grid(arr):
    """Convert 1D or 2D array to a square-ish 2D array for display."""
    if arr.ndim == 1:
        n = len(arr)
        h = int(np.ceil(np.sqrt(n)))
        w = int(np.ceil(n / h))
        padded = np.zeros(h * w, dtype=arr.dtype)
        padded[:n] = arr
        return padded.reshape(h, w)

    if arr.ndim == 2:
        rows, cols = arr.shape
        sqrt_cols = int(np.sqrt(cols))
        if sqrt_cols * sqrt_cols == cols:
            patch_h = sqrt_cols
            patch_w = sqrt_cols
            side = int(np.ceil(np.sqrt(rows)))
            total = side * side
            pad = total - rows

            weights = arr
            if pad > 0:
                weights = np.vstack([weights, np.zeros((pad, cols), dtype=weights.dtype)])

            return (
                weights.reshape(side, side, patch_h, patch_w)
                .transpose(0, 2, 1, 3)
                .reshape(side * patch_h, side * patch_w)
            )

        n = rows * cols
        h = int(np.ceil(np.sqrt(n)))
        w = int(np.ceil(n / h))
        flattened = arr.flatten()
        padded = np.zeros(h * w, dtype=arr.dtype)
        padded[:n] = flattened
        return padded.reshape(h, w)

    return arr
