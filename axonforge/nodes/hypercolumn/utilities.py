import numpy as np

def to_display_grid(arr):
    """Convert 1D or 2D array to a square-ish 2D array for display.

    For 1D arrays: reshape into a near-square 2D array.
    For 2D arrays (weights matrix): create a mosaic of weight patches.
    """
    import numpy as np

    if arr.ndim == 1:
        n = len(arr)
        # Find dimensions for a near-square rectangle
        h = int(np.ceil(np.sqrt(n)))
        w = int(np.ceil(n / h))
        # Pad with zeros if needed
        padded = np.zeros(h * w, dtype=arr.dtype)
        padded[:n] = arr
        return padded.reshape(h, w)
    elif arr.ndim == 2:
        M, D = arr.shape
        # Check if D is a perfect square (for weight patches)
        sqrt_D = int(np.sqrt(D))
        if sqrt_D * sqrt_D == D:
            # This is a weight matrix with square patches
            H = sqrt_D
            W = sqrt_D
            S = int(np.ceil(np.sqrt(M)))
            total = S * S
            pad = total - M

            weights = arr
            if pad > 0:
                weights = np.vstack([weights, np.zeros((pad, D), dtype=weights.dtype)])

            mosaic = (
                weights.reshape(S, S, H, W).transpose(0, 2, 1, 3).reshape(S * H, S * W)
            )
            return mosaic
        else:
            # Not a square patch matrix, just reshape to near-square
            n = M * D
            h = int(np.ceil(np.sqrt(n)))
            w = int(np.ceil(n / h))
            flattened = arr.flatten()
            padded = np.zeros(h * w, dtype=arr.dtype)
            padded[:n] = flattened
            return padded.reshape(h, w)
    else:
        # Return as-is for other dimensions
        return arr

def scale_to_bwr(arr):
    """Scale array values to [-1, 1] range for full bwr color utilization.

    Uses min-max normalization: values are linearly mapped from [min, max]
    to [-1, 1]. This ensures unit vectors and other normalized data use
    the full blue-white-red color range instead of appearing whitish.
    """
    arr = np.asarray(arr)
    min_val = arr.min()
    max_val = arr.max()

    if max_val == min_val:
        # Constant array - return zeros to show as white/neutral
        return np.zeros_like(arr)

    # Scale to [-1, 1]: 2 * (x - min) / (max - min) - 1
    return 2 * (arr - min_val) / (max_val - min_val) - 1

def softmax(arr, axis=-1, temperature=1.0):
    """Numerically stable softmax with temperature scaling."""
    arr = np.asarray(arr)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    arr = arr / float(temperature)
    max_val = np.max(arr, axis=axis, keepdims=True)
    shifted = arr - max_val
    exp = np.exp(shifted)
    sum_exp = np.sum(exp, axis=axis, keepdims=True)
    return exp / sum_exp
