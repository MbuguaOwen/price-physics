import numpy as np
import pandas as pd
import warnings


def _resize2d_linear(a, new_h, new_w):
    """Lightweight 2D resize using bilinear via separable 1D interpolation.
    Avoids extra deps (cv2/skimage). Input -> float32."""
    a = np.asarray(a, dtype=np.float32)
    h, w = a.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((new_h, new_w), dtype=np.float32)
    ys = np.linspace(0, max(1, h - 1), int(new_h))
    xs = np.linspace(0, max(1, w - 1), int(new_w))
    x_idx = np.arange(w, dtype=np.float32)
    tmp = np.vstack([np.interp(xs, x_idx, a[int(min(max(y, 0), h - 1))]) for y in ys])
    return tmp.astype(np.float32)


def _effective_image_size(image_size, n_timestamps):
    """Return an image size that respects pyts constraints.
    Accepts int, None, 'auto'. Clamps to [1, n_timestamps]."""
    if image_size in (None, "auto", "AUTO"):
        return max(1, int(n_timestamps))
    try:
        s = int(image_size)
    except Exception:
        s = int(n_timestamps)
    s = max(1, s)
    s = min(s, int(n_timestamps))
    return s

def robust_scale(x: np.ndarray):
    med = np.median(x)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    iqr = iqr if iqr != 0 else 1.0
    return (x - med) / iqr

def make_gaf_gadf(x: np.ndarray, image_size: int):
    from pyts.image import GramianAngularField  # lazy import
    x_scaled = (x - x.min()) / (x.max() - x.min() + 1e-12)
    s = _effective_image_size(image_size, len(x))
    gasf = GramianAngularField(method='summation', image_size=s).fit_transform(x_scaled[None, :])[0]
    gadf = GramianAngularField(method='difference', image_size=s).fit_transform(x_scaled[None, :])[0]
    return gasf, gadf

def _discretize_series(x, n_bins=8):
    """Robust discretization without pyts.
    - Try quantile edges; if non-increasing, fallback to uniform.
    Returns integer labels in [0, bins-1] and effective bins used.
    """
    arr = np.asarray(x, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.zeros(1, dtype=np.int32), 2
    uniq = np.unique(arr)
    bins_req = int(max(2, n_bins or 8))
    if uniq.size < 2:
        return np.zeros_like(arr, dtype=np.int32), 2
    # Try quantile edges
    q = np.linspace(0, 1, bins_req + 1)
    try:
        edges = np.quantile(arr, q, method="linear")
    except TypeError:
        edges = np.percentile(arr, q * 100.0)
    # Ensure strictly increasing; otherwise uniform edges
    if not np.all(np.diff(edges) > 0):
        edges = np.linspace(arr.min(), arr.max(), min(bins_req, uniq.size) + 1)
    z = np.digitize(arr, edges[1:-1], right=False).astype(np.int32)
    eff_bins = int(edges.size - 1)
    eff_bins = max(2, min(eff_bins, bins_req))
    return z, eff_bins


def _mtf_from_discrete(z, eff_bins, image_size):
    """Build Markov Transition Field from discrete series.
    - Estimate row-normalized transition matrix P (KxK)
    - M[i,j] = P[z[i], z[j]]; resized to (image_size, image_size)
    """
    z = np.asarray(z, dtype=np.int32)
    K = int(max(2, eff_bins))
    counts = np.zeros((K, K), dtype=np.float32)
    if z.size >= 2:
        a = z[:-1]; b = z[1:]
        np.add.at(counts, (a, b), 1)
    row_sum = counts.sum(axis=1, keepdims=True)
    P = counts / np.maximum(row_sum, 1.0)
    if z.size == 0:
        M = np.zeros((1, 1), dtype=np.float32)
    else:
        M = P[z[:, None], z[None, :]]
    s = int(image_size)
    if M.shape[0] != s:
        M = _resize2d_linear(M, s, s)
    return M.astype(np.float32)


def make_mtf(x: np.ndarray, image_size: int, n_bins: int = 8):
    s = _effective_image_size(image_size, len(x))
    z, eff_bins = _discretize_series(x, n_bins=n_bins)
    M = _mtf_from_discrete(z, eff_bins, s)
    return M.astype(np.float32)

def make_rp(x: np.ndarray, image_size: int):
    from pyts.image import RecurrencePlot  # lazy import
    x_scaled = (x - x.min()) / (x.max() - x.min() + 1e-12)
    s = _effective_image_size(image_size, len(x))
    last_err = None
    for kwargs in (
        dict(dimension=1, epsilon='percentage_distance', percentage=20),
        dict(dimension=1, percentage=20),
        dict(dimension=1),
        dict(),
    ):
        try:
            rp_obj = RecurrencePlot(**kwargs)
            rp_mat = rp_obj.fit_transform(x_scaled[None, :])[0]
            return rp_mat[:s, :s]
        except TypeError as e:
            last_err = e
            continue
    # Fallback: simple distance-threshold RP at 20th percentile
    D = np.abs(x_scaled[:, None] - x_scaled[None, :])
    thr = np.percentile(D, 20.0)
    rp_mat = (D <= thr).astype(float)
    return rp_mat[:s, :s]

def make_cwt(x: np.ndarray, image_size: int, wavelet: str = 'morl', scales: int | None = None):
    import pywt  # lazy import
    # Normalize and choose an efficient set of scales
    x = np.asarray(x, dtype=float)
    x = (x - x.mean()) / (x.std() + 1e-12)
    s = _effective_image_size(image_size, len(x))
    # Limit the number of scales for speed; prefer geometric spacing
    num_scales = int(min(32, max(4, int(scales) if scales is not None else s)))
    max_width = max(2, s)
    widths = np.geomspace(1, max_width, num=num_scales)
    cwtmatr, _ = pywt.cwt(x, widths, wavelet)
    img = (cwtmatr - cwtmatr.min()) / (cwtmatr.max() - cwtmatr.min() + 1e-12)
    # Resize to (s, s)
    img = _resize2d_linear(img, s, s)
    return img.astype(np.float32)
