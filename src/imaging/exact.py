import numpy as np
import pandas as pd

def robust_scale(x: np.ndarray):
    med = np.median(x)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    iqr = iqr if iqr != 0 else 1.0
    return (x - med) / iqr

def make_gaf_gadf(x: np.ndarray, image_size: int):
    from pyts.image import GramianAngularField  # lazy import
    x_scaled = (x - x.min()) / (x.max() - x.min() + 1e-12)
    gasf = GramianAngularField(method='summation', image_size=image_size).fit_transform(x_scaled[None, :])[0]
    gadf = GramianAngularField(method='difference', image_size=image_size).fit_transform(x_scaled[None, :])[0]
    return gasf, gadf

def make_mtf(x: np.ndarray, image_size: int, n_bins: int = 8):
    from pyts.image import MarkovTransitionField  # lazy import
    x_scaled = (x - x.min()) / (x.max() - x.min() + 1e-12)
    mtf = MarkovTransitionField(image_size=image_size, n_bins=n_bins).fit_transform(x_scaled[None, :])[0]
    return mtf

def make_rp(x: np.ndarray, image_size: int):
    from pyts.image import RecurrencePlot  # lazy import
    x_scaled = (x - x.min()) / (x.max() - x.min() + 1e-12)
    rp = RecurrencePlot(dimension=1, epsilon='percentage_distance', percentage=20).fit_transform(x_scaled[None, :])[0]
    return rp[:image_size, :image_size]

def make_cwt(x: np.ndarray, image_size: int, wavelet: str = 'morl', scales: int = 64):
    import pywt  # lazy import
    x = (x - x.mean()) / (x.std() + 1e-12)
    widths = np.arange(1, scales + 1)
    cwtmatr, _ = pywt.cwt(x, widths, wavelet)
    img = (cwtmatr - cwtmatr.min()) / (cwtmatr.max() - cwtmatr.min() + 1e-12)
    img = img[:image_size, :image_size]
    if img.shape[0] < image_size or img.shape[1] < image_size:
        pad_h = image_size - img.shape[0]; pad_w = image_size - img.shape[1]
        img = np.pad(img, ((0, max(0, pad_h)), (0, max(0, pad_w))), mode='edge')
    return img
