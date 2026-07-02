"""Feature extraction for windowed sEMG.

Each extractor takes a window of shape (window_samples, n_channels) and returns
a flat feature vector plus matching feature names, so downstream code can build
a labelled feature matrix.

Fatigue signature (the physiological basis for these features):
    As a muscle fatigues, motor-unit conduction velocity drops and firing
    synchronization changes. In the sEMG this shows up as a *spectral
    compression* -- median and mean frequency shift downward -- while
    amplitude features (RMS, MAV) tend to rise during sustained effort. The
    frequency-domain features below are therefore the highest-value signals,
    with time-domain amplitude/complexity features as complementary inputs.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..config import FeatureConfig, SignalConfig
from .time_domain import time_domain_features
from .freq_domain import freq_domain_features

__all__ = ["extract_features", "time_domain_features", "freq_domain_features"]


def extract_features(
    window: np.ndarray,
    fs: float,
    fcfg: FeatureConfig,
) -> Tuple[np.ndarray, List[str]]:
    """Extract the configured feature set from one multi-channel window.

    Returns (values, names). Channels are suffixed as ``chN`` in names.
    """
    window = np.asarray(window, dtype=np.float64)
    if window.ndim == 1:
        window = window[:, None]
    n_ch = window.shape[1]

    values: List[float] = []
    names: List[str] = []
    for ch in range(n_ch):
        sig = window[:, ch]
        if fcfg.time_domain:
            v, n = time_domain_features(sig, fcfg)
            values.extend(v)
            names.extend(f"ch{ch}_{name}" for name in n)
        if fcfg.freq_domain:
            v, n = freq_domain_features(sig, fs, fcfg)
            values.extend(v)
            names.extend(f"ch{ch}_{name}" for name in n)
    return np.asarray(values, dtype=np.float64), names
