"""Frequency-domain sEMG features.

These are the primary fatigue indicators: as a muscle fatigues the power
spectrum compresses toward lower frequencies, so median frequency (MDF) and
mean frequency (MNF) decrease over time. Computed from a Welch PSD of the
band-passed (non-rectified) signal.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.signal import welch

from ..config import FeatureConfig


def power_spectrum(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD. nperseg adapts to short windows to stay well-defined."""
    x = np.asarray(x, dtype=np.float64)
    nperseg = min(len(x), 256)
    freqs, psd = welch(x, fs=fs, nperseg=nperseg)
    return freqs, psd


def mean_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    total = np.sum(psd) + 1e-12
    return float(np.sum(freqs * psd) / total)


def median_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    cumulative = np.cumsum(psd)
    total = cumulative[-1] + 1e-12
    idx = int(np.searchsorted(cumulative, 0.5 * total))
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def peak_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    return float(freqs[int(np.argmax(psd))])


def spectral_entropy(psd: np.ndarray) -> float:
    """Shannon entropy of the normalised PSD, in [0, 1]."""
    p = psd / (np.sum(psd) + 1e-12)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    ent = -np.sum(p * np.log(p))
    return float(ent / np.log(len(p) + 1e-12)) if len(p) > 1 else 0.0


def band_powers(freqs: np.ndarray, psd: np.ndarray, bands) -> List[float]:
    out = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        out.append(float(np.sum(psd[mask])))
    return out


def freq_domain_features(x: np.ndarray, fs: float, fcfg: FeatureConfig) -> Tuple[List[float], List[str]]:
    freqs, psd = power_spectrum(x, fs)
    values = [
        mean_frequency(freqs, psd),
        median_frequency(freqs, psd),
        peak_frequency(freqs, psd),
        spectral_entropy(psd),
        float(np.sum(psd)),  # total power
    ]
    names = ["mnf", "mdf", "peak_freq", "spectral_entropy", "total_power"]
    bp = band_powers(freqs, psd, fcfg.bands)
    for (lo, hi), val in zip(fcfg.bands, bp):
        values.append(val)
        names.append(f"bandpower_{int(lo)}_{int(hi)}")
    return values, names
