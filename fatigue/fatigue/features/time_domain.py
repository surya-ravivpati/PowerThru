"""Time-domain sEMG features.

All functions operate on a single-channel 1-D window. They are deliberately
dependency-free (NumPy only) and cheap, so the same code can be ported to the
embedded target for on-device feature extraction.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..config import FeatureConfig


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def mav(x: np.ndarray) -> float:
    """Mean absolute value."""
    return float(np.mean(np.abs(x)))


def iemg(x: np.ndarray) -> float:
    """Integrated EMG (sum of absolute values)."""
    return float(np.sum(np.abs(x)))


def ssi(x: np.ndarray) -> float:
    """Simple square integral (energy)."""
    return float(np.sum(x**2))


def variance(x: np.ndarray) -> float:
    return float(np.var(x))


def waveform_length(x: np.ndarray) -> float:
    """Cumulative length of the waveform (sum of |diff|)."""
    return float(np.sum(np.abs(np.diff(x))))


def zero_crossings(x: np.ndarray, threshold: float) -> float:
    """Count sign changes exceeding a deadzone (noise-robust)."""
    x0, x1 = x[:-1], x[1:]
    crossing = (x0 * x1 < 0) & (np.abs(x0 - x1) >= threshold)
    return float(np.sum(crossing))


def slope_sign_changes(x: np.ndarray, threshold: float) -> float:
    """Count changes in slope sign (a frequency proxy)."""
    d1 = x[1:-1] - x[:-2]
    d2 = x[1:-1] - x[2:]
    change = (d1 * d2 > 0) & ((np.abs(d1) >= threshold) | (np.abs(d2) >= threshold))
    return float(np.sum(change))


def willison_amplitude(x: np.ndarray, threshold: float) -> float:
    """Count of sample-to-sample changes exceeding a threshold (WAMP)."""
    return float(np.sum(np.abs(np.diff(x)) >= threshold))


def hjorth(x: np.ndarray) -> Tuple[float, float, float]:
    """Hjorth activity, mobility, complexity."""
    d1 = np.diff(x)
    d2 = np.diff(d1)
    var_x = np.var(x) + 1e-12
    var_d1 = np.var(d1) + 1e-12
    var_d2 = np.var(d2) + 1e-12
    activity = var_x
    mobility = np.sqrt(var_d1 / var_x)
    complexity = np.sqrt(var_d2 / var_d1) / (mobility + 1e-12)
    return float(activity), float(mobility), float(complexity)


def time_domain_features(x: np.ndarray, fcfg: FeatureConfig) -> Tuple[List[float], List[str]]:
    x = np.asarray(x, dtype=np.float64)
    activity, mobility, complexity = hjorth(x)
    values = [
        rms(x),
        mav(x),
        iemg(x),
        ssi(x),
        variance(x),
        waveform_length(x),
        zero_crossings(x, fcfg.zc_threshold),
        slope_sign_changes(x, fcfg.zc_threshold),
        willison_amplitude(x, fcfg.wamp_threshold),
        activity,
        mobility,
        complexity,
    ]
    names = [
        "rms", "mav", "iemg", "ssi", "var", "wl", "zc", "ssc", "wamp",
        "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
    ]
    return values, names
