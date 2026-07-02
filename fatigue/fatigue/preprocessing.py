"""Biomedical sEMG preprocessing and windowing.

Pipeline order (per channel):
    1. DC removal / band-pass (20-450 Hz, zero-phase Butterworth)
    2. Power-line notch (50/60 Hz)
    3. (optional) full-wave rectification -> linear envelope, for amplitude
       features and visualisation ONLY. Spectral features are computed on the
       band-passed (non-rectified) signal, because rectification distorts the
       power spectrum and would corrupt median/mean-frequency estimates.
    4. Windowing with overlap + artifact rejection.

Normalization is intentionally NOT applied here: it is fit on the training
split only (see `data.py`) so that test-subject statistics never leak into the
scaler.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from .config import SignalConfig, WindowConfig


def design_bandpass(cfg: SignalConfig):
    nyq = cfg.sampling_rate / 2.0
    low = cfg.bandpass_low / nyq
    high = min(cfg.bandpass_high / nyq, 0.999)
    return butter(cfg.bandpass_order, [low, high], btype="band")


def bandpass_filter(signal: np.ndarray, cfg: SignalConfig) -> np.ndarray:
    """Zero-phase band-pass. `signal` is (samples,) or (samples, channels)."""
    b, a = design_bandpass(cfg)
    # filtfilt over axis 0 (time); zero-phase avoids the group delay that would
    # otherwise misalign the signal with the fatigue labels.
    return filtfilt(b, a, signal, axis=0)


def notch_filter(signal: np.ndarray, cfg: SignalConfig) -> np.ndarray:
    if cfg.notch_freq is None:
        return signal
    w0 = cfg.notch_freq / (cfg.sampling_rate / 2.0)
    if not 0 < w0 < 1:
        return signal
    b, a = iirnotch(w0, cfg.notch_q)
    return filtfilt(b, a, signal, axis=0)


def clean_signal(signal: np.ndarray, cfg: SignalConfig) -> np.ndarray:
    """Full band-pass + notch clean of a raw multi-channel recording."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[:, None]
    filtered = bandpass_filter(signal, cfg)
    filtered = notch_filter(filtered, cfg)
    return filtered


def rectify(signal: np.ndarray) -> np.ndarray:
    """Full-wave rectification (amplitude features / envelope only)."""
    return np.abs(signal)


@dataclass
class Window:
    """A single segmented window of multi-channel sEMG.

    data:  (window_samples, n_channels) band-passed signal.
    start: index of the window start within the source recording.
    """

    data: np.ndarray
    start: int


def _robust_std(x: np.ndarray) -> float:
    """Median-absolute-deviation based std, robust to spikes."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad + 1e-12


def is_artifact(window: np.ndarray, wcfg: WindowConfig) -> bool:
    """Reject saturated (clipping/motion spikes) or flatlined windows."""
    per_ch_std = window.std(axis=0)
    if np.any(per_ch_std < wcfg.min_std):
        return True
    rstd = _robust_std(window.reshape(-1))
    if np.max(np.abs(window)) > wcfg.saturation_z * rstd:
        return True
    return False


def segment(
    signal: np.ndarray,
    window_samples: int,
    step_samples: int,
    wcfg: Optional[WindowConfig] = None,
    reject_artifacts: bool = True,
):
    """Slide a window over a cleaned recording.

    Yields `Window` objects. Windows shorter than `window_samples` at the tail
    are dropped. Artifact windows are skipped when `reject_artifacts` is True.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[:, None]
    n = signal.shape[0]
    for start in range(0, n - window_samples + 1, step_samples):
        win = signal[start : start + window_samples]
        if reject_artifacts and wcfg is not None and is_artifact(win, wcfg):
            continue
        yield Window(data=win, start=start)
