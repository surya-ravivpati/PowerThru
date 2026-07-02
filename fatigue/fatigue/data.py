"""Dataset loading, windowing, subject-aware splitting.

The public sEMG dataset is not vendored in this repo. The loader here defines
the *interface* the rest of the pipeline expects and documents the mapping you
must implement once the raw files are placed under ``config.data_root``. A
synthetic generator (`make_synthetic_dataset`) produces data with the same
shape so the whole pipeline is runnable and testable without the download.

Leakage policy: splits are by *subject* (LOSO / GroupKFold), never by random
window, and the feature scaler is fit on training subjects only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from .config import Config
from .features import extract_features
from .preprocessing import clean_signal, segment


@dataclass
class Recording:
    """One continuous multi-channel recording with time-synced labels.

    signal:  (samples, n_channels) raw sEMG.
    labels:  (samples,) integer fatigue level (0/1/2), time-synchronised.
    subject: participant id (used as the grouping key for splits).
    movement: movement/protocol id (used for per-movement evaluation).
    """

    signal: np.ndarray
    labels: np.ndarray
    subject: int
    movement: int


def load_dataset(cfg: Config) -> List[Recording]:  # pragma: no cover - needs real data
    """Load the real dataset from ``cfg.data_root``.

    IMPLEMENT ME against the actual file layout. The expected result is a list
    of `Recording`. Typical steps for the target Delsys/sEMG dataset:

      1. Walk ``data_root`` for per-participant folders.
      2. For each movement trial, read the raw sEMG (e.g. .csv/.mat) into a
         (samples, 8) array, ordered consistently by muscle/channel.
      3. Read the time-synced self-perceived fatigue level into a (samples,)
         integer array (0=none, 1=moderate, 2=high).
      4. yield Recording(signal, labels, subject=<pid>, movement=<mid>).

    Keep channel ordering identical across participants, and record any
    missing/corrupted trials in a manifest rather than silently dropping them.
    """
    raise NotImplementedError(
        "Implement load_dataset() for your local copy of the dataset under "
        f"{cfg.data_root!s}. See make_synthetic_dataset() for the expected shape."
    )


def make_synthetic_dataset(
    cfg: Config,
    n_subjects: int = 6,
    movements: int = 3,
    seconds: float = 20.0,
    seed: int = 0,
) -> List[Recording]:
    """Physiologically-flavoured synthetic data for pipeline testing ONLY.

    Fatigue is encoded the way it appears in real sEMG: the dominant spectral
    frequency drifts *down* and amplitude drifts *up* as the (synthetic)
    fatigue level rises. This is enough to exercise every stage of the pipeline
    end to end; it is NOT a substitute for the real dataset.
    """
    rng = np.random.default_rng(seed)
    fs = cfg.signal.sampling_rate
    n = int(seconds * fs)
    t = np.arange(n) / fs
    recs: List[Recording] = []
    for subj in range(n_subjects):
        subj_gain = rng.uniform(0.8, 1.2)  # between-subject variability
        for mv in range(movements):
            # Fatigue ramps 0 -> 2 across the trial.
            level = np.clip((t / seconds) * 3.0, 0, 2).astype(int)
            base_freq = 120.0 - 30.0 * (level / 2.0)  # spectral compression
            amp = subj_gain * (1.0 + 0.5 * (level / 2.0))  # amplitude rise
            sig = np.zeros((n, cfg.signal.n_channels))
            for ch in range(cfg.signal.n_channels):
                phase = rng.uniform(0, 2 * np.pi)
                carrier = amp * np.sin(2 * np.pi * base_freq * t + phase)
                noise = rng.normal(0, 0.1, n)
                sig[:, ch] = carrier + noise
            recs.append(Recording(sig, level, subject=subj, movement=mv))
    return recs


def _map_label(level: int, cfg: Config) -> int:
    if cfg.label.scheme == "binary":
        return int(level >= cfg.label.binary_threshold)
    return int(level)


def build_feature_matrix(
    recordings: List[Recording], cfg: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Clean -> window -> feature matrix.

    Returns X (n_windows, n_features), y, groups (subject id per window),
    movements (movement id per window), and feature names.
    """
    X: List[np.ndarray] = []
    y: List[int] = []
    groups: List[int] = []
    movements: List[int] = []
    names: Optional[List[str]] = None

    win = cfg.window_samples
    step = cfg.step_samples

    for rec in recordings:
        cleaned = clean_signal(rec.signal, cfg.signal)
        for w in segment(cleaned, win, step, cfg.window, reject_artifacts=True):
            feats, fnames = extract_features(w.data, cfg.signal.sampling_rate, cfg.features)
            if names is None:
                names = fnames
            # Majority label over the window (labels are time-synced).
            seg_labels = rec.labels[w.start : w.start + win]
            level = int(np.round(np.mean(seg_labels)))
            X.append(feats)
            y.append(_map_label(level, cfg))
            groups.append(rec.subject)
            movements.append(rec.movement)

    if not X:
        raise RuntimeError("No windows produced -- check window size vs recording length.")
    return (
        np.vstack(X),
        np.asarray(y, dtype=int),
        np.asarray(groups, dtype=int),
        np.asarray(movements, dtype=int),
        names or [],
    )


def build_window_tensor(
    recordings: List[Recording], cfg: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clean -> window into raw tensors for the CNN.

    Returns X (n_windows, n_channels, window_samples), y, groups, movements.
    Unlike `build_feature_matrix` this keeps the raw cleaned samples so the
    network can learn its own features (and be deployed without the hand-
    crafted feature code).
    """
    X: List[np.ndarray] = []
    y: List[int] = []
    groups: List[int] = []
    movements: List[int] = []
    win = cfg.window_samples
    step = cfg.step_samples

    for rec in recordings:
        cleaned = clean_signal(rec.signal, cfg.signal)
        for w in segment(cleaned, win, step, cfg.window, reject_artifacts=True):
            X.append(w.data.T.astype(np.float32))  # (channels, time)
            seg_labels = rec.labels[w.start : w.start + win]
            level = int(np.round(np.mean(seg_labels)))
            y.append(_map_label(level, cfg))
            groups.append(rec.subject)
            movements.append(rec.movement)

    if not X:
        raise RuntimeError("No windows produced -- check window size vs recording length.")
    return (
        np.stack(X),
        np.asarray(y, dtype=int),
        np.asarray(groups, dtype=int),
        np.asarray(movements, dtype=int),
    )


def loso_splits(groups: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Leave-One-Subject-Out folds.

    LOSO is the honest estimator of *cross-subject* generalisation: each fold's
    test subject is entirely unseen in training, which is exactly the
    deployment condition (a new athlete puts the device on). With only ~13
    subjects it is also the most data-efficient use of the cohort.
    """
    logo = LeaveOneGroupOut()
    dummy = np.zeros(len(groups))
    yield from logo.split(dummy, groups=groups)


def group_kfold_splits(groups: np.ndarray, n_splits: int = 5):
    """GroupKFold alternative for larger cohorts (faster than full LOSO)."""
    gkf = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(groups))
    return gkf.split(dummy, groups=groups)


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Standardise features using TRAIN statistics only."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler
