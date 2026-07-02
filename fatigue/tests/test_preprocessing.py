"""Preprocessing + pipeline-integration tests."""
import numpy as np

from fatigue.config import Config, SignalConfig
from fatigue.data import (
    build_feature_matrix,
    build_window_tensor,
    loso_splits,
    make_synthetic_dataset,
)
from fatigue.preprocessing import clean_signal, is_artifact, segment
from fatigue.evaluate import compute_metrics


def test_bandpass_removes_dc_offset():
    fs = 1259.0
    scfg = SignalConfig(sampling_rate=fs)
    t = np.arange(int(fs * 2)) / fs
    x = 5.0 + np.sin(2 * np.pi * 100 * t)  # large DC offset + 100 Hz
    cleaned = clean_signal(x, scfg)
    assert abs(cleaned.mean()) < 0.1  # DC removed


def test_notch_attenuates_powerline():
    fs = 1259.0
    scfg = SignalConfig(sampling_rate=fs, notch_freq=60.0)
    t = np.arange(int(fs * 2)) / fs
    interference = np.sin(2 * np.pi * 60 * t)
    signal_100 = np.sin(2 * np.pi * 100 * t)
    cleaned = clean_signal(interference + signal_100, scfg).ravel()
    # 60 Hz power should be substantially reduced relative to raw.
    def power_at(sig, f):
        freqs = np.fft.rfftfreq(len(sig), 1 / fs)
        mag = np.abs(np.fft.rfft(sig))
        return mag[np.argmin(np.abs(freqs - f))]
    raw_60 = power_at(interference + signal_100, 60)
    clean_60 = power_at(cleaned, 60)
    assert clean_60 < 0.5 * raw_60


def test_segment_counts_and_shapes():
    fs = 1259.0
    scfg = SignalConfig(sampling_rate=fs)
    x = np.random.randn(int(fs * 2), 8)
    win, step = 630, 315
    windows = list(segment(x, win, step, wcfg=None, reject_artifacts=False))
    assert all(w.data.shape == (win, 8) for w in windows)
    assert len(windows) == (x.shape[0] - win) // step + 1


def test_artifact_flags_flatline():
    from fatigue.config import WindowConfig
    flat = np.zeros((630, 8))
    assert is_artifact(flat, WindowConfig())


def test_synthetic_pipeline_end_to_end():
    cfg = Config()
    recs = make_synthetic_dataset(cfg, n_subjects=4, movements=2, seconds=8.0)
    X, y, groups, movements, names = build_feature_matrix(recs, cfg)
    assert X.shape[0] == len(y) == len(groups)
    assert X.shape[1] == len(names)
    assert set(np.unique(y)).issubset({0, 1})
    # Window tensor path.
    Xt, yt, gt, mt = build_window_tensor(recs, cfg)
    assert Xt.shape[1] == cfg.signal.n_channels
    assert Xt.shape[2] == cfg.window_samples


def test_loso_has_no_subject_leakage():
    cfg = Config()
    recs = make_synthetic_dataset(cfg, n_subjects=4, movements=2, seconds=8.0)
    _, y, groups, _, _ = build_feature_matrix(recs, cfg)
    n_folds = 0
    for tr, te in loso_splits(groups):
        train_subjects = set(groups[tr])
        test_subjects = set(groups[te])
        assert train_subjects.isdisjoint(test_subjects)
        assert len(test_subjects) == 1
        n_folds += 1
    assert n_folds == len(np.unique(groups))


def test_metrics_perfect_prediction():
    y = np.array([0, 1, 0, 1, 1])
    m = compute_metrics(y, y, y.astype(float))
    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["mcc"] == 1.0
