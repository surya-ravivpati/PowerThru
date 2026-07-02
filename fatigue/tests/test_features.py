"""Feature correctness tests using signals with known properties."""
import numpy as np

from fatigue.config import FeatureConfig
from fatigue.features.freq_domain import (
    freq_domain_features,
    mean_frequency,
    median_frequency,
    peak_frequency,
    power_spectrum,
)
from fatigue.features.time_domain import mav, rms, zero_crossings


def _sine(freq, fs, seconds=2.0, amp=1.0):
    t = np.arange(int(fs * seconds)) / fs
    return amp * np.sin(2 * np.pi * freq * t)


def test_rms_and_mav_of_sine():
    # High sampling rate (many samples/period) so the discrete estimates
    # converge to the continuous closed forms.
    fs = 10000.0
    x = _sine(100, fs, amp=2.0)
    # RMS of a sine = amplitude / sqrt(2); MAV = 2*amp/pi.
    assert abs(rms(x) - 2.0 / np.sqrt(2)) < 1e-2
    assert abs(mav(x) - 2.0 * (2.0 / np.pi)) < 1e-2


def test_zero_crossings_track_frequency():
    fs = 1000.0
    x = _sine(50, fs, seconds=1.0)
    # A 50 Hz sine over 1 s crosses zero ~100 times.
    assert 95 <= zero_crossings(x, 0.0) <= 105


def test_spectral_features_of_pure_sine():
    fs = 1000.0
    x = _sine(120, fs, seconds=4.0)
    freqs, psd = power_spectrum(x, fs)
    assert abs(peak_frequency(freqs, psd) - 120) < 5
    assert abs(median_frequency(freqs, psd) - 120) < 8
    assert abs(mean_frequency(freqs, psd) - 120) < 12


def test_median_frequency_drops_with_fatigue_proxy():
    """Spectral compression: a lower-frequency sine => lower MDF (the fatigue signature)."""
    fs = 1259.0
    fresh = _sine(140, fs, seconds=3.0)
    fatigued = _sine(80, fs, seconds=3.0)
    f1, p1 = power_spectrum(fresh, fs)
    f2, p2 = power_spectrum(fatigued, fs)
    assert median_frequency(f2, p2) < median_frequency(f1, p1)


def test_feature_vector_shape_and_names_align():
    fs = 1259.0
    fcfg = FeatureConfig()
    x = _sine(100, fs, seconds=0.5)
    values, names = freq_domain_features(x, fs, fcfg)
    assert len(values) == len(names)
    assert "mdf" in names and "mnf" in names
