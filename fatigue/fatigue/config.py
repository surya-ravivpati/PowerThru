"""Typed configuration for the fatigue pipeline.

Every tunable value lives here (or in the YAML that populates it) so that
experiments are reproducible and the reasoning behind each default is
documented in one place rather than scattered as magic numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a light optional dep
    yaml = None


@dataclass
class SignalConfig:
    """Acquisition + filtering parameters.

    Defaults target the target dataset (Delsys Trigno, 1259 Hz, 8 channels).
    """

    sampling_rate: float = 1259.0        # Hz, from the dataset spec.
    n_channels: int = 8                  # 8 muscles.

    # Band-pass: sEMG power is concentrated ~20-450 Hz. The 20 Hz high-pass
    # edge removes DC offset and low-frequency motion/movement artifact; the
    # 450 Hz low-pass edge sits comfortably below Nyquist (629.5 Hz) and above
    # the bulk of myoelectric energy.
    bandpass_low: float = 20.0
    bandpass_high: float = 450.0
    bandpass_order: int = 4              # 4th-order Butterworth, zero-phase.

    # Power-line interference. Delsys hardware is US-sourced -> 60 Hz default;
    # set to 50.0 for EU recordings. Q controls notch width.
    notch_freq: Optional[float] = 60.0
    notch_q: float = 30.0


@dataclass
class WindowConfig:
    """Segmentation parameters.

    A 0.5 s window at 1259 Hz gives ~630 samples: enough for a stable Welch
    PSD (so median/mean-frequency estimates are low-variance) while staying
    short enough to track fatigue during dynamic movements. 50% overlap
    doubles the number of training segments without heavy redundancy.
    """

    window_sec: float = 0.5
    overlap: float = 0.5                 # fraction; step = window * (1 - overlap)

    # Artifact rejection: drop windows whose peak amplitude exceeds
    # `saturation_z` robust std devs (clipping/motion) or that are flatlined.
    saturation_z: float = 8.0
    min_std: float = 1e-6                # below this a window is considered dead.


@dataclass
class FeatureConfig:
    time_domain: bool = True
    freq_domain: bool = True
    wavelet: bool = False                # requires PyWavelets; off by default.
    nonlinear: bool = False              # sample entropy etc. (slower).
    # Frequency bands (Hz) for band-power features.
    bands: tuple = ((20, 50), (50, 100), (100, 200), (200, 450))
    wamp_threshold: float = 1e-3         # Willison amplitude threshold (post-norm).
    zc_threshold: float = 1e-3           # zero-crossing / SSC deadzone.


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    early_stopping_patience: int = 12
    grad_clip: float = 1.0
    mixed_precision: bool = True
    num_workers: int = 2


@dataclass
class LabelConfig:
    # Dataset labels: 0=none, 1=moderate, 2=high. `scheme` selects the task.
    scheme: str = "binary"               # {"binary", "three_class"}
    # For binary: labels >= binary_threshold become positive (fatigued).
    binary_threshold: int = 1


@dataclass
class Config:
    data_root: Path = Path("data/raw")
    cache_dir: Path = Path("data/processed")
    output_dir: Path = Path("outputs")
    signal: SignalConfig = field(default_factory=SignalConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    label: LabelConfig = field(default_factory=LabelConfig)

    @property
    def window_samples(self) -> int:
        return int(round(self.window.window_sec * self.signal.sampling_rate))

    @property
    def step_samples(self) -> int:
        return max(1, int(round(self.window_samples * (1.0 - self.window.overlap))))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML configs.")
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        cfg = cls()
        for key, value in raw.items():
            if not hasattr(cfg, key):
                raise KeyError(f"Unknown config key: {key}")
            current = getattr(cfg, key)
            if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
                setattr(cfg, key, type(current)(**value))
            elif key in {"data_root", "cache_dir", "output_dir"}:
                setattr(cfg, key, Path(value))
            else:
                setattr(cfg, key, value)
        return cfg

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("data_root", "cache_dir", "output_dir"):
            d[k] = str(d[k])
        return d
