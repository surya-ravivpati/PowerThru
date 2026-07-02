"""PowerThru muscle-fatigue detection pipeline.

A subject-generalizing sEMG fatigue detector intended to run (after
quantization) on an ARM Cortex-M wearable and later serve as one branch of
PowerThru's multimodal cramp-prediction system.

Sub-modules:
    config          Typed configuration loaded from YAML.
    preprocessing   Biomedical sEMG cleaning + windowing.
    features        Time-, frequency-, and (optional) wavelet/nonlinear features.
    data            Dataset loading, windowing and subject-aware splits (LOSO).
    models          Classical and deep (1D-CNN) classifiers.
    train           Training loop (early stopping, scheduling, class balancing).
    evaluate        Full classification metric suite.
"""

__version__ = "0.1.0"
