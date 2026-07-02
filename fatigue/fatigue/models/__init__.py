"""Fatigue classification models: classical baselines and a compact 1D-CNN."""
from .classical import build_classical_model, CLASSICAL_MODELS

__all__ = ["build_classical_model", "CLASSICAL_MODELS"]
