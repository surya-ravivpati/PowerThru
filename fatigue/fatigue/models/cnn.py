"""Compact 1D-CNN for on-device sEMG fatigue detection.

Design goals (Step 7 -- production architecture):
    * Operate directly on a cleaned window (n_channels, window_samples) so no
      hand-crafted feature code is needed at inference on the wearable.
    * Small enough to quantise to int8 and fit ARM Cortex-M flash/RAM: depthwise
      -> pointwise separable convs keep the parameter count in the low tens of
      thousands.
    * Expose a fixed-width embedding (the pooled feature vector) so this model
      can later become one branch of the multimodal cramp predictor (Step 12).

`forward` returns logits. `predict` returns a structured output
(probability / stage / confidence / embedding) matching the integration spec.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False


if _TORCH:

    class SeparableConv1d(nn.Module):
        """Depthwise-separable 1D conv (mobilenet-style) for a small footprint."""

        def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1):
            super().__init__()
            pad = kernel // 2
            self.depthwise = nn.Conv1d(in_ch, in_ch, kernel, stride, pad, groups=in_ch)
            self.pointwise = nn.Conv1d(in_ch, out_ch, 1)
            self.bn = nn.BatchNorm1d(out_ch)

        def forward(self, x):
            return F.relu(self.bn(self.pointwise(self.depthwise(x))))

    class FatigueCNN(nn.Module):
        def __init__(
            self,
            in_channels: int = 8,
            n_classes: int = 2,
            width: int = 24,
            dropout: float = 0.3,
        ):
            super().__init__()
            self.n_classes = n_classes
            self.stem = nn.Sequential(
                nn.Conv1d(in_channels, width, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(width),
                nn.ReLU(),
            )
            self.block1 = SeparableConv1d(width, width * 2, kernel=5, stride=2)
            self.block2 = SeparableConv1d(width * 2, width * 2, kernel=3, stride=2)
            self.pool = nn.AdaptiveAvgPool1d(1)  # -> fixed embedding regardless of len
            self.dropout = nn.Dropout(dropout)
            self.embedding_dim = width * 2
            self.head = nn.Linear(self.embedding_dim, n_classes)

        def embed(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.stem(x)
            x = self.block1(x)
            x = self.block2(x)
            return self.pool(x).squeeze(-1)  # (batch, embedding_dim)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            emb = self.embed(x)
            return self.head(self.dropout(emb))

        @torch.no_grad()
        def predict(self, x: "torch.Tensor") -> "FatigueOutput":
            """Structured inference output for downstream fusion."""
            self.eval()
            emb = self.embed(x)
            logits = self.head(emb)
            probs = F.softmax(logits, dim=-1)
            stage = int(torch.argmax(probs, dim=-1)[0].item())
            confidence = float(probs.max(dim=-1).values[0].item())
            fatigue_prob = float(probs[0, -1].item())  # highest-fatigue class prob
            return FatigueOutput(
                fatigue_probability=fatigue_prob,
                fatigue_stage=stage,
                confidence=confidence,
                embedding=emb[0].cpu().numpy(),
            )


@dataclass
class FatigueOutput:
    """Contract consumed by the future multimodal cramp model (Step 12)."""

    fatigue_probability: float
    fatigue_stage: int
    confidence: float
    embedding: np.ndarray


def build_cnn(in_channels: int = 8, n_classes: int = 2, dropout: float = 0.3):
    if not _TORCH:
        raise ImportError("PyTorch is required for the CNN model.")
    return FatigueCNN(in_channels=in_channels, n_classes=n_classes, dropout=dropout)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
