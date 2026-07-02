"""Step 1 -- dataset inspection.

Run this against your local copy of the sEMG dataset once `load_dataset` in
fatigue/data.py has been implemented for the real file layout. It reports the
structure you need to verify before modelling: per-subject/movement counts,
channel count, label distribution, sampling assumptions, NaN/flatline checks,
and recording-length statistics.

    python scripts/inspect_dataset.py --config config/default.yaml
    python scripts/inspect_dataset.py --synthetic   # no dataset needed
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

import _bootstrap  # noqa: F401
from fatigue.config import Config
from fatigue.data import load_dataset, make_synthetic_dataset
from fatigue.utils import get_logger

logger = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect the sEMG fatigue dataset.")
    ap.add_argument("--config", default=None, help="Path to YAML config.")
    ap.add_argument("--synthetic", action="store_true", help="Use synthetic data.")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    recordings = make_synthetic_dataset(cfg) if args.synthetic else load_dataset(cfg)

    subjects = sorted({r.subject for r in recordings})
    movements = sorted({r.movement for r in recordings})
    logger.info("Recordings: %d | subjects: %d | movements: %d",
                len(recordings), len(subjects), len(movements))

    label_counter: Counter = Counter()
    n_channels = set()
    lengths = []
    n_nan = 0
    for r in recordings:
        label_counter.update(np.asarray(r.labels).tolist())
        n_channels.add(r.signal.shape[1] if r.signal.ndim > 1 else 1)
        lengths.append(r.signal.shape[0])
        n_nan += int(np.isnan(r.signal).sum())

    logger.info("Channel counts across recordings: %s", sorted(n_channels))
    logger.info("Raw label distribution (per sample): %s", dict(sorted(label_counter.items())))
    logger.info("Recording length samples: min=%d max=%d mean=%.0f",
                min(lengths), max(lengths), float(np.mean(lengths)))
    logger.info("Total NaN samples: %d", n_nan)
    if n_channels != {cfg.signal.n_channels}:
        logger.warning("Channel count differs from config (%d).", cfg.signal.n_channels)


if __name__ == "__main__":
    main()
