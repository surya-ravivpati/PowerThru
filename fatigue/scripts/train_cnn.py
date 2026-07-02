"""Train + evaluate the 1D-CNN with Leave-One-Subject-Out CV.

    python scripts/train_cnn.py --synthetic
    python scripts/train_cnn.py --config config/default.yaml

For each LOSO fold the held-out subject is used as the validation/test set.
Per-subject normalisation is fit on the training subjects only.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import _bootstrap  # noqa: F401
from fatigue.config import Config
from fatigue.data import build_window_tensor, load_dataset, loso_splits, make_synthetic_dataset
from fatigue.evaluate import compute_metrics
from fatigue.models.cnn import build_cnn, count_parameters
from fatigue.train import train_cnn
from fatigue.utils import get_logger, set_seed

logger = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    set_seed(cfg.train.seed)

    recordings = make_synthetic_dataset(cfg) if args.synthetic else load_dataset(cfg)
    logger.info("Windowing raw signals...")
    X, y, groups, _ = build_window_tensor(recordings, cfg)
    logger.info("Tensor: X=%s, %d subjects", X.shape, len(np.unique(groups)))

    n_classes = len(np.unique(y))
    fold_scores = []
    for fold, (tr, te) in enumerate(loso_splits(groups)):
        model = build_cnn(in_channels=X.shape[1], n_classes=n_classes, dropout=cfg.train.dropout)
        if fold == 0:
            logger.info("CNN trainable parameters: %d", count_parameters(model))
        result = train_cnn(model, X[tr], y[tr], X[te], y[te], cfg.train)
        subj = int(np.unique(groups[te])[0])
        logger.info("Fold %d (subject %d): %s", fold, subj,
                    {k: round(v, 3) for k, v in result.best_val_metrics.items()})
        fold_scores.append(result.best_val_metrics)

    mean_bacc = float(np.mean([m["balanced_accuracy"] for m in fold_scores]))
    std_bacc = float(np.std([m["balanced_accuracy"] for m in fold_scores]))
    logger.info("=== CNN cross-subject balanced accuracy: %.3f +/- %.3f ===", mean_bacc, std_bacc)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.output_dir / "cnn_loso.json"
    with open(out, "w") as fh:
        json.dump({"per_fold": fold_scores, "mean_balanced_acc": mean_bacc,
                   "std_balanced_acc": std_bacc}, fh, indent=2)
    logger.info("Saved results -> %s", out)


if __name__ == "__main__":
    main()
