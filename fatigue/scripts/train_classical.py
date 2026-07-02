"""Train + evaluate a classical baseline with Leave-One-Subject-Out CV.

    python scripts/train_classical.py --synthetic --model random_forest
    python scripts/train_classical.py --config config/default.yaml --model xgboost

The scaler is fit on training subjects only; the held-out subject is fully
unseen. Reports per-fold and aggregate cross-subject metrics.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import _bootstrap  # noqa: F401
from fatigue.config import Config
from fatigue.data import (
    build_feature_matrix,
    fit_scaler,
    load_dataset,
    loso_splits,
    make_synthetic_dataset,
)
from fatigue.evaluate import compute_metrics, per_group_metrics
from fatigue.models import build_classical_model
from fatigue.utils import get_logger, set_seed

logger = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--model", default="random_forest")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    set_seed(cfg.train.seed)

    recordings = make_synthetic_dataset(cfg) if args.synthetic else load_dataset(cfg)
    logger.info("Building feature matrix...")
    X, y, groups, movements, names = build_feature_matrix(recordings, cfg)
    logger.info("Feature matrix: X=%s, %d features, %d subjects",
                X.shape, len(names), len(np.unique(groups)))

    n_classes = len(np.unique(y))
    all_true, all_pred, all_group, all_move = [], [], [], []
    fold_metrics = []

    for fold, (tr, te) in enumerate(loso_splits(groups)):
        scaler = fit_scaler(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        model = build_classical_model(args.model, cfg.train.seed, n_classes)
        model.fit(Xtr, y[tr])
        pred = model.predict(Xte)
        score = None
        if n_classes == 2 and hasattr(model, "predict_proba"):
            score = model.predict_proba(Xte)[:, 1]
        m = compute_metrics(y[te], pred, score)
        fold_metrics.append(m)
        test_subj = int(np.unique(groups[te])[0])
        logger.info("Fold %d (subject %d): balanced_acc=%.3f f1=%.3f mcc=%.3f",
                    fold, test_subj, m["balanced_accuracy"], m["f1"], m["mcc"])
        all_true.extend(y[te]); all_pred.extend(pred)
        all_group.extend(groups[te]); all_move.extend(movements[te])

    all_true = np.array(all_true); all_pred = np.array(all_pred)
    agg = compute_metrics(all_true, all_pred)
    logger.info("=== Aggregate cross-subject metrics ===")
    for k, v in agg.items():
        logger.info("  %-18s %.4f", k, v)

    mean_bacc = float(np.mean([m["balanced_accuracy"] for m in fold_metrics]))
    std_bacc = float(np.std([m["balanced_accuracy"] for m in fold_metrics]))
    logger.info("Per-subject balanced accuracy: %.3f +/- %.3f", mean_bacc, std_bacc)

    per_move = per_group_metrics(all_true, all_pred, np.array(all_move))
    logger.info("Per-movement balanced accuracy: %s",
                {k: round(v["balanced_accuracy"], 3) for k, v in per_move.items()})

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.output_dir / f"classical_{args.model}_loso.json"
    with open(out, "w") as fh:
        json.dump({"aggregate": agg, "per_fold": fold_metrics,
                   "mean_balanced_acc": mean_bacc, "std_balanced_acc": std_bacc}, fh, indent=2)
    logger.info("Saved results -> %s", out)


if __name__ == "__main__":
    main()
