"""Training loop for the 1D-CNN.

Implements the Step 8 checklist: seed reproducibility, class-balanced loss,
AdamW weight decay, dropout (in the model), ReduceLROnPlateau scheduling,
gradient clipping, optional mixed precision, and early stopping on validation
balanced accuracy.

Per-subject normalisation of the raw windows is fit on the training split only
and applied to validation, so no test-subject statistics leak.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .config import TrainConfig
from .evaluate import compute_metrics
from .utils import get_logger, set_seed

logger = get_logger()

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False


@dataclass
class TrainResult:
    best_val_metrics: Dict[str, float]
    best_epoch: int
    history: list


def _standardize(train: np.ndarray, val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel z-score using train statistics only. X is (n, ch, time)."""
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True) + 1e-8
    return (train - mean) / std, (val - mean) / std


def train_cnn(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    device: Optional[str] = None,
) -> TrainResult:
    if not _TORCH:
        raise ImportError("PyTorch is required to train the CNN.")
    set_seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X_train, X_val = _standardize(X_train, X_val)

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # Class-balanced cross-entropy (inverse-frequency weights).
    classes, counts = np.unique(y_train, return_counts=True)
    weights = len(y_train) / (len(classes) * counts)
    weight_tensor = torch.zeros(model.n_classes, device=device)
    for c, w in zip(classes, weights):
        weight_tensor[int(c)] = w
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=5)
    use_amp = cfg.mixed_precision and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_score = -np.inf
    best_metrics: Dict[str, float] = {}
    best_epoch = -1
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device, enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        preds, probs = _predict_loader(model, val_loader, device)
        pos_score = probs[:, -1] if probs.shape[1] == 2 else None
        val_metrics = compute_metrics(y_val, preds, pos_score)
        scheduler.step(val_metrics["balanced_accuracy"])
        history.append({"epoch": epoch, **val_metrics})

        score = val_metrics["balanced_accuracy"]
        if score > best_score:
            best_score, best_metrics, best_epoch = score, val_metrics, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stopping_patience:
                logger.info("Early stopping at epoch %d (best epoch %d).", epoch, best_epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(best_val_metrics=best_metrics, best_epoch=best_epoch, history=history)


def _predict_loader(model, loader, device):
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(probs.argmax(dim=-1).cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_probs)
