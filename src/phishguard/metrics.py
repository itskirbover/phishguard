"""From-scratch ML evaluation metrics, threshold tuning, and stratified K-fold CV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ConfusionMatrix:
    """Binary classification confusion matrix."""

    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> dict[str, int]:
        return {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn}


@dataclass(frozen=True)
class ClassificationMetrics:
    """Full binary classification evaluation summary."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: float
    roc_auc: float
    pr_auc: float
    threshold: float
    confusion_matrix: ConfusionMatrix

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "fpr": round(self.fpr, 6),
            "roc_auc": round(self.roc_auc, 6),
            "pr_auc": round(self.pr_auc, 6),
            "threshold": round(self.threshold, 6),
            "confusion_matrix": self.confusion_matrix.to_dict(),
        }


def compute_confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> ConfusionMatrix:
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    return ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)


def compute_roc_auc(
    y_true: Sequence[int] | np.ndarray,
    y_scores: Sequence[float] | np.ndarray,
) -> float:
    """Compute ROC-AUC using exact trapezoidal integration with tie handling."""
    yt = np.asarray(y_true, dtype=np.int64)
    ys = np.asarray(y_scores, dtype=np.float64)

    n_pos = int(np.sum(yt == 1))
    n_neg = int(np.sum(yt == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Sort scores descending and group tied scores into single ROC steps
    order = np.argsort(-ys, kind="mergesort")
    yt_sorted = yt[order]
    ys_sorted = ys[order]

    distinct_indices = np.where(np.diff(ys_sorted) != 0)[0]
    threshold_indices = np.r_[distinct_indices, yt_sorted.size - 1]

    tps = np.cumsum(yt_sorted)[threshold_indices]
    fps = (1 + threshold_indices) - tps

    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]

    # Trapezoidal integration
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) * 0.5))


def compute_pr_auc(
    y_true: Sequence[int] | np.ndarray,
    y_scores: Sequence[float] | np.ndarray,
) -> float:
    """Compute Average Precision (area under the Precision-Recall curve)."""
    yt = np.asarray(y_true, dtype=np.int64)
    ys = np.asarray(y_scores, dtype=np.float64)

    n_pos = int(np.sum(yt == 1))
    if n_pos == 0:
        return 0.0

    order = np.argsort(-ys, kind="mergesort")
    yt_sorted = yt[order]
    ys_sorted = ys[order]

    distinct_indices = np.where(np.diff(ys_sorted) != 0)[0]
    threshold_indices = np.r_[distinct_indices, yt_sorted.size - 1]

    tps = np.cumsum(yt_sorted)[threshold_indices]
    fps = (1 + threshold_indices) - tps

    precision = tps / (tps + fps)
    recall = tps / n_pos

    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def evaluate_predictions(
    y_true: Sequence[int] | np.ndarray,
    y_scores: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute comprehensive binary classification metrics at a given threshold."""
    yt = np.asarray(y_true, dtype=np.int64)
    ys = np.asarray(y_scores, dtype=np.float64)
    yp = (ys >= threshold).astype(np.int64)

    cm = compute_confusion_matrix(yt, yp)
    total = cm.tp + cm.fp + cm.tn + cm.fn
    accuracy = (cm.tp + cm.tn) / total if total > 0 else 0.0
    precision = cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) > 0 else 0.0
    recall = cm.tp / (cm.tp + cm.fn) if (cm.tp + cm.fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = cm.fp / (cm.fp + cm.tn) if (cm.fp + cm.tn) > 0 else 0.0

    roc_auc = compute_roc_auc(yt, ys)
    pr_auc = compute_pr_auc(yt, ys)

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fpr,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        threshold=threshold,
        confusion_matrix=cm,
    )


def find_optimal_threshold(
    y_true: Sequence[int] | np.ndarray,
    y_scores: Sequence[float] | np.ndarray,
    max_fpr: float | None = None,
) -> float:
    """Find decision threshold maximizing F1 (or Recall subject to FPR <= max_fpr)."""
    yt = np.asarray(y_true, dtype=np.int64)
    ys = np.asarray(y_scores, dtype=np.float64)
    candidates = np.unique(np.clip(ys, 0.05, 0.95))
    if candidates.size == 0:
        return 0.5

    best_threshold = 0.5
    best_objective = -1.0

    for thr in candidates:
        metrics = evaluate_predictions(yt, ys, threshold=float(thr))
        if max_fpr is not None:
            if metrics.fpr <= max_fpr and metrics.recall > best_objective:
                best_objective = metrics.recall
                best_threshold = float(thr)
        else:
            if metrics.f1 > best_objective:
                best_objective = metrics.f1
                best_threshold = float(thr)

    return best_threshold


def stratified_kfold_indices(
    y: Sequence[int] | np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate deterministic stratified (train_idx, val_idx) splits."""
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")

    yt = np.asarray(y, dtype=np.int64)
    rng = np.random.default_rng(seed)

    pos_idx = np.where(yt == 1)[0]
    neg_idx = np.where(yt == 0)[0]
    if len(pos_idx) < n_splits or len(neg_idx) < n_splits:
        raise ValueError(
            f"Each class needs at least {n_splits} samples for {n_splits}-fold stratified CV."
        )

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    pos_folds = np.array_split(pos_idx, n_splits)
    neg_folds = np.array_split(neg_idx, n_splits)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        val_idx = np.concatenate([pos_folds[fold], neg_folds[fold]])
        train_parts = [
            np.concatenate([pos_folds[k], neg_folds[k]])
            for k in range(n_splits)
            if k != fold
        ]
        train_idx = np.concatenate(train_parts)
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        splits.append((train_idx, val_idx))
    return splits
