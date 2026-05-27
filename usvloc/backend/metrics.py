from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def precision_recall_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    total_positives: int | None = None,
) -> Dict[str, np.ndarray]:
    """Build the paper-style PR curve from thresholded query scores.

    BEVPlace++ reports loop-closure PR by thresholding the nearest descriptor
    distance for every query. Recall is TP / all actual positive queries, not
    TP / positive top-1 predictions. ``total_positives`` keeps that denominator
    explicit; the old ``sum(labels)`` fallback is kept for non-paper uses.
    """
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.size == 0:
        empty = np.asarray([], dtype=np.float32)
        return {"precision": empty, "recall": empty, "thresholds": empty}
    order = np.argsort(scores)[::-1]
    labels = labels[order]
    thresholds = scores[order]
    true_positive = np.cumsum(labels == 1)
    false_positive = np.cumsum(labels == 0)
    if total_positives is None:
        total_positive = int((labels == 1).sum())
    else:
        total_positive = int(total_positives)
    total_positive = max(total_positive, 1)
    precision = true_positive / np.maximum(true_positive + false_positive, 1)
    recall = true_positive / total_positive
    return {
        "precision": precision.astype(np.float32, copy=False),
        "recall": recall.astype(np.float32, copy=False),
        "thresholds": thresholds.astype(np.float32, copy=False),
    }


def area_under_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    """Standard average-precision area under the paper PR curve.

    The curve points are already ordered by descending threshold, so recall is
    non-decreasing. We use the step-wise AP integral, which is the conventional
    "average precision" used for retrieval and loop-closure PR reporting.
    """
    precision = np.asarray(precision, dtype=np.float32)
    recall = np.asarray(recall, dtype=np.float32)
    if precision.size == 0 or recall.size == 0:
        return 0.0
    order = np.argsort(recall, kind="stable")
    precision = precision[order]
    recall = recall[order]
    previous_recall = np.concatenate([np.asarray([0.0], dtype=np.float32), recall[:-1]])
    delta = np.maximum(recall - previous_recall, 0.0)
    return float(np.sum(delta * precision))


def best_f1_and_threshold(
    precision: np.ndarray,
    recall: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[float, float]:
    precision = np.asarray(precision, dtype=np.float32)
    recall = np.asarray(recall, dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if precision.size == 0 or recall.size == 0 or thresholds.size == 0:
        return 0.0, float("inf")
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1.0e-8)
    index = int(np.argmax(f1))
    return float(f1[index]), float(thresholds[index])


def max_recall_at_precision(
    precision: np.ndarray,
    recall: np.ndarray,
    target_precision: float = 1.0,
) -> float:
    precision = np.asarray(precision, dtype=np.float32)
    recall = np.asarray(recall, dtype=np.float32)
    if precision.size == 0 or recall.size == 0:
        return 0.0
    mask = precision >= float(target_precision)
    if not np.any(mask):
        return 0.0
    return float(np.max(recall[mask]))


def mean_or_zero(values: Sequence[float]) -> float:
    values = [float(value) for value in values]
    return float(np.mean(values)) if values else 0.0


def summarize_runtime(values_ms: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values_ms), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "fps": 0.0}
    mean_ms = float(arr.mean())
    return {
        "mean": mean_ms,
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "fps": float(1000.0 / mean_ms) if mean_ms > 0.0 else 0.0,
    }
