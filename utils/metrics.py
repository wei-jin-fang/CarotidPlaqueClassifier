from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Sequence

matplotlib_config_dir = Path(os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib-carotid"))
try:
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
except OSError:
    matplotlib_config_dir = Path("/tmp/matplotlib-carotid")
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
if not os.access(matplotlib_config_dir, os.W_OK):
    matplotlib_config_dir = Path("/tmp/matplotlib-carotid")
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(matplotlib_config_dir)

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRIC_ORDER = [
    "accuracy",
    "auc",
    "recall",
    "f1",
    "precision",
    "specificity",
    "brier_score",
    "cohen_kappa",
    "pr_auc",
    "roc_auc",
    "balanced_accuracy",
    "npv",
]


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def compute_classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    probabilities: Sequence[float],
) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    sensitivity = recall_score(targets, predictions, zero_division=0)
    specificity = _safe_divide(tn, tn + fp)
    balanced_accuracy = (
        (sensitivity + specificity) / 2
        if not math.isnan(specificity)
        else float("nan")
    )
    cohen_kappa = (
        cohen_kappa_score(targets, predictions)
        if len(set(targets)) > 1 or len(set(predictions)) > 1
        else float("nan")
    )

    metrics = {
        "accuracy": accuracy_score(targets, predictions),
        "auc": float("nan"),
        "recall": sensitivity,
        "f1": f1_score(targets, predictions, zero_division=0),
        "precision": precision_score(targets, predictions, zero_division=0),
        "specificity": specificity,
        "brier_score": brier_score_loss(targets, probabilities),
        "cohen_kappa": cohen_kappa,
        "pr_auc": float("nan"),
        "roc_auc": float("nan"),
        "balanced_accuracy": balanced_accuracy,
        "npv": _safe_divide(tn, tn + fn),
    }

    unique_targets = sorted(set(targets))
    if len(unique_targets) == 2:
        metrics["roc_auc"] = roc_auc_score(targets, probabilities)
        metrics["auc"] = metrics["roc_auc"]
        metrics["pr_auc"] = average_precision_score(targets, probabilities)
    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    parts = []
    for key in METRIC_ORDER:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, float) and math.isnan(value):
            parts.append(f"{key}=nan")
        else:
            parts.append(f"{key}={value:.4f}")
    return ", ".join(parts)
