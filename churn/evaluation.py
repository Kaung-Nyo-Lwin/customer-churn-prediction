"""Evaluate churn separately from value segmentation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_threshold(y_true, probabilities) -> float:
    """Maximize F2 on out-of-fold training scores; break ties toward fewer alerts."""
    grid = np.linspace(0.10, 0.80, 71)
    scores = [fbeta_score(y_true, probabilities >= t, beta=2, zero_division=0) for t in grid]
    best = max(scores)
    return float(max(t for t, score in zip(grid, scores, strict=True) if score == best))


def evaluate(y_true, probabilities, charges, threshold: float) -> dict:
    """Historical charges of missed churners are a proxy, never realized revenue loss."""
    actual = np.asarray(y_true, dtype=bool)
    probability = np.asarray(probabilities, dtype=float)
    predicted = probability >= threshold
    historical_charges = np.nan_to_num(np.asarray(charges, dtype=float), nan=0.0)
    missed = actual & ~predicted
    total = float(historical_charges[actual].sum())
    missed_charges = float(historical_charges[missed].sum())
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(actual, predicted)),
        "roc_auc": float(roc_auc_score(actual, probability)),
        "average_precision": float(average_precision_score(actual, probability)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "f2": float(fbeta_score(actual, predicted, beta=2, zero_division=0)),
        "brier_score": float(brier_score_loss(actual, probability)),
        "flagged_customers": int(predicted.sum()),
        "false_negatives": int(missed.sum()),
        "missed_churn_historical_charges": missed_charges,
        "missed_charge_share": missed_charges / total if total else 0.0,
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[False, True]).tolist(),
    }
