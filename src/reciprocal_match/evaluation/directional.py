from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


DIRECTIONAL_METRIC_COLUMNS = [
    "wave",
    "n_train",
    "n_test",
    "test_positive_rate",
    "log_loss",
    "roc_auc",
    "pr_auc",
    "brier",
]


def _safe_roc_auc(y_true: pd.Series, p: np.ndarray) -> float:
    if pd.Series(y_true).nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, p))


def directional_probability_metrics(
    y_true: pd.Series,
    p: np.ndarray,
) -> dict[str, float]:
    y = pd.Series(y_true).astype(int)
    p = np.asarray(p, dtype=float)

    if len(y) != len(p):
        raise ValueError("y_true and p must have equal length")
    if len(y) == 0:
        raise ValueError("Cannot evaluate an empty fold")
    if not np.isfinite(p).all():
        raise ValueError("Predicted probabilities must be finite")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("Predicted probabilities must lie in [0, 1]")

    return {
        "test_positive_rate": float(y.mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": _safe_roc_auc(y, p),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def validate_prediction_frame(predictions: pd.DataFrame) -> None:
    required = {
        "iid",
        "pid",
        "wave",
        "y_like",
        "y_match",
        "p_like",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction frame missing columns: {missing}")

    p = predictions["p_like"]
    if p.isna().any() or (~np.isfinite(p)).any():
        raise ValueError("p_like contains missing/non-finite values")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("p_like must lie in [0, 1]")


def render_directional_summary(
    per_wave: pd.DataFrame,
    predictions: pd.DataFrame,
) -> str:
    validate_prediction_frame(predictions)
    pooled = directional_probability_metrics(
        predictions["y_like"],
        predictions["p_like"].to_numpy(),
    )

    mean_log_loss = float(per_wave["log_loss"].mean())
    mean_roc_auc = float(per_wave["roc_auc"].mean())
    mean_pr_auc = float(per_wave["pr_auc"].mean())
    mean_brier = float(per_wave["brier"].mean())

    lines = [
        "# Directional Logistic Regression Baseline",
        "",
        "## Scope",
        "",
        "Leakage-safe leave-one-wave-out evaluation on the frozen primary protocol group.",
        "Preprocessing is fit separately inside every training fold.",
        "",
        "## Aggregate metrics",
        "",
        "| aggregation | log_loss | roc_auc | pr_auc | brier |",
        "|---|---:|---:|---:|---:|",
        f"| mean of wave metrics | {mean_log_loss:.4f} | {mean_roc_auc:.4f} | {mean_pr_auc:.4f} | {mean_brier:.4f} |",
        f"| pooled held-out predictions | {pooled['log_loss']:.4f} | {pooled['roc_auc']:.4f} | {pooled['pr_auc']:.4f} | {pooled['brier']:.4f} |",
        "",
        "## Per-wave results",
        "",
        "| wave | n_test | positive_rate | log_loss | roc_auc | pr_auc | brier |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in per_wave.sort_values("wave").itertuples(index=False):
        roc = "NA" if math.isnan(row.roc_auc) else f"{row.roc_auc:.4f}"
        lines.append(
            f"| {int(row.wave)} | {int(row.n_test)} | {row.test_positive_rate:.4f} | "
            f"{row.log_loss:.4f} | {roc} | {row.pr_auc:.4f} | {row.brier:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "This milestone establishes a directional probability baseline only.",
            "Do not interpret these results as reciprocal-match performance.",
            "Held-out probabilities are saved for later calibration and reciprocal-scoring milestones.",
            "",
        ]
    )
    return "\n".join(lines)
