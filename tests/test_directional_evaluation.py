import numpy as np
import pandas as pd
import pytest

from reciprocal_match.evaluation.directional import (
    DIRECTIONAL_METRIC_COLUMNS,
    directional_probability_metrics,
    validate_prediction_frame,
)


def test_directional_metrics_schema_and_probability_metrics():
    y = pd.Series([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    out = directional_probability_metrics(y, p)
    assert set(out) == {
        "test_positive_rate", "log_loss", "roc_auc", "pr_auc", "brier"
    }
    assert out["roc_auc"] == 1.0
    assert out["pr_auc"] == 1.0
    assert 0 <= out["brier"] <= 1


def test_directional_metrics_reject_invalid_probabilities():
    with pytest.raises(ValueError):
        directional_probability_metrics(pd.Series([0, 1]), np.array([0.2, 1.2]))


def test_prediction_frame_schema_and_range():
    df = pd.DataFrame({
        "iid": [1], "pid": [2], "wave": [1],
        "y_like": [1], "y_match": [0], "p_like": [0.7],
    })
    validate_prediction_frame(df)
