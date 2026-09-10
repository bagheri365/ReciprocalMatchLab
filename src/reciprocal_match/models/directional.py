from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from reciprocal_match.data.preprocessing import (
    build_directed_feature_frame,
    feature_types,
    make_preprocessor,
)


@dataclass(frozen=True)
class DirectionalModelSpec:
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 2000
    random_state: int = 365


def make_directional_logistic_pipeline(
    X_train: pd.DataFrame,
    decisions: dict,
    spec: DirectionalModelSpec | None = None,
) -> Pipeline:
    spec = spec or DirectionalModelSpec()
    numeric, categorical = feature_types(X_train, decisions)
    preprocessor = make_preprocessor(numeric, categorical)

    model = LogisticRegression(
        C=spec.C,
        solver=spec.solver,
        max_iter=spec.max_iter,
        random_state=spec.random_state,
    )

    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def predict_positive_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(model.named_steps["model"].classes_)
    if 1 not in classes:
        raise ValueError(f"Positive class 1 absent from fitted model classes={classes}")
    return proba[:, classes.index(1)]
