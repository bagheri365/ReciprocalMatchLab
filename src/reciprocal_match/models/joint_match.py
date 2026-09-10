from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class JointMatchModelSpec:
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 2000
    random_state: int = 365


def make_joint_match_pipeline(
    spec: JointMatchModelSpec | None = None,
) -> Pipeline:
    spec = spec or JointMatchModelSpec()
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=spec.C,
                    solver=spec.solver,
                    max_iter=spec.max_iter,
                    random_state=spec.random_state,
                ),
            ),
        ]
    )


def predict_match_probability(model: Pipeline, X: pd.DataFrame):
    proba = model.predict_proba(X)
    classes = list(model.named_steps["model"].classes_)
    if 1 not in classes:
        raise ValueError("Positive match class absent")
    return proba[:, classes.index(1)]
