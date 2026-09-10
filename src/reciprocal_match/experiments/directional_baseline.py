from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yaml

from reciprocal_match.data.preprocessing import (
    build_directed_feature_frame,
    leave_one_wave_out,
)
from reciprocal_match.evaluation.directional import (
    DIRECTIONAL_METRIC_COLUMNS,
    directional_probability_metrics,
    validate_prediction_frame,
)
from reciprocal_match.models.directional import (
    DirectionalModelSpec,
    make_directional_logistic_pipeline,
    predict_positive_probability,
)


@dataclass(frozen=True)
class DirectionalExperimentResult:
    per_wave: pd.DataFrame
    predictions: pd.DataFrame


def run_directional_loow(
    df: pd.DataFrame,
    taxonomy: dict,
    decisions: dict,
    primary_waves: list[int],
    spec: DirectionalModelSpec | None = None,
) -> DirectionalExperimentResult:
    metric_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []

    scoped = df[df["wave"].isin(primary_waves)].copy()

    for fold in leave_one_wave_out(scoped, primary_waves):
        X_train, y_train, _, meta_train = build_directed_feature_frame(
            fold.train, taxonomy
        )
        X_test, y_test, y_match_test, meta_test = build_directed_feature_frame(
            fold.test, taxonomy
        )

        if fold.test_wave in set(meta_train["wave"].astype(int)):
            raise AssertionError("Held-out wave leaked into training metadata")
        if set(meta_test["wave"].astype(int)) != {fold.test_wave}:
            raise AssertionError("Test metadata contains unexpected waves")

        pipeline = make_directional_logistic_pipeline(X_train, decisions, spec)
        pipeline.fit(X_train, y_train)
        p = predict_positive_probability(pipeline, X_test)

        metrics = directional_probability_metrics(y_test, p)
        metric_rows.append(
            {
                "wave": fold.test_wave,
                "n_train": len(y_train),
                "n_test": len(y_test),
                **metrics,
            }
        )

        pred = meta_test.copy()
        pred["y_like"] = y_test.to_numpy()
        pred["y_match"] = y_match_test.to_numpy()
        pred["p_like"] = p
        pred_frames.append(pred)

    per_wave = pd.DataFrame(metric_rows)[DIRECTIONAL_METRIC_COLUMNS]
    predictions = pd.concat(pred_frames, ignore_index=True)
    predictions = predictions.sort_values(["wave", "iid", "pid"]).reset_index(drop=True)
    validate_prediction_frame(predictions)

    if set(per_wave["wave"].astype(int)) != set(primary_waves):
        raise AssertionError("Not all primary waves produced an evaluation row")
    if len(predictions) != len(scoped):
        raise AssertionError("Each scoped directed row must receive one held-out prediction")

    return DirectionalExperimentResult(
        per_wave=per_wave.sort_values("wave").reset_index(drop=True),
        predictions=predictions,
    )
