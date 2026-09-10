from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from reciprocal_match.data.preprocessing import leave_one_wave_out
from reciprocal_match.data.symmetric_pairs import build_unique_pair_frame
from reciprocal_match.evaluation.direct_joint import (
    evaluate_direct_joint_vs_baselines,
    paired_policy_differences,
)
from reciprocal_match.models.joint_match import (
    JointMatchModelSpec,
    make_joint_match_pipeline,
    predict_match_probability,
)


@dataclass(frozen=True)
class DirectJointExperimentResult:
    predictions: pd.DataFrame
    per_wave_metrics: pd.DataFrame
    paired_differences: pd.DataFrame


def add_direct_joint_scores(
    raw_df: pd.DataFrame,
    reciprocal_predictions: pd.DataFrame,
    taxonomy: dict,
    primary_waves: list[int],
    spec: JointMatchModelSpec | None = None,
    k_values: list[int] | None = None,
) -> DirectJointExperimentResult:
    spec = spec or JointMatchModelSpec()
    k_values = k_values or [1, 3, 5]
    scoped = raw_df[raw_df["wave"].isin(primary_waves)].copy()

    pair_predictions = []
    for fold in leave_one_wave_out(scoped, primary_waves):
        X_train, y_train, _ = build_unique_pair_frame(fold.train, taxonomy)
        X_test, y_test, meta_test = build_unique_pair_frame(fold.test, taxonomy)

        model = make_joint_match_pipeline(spec)
        model.fit(X_train, y_train)
        p_match = predict_match_probability(model, X_test)

        out = meta_test.copy()
        out["p_match_direct"] = p_match
        out["y_match_pair"] = y_test.to_numpy()
        pair_predictions.append(out)

    pair_pred = pd.concat(pair_predictions, ignore_index=True)

    scored = reciprocal_predictions.copy()
    scored["u"] = scored[["iid", "pid"]].min(axis=1)
    scored["v"] = scored[["iid", "pid"]].max(axis=1)
    scored = scored.merge(
        pair_pred[["wave", "u", "v", "p_match_direct"]],
        on=["wave", "u", "v"],
        how="left",
        validate="many_to_one",
    )
    if scored["p_match_direct"].isna().any():
        raise AssertionError("Missing direct joint score for some directed rows")

    scored["score_direct_joint"] = scored["p_match_direct"]
    metrics = evaluate_direct_joint_vs_baselines(scored, k_values)
    paired = paired_policy_differences(metrics, "direct_joint")

    return DirectJointExperimentResult(
        predictions=scored,
        per_wave_metrics=metrics,
        paired_differences=paired,
    )
