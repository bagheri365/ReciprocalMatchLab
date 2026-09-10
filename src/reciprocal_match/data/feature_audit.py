from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from reciprocal_match.data.taxonomy import (
    primary_model_features,
    validate_primary_vs_blocked,
    validate_taxonomy_columns,
)


@dataclass(frozen=True)
class FeatureAuditSummary:
    n_columns: int
    n_primary_features: int
    n_missing_taxonomy_columns: int
    n_duplicate_taxonomy_columns: int
    n_unexpected_taxonomy_columns: int
    n_primary_features_over_review_threshold: int


def primary_missingness_by_wave(
    df: pd.DataFrame,
    taxonomy: dict,
) -> pd.DataFrame:
    features = primary_model_features(taxonomy)
    available = [c for c in features if c in df.columns]
    if not available:
        return pd.DataFrame(columns=["wave", "feature", "missing_rate"])

    out = (
        df.groupby("wave")[available]
        .agg(lambda s: s.isna().mean())
        .stack()
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"level_1": "feature"})
    )
    return out.sort_values(["feature", "wave"]).reset_index(drop=True)


def feature_missingness_summary(
    df: pd.DataFrame,
    taxonomy: dict,
) -> pd.DataFrame:
    features = primary_model_features(taxonomy)
    rows = []
    for feature in features:
        if feature not in df.columns:
            continue
        per_wave = df.groupby("wave")[feature].apply(lambda s: s.isna().mean())
        rows.append(
            {
                "feature": feature,
                "overall_missing_rate": float(df[feature].isna().mean()),
                "max_wave_missing_rate": float(per_wave.max()),
                "median_wave_missing_rate": float(per_wave.median()),
                "waves_with_any_missing": int((per_wave > 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["max_wave_missing_rate", "overall_missing_rate", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_feature_audit_summary(
    df: pd.DataFrame,
    taxonomy: dict,
    review_threshold: float,
) -> FeatureAuditSummary:
    validation = validate_taxonomy_columns(df.columns, taxonomy)
    validate_primary_vs_blocked(taxonomy)
    missing = feature_missingness_summary(df, taxonomy)
    over = int((missing["max_wave_missing_rate"] > review_threshold).sum()) if len(missing) else 0

    return FeatureAuditSummary(
        n_columns=len(df.columns),
        n_primary_features=len(primary_model_features(taxonomy)),
        n_missing_taxonomy_columns=len(validation.missing_columns),
        n_duplicate_taxonomy_columns=len(validation.duplicate_columns),
        n_unexpected_taxonomy_columns=len(validation.unexpected_columns),
        n_primary_features_over_review_threshold=over,
    )


def render_feature_audit_markdown(
    summary: FeatureAuditSummary,
    missingness: pd.DataFrame,
    review_threshold: float,
) -> str:
    flagged = missingness[missingness["max_wave_missing_rate"] > review_threshold]

    lines = [
        "# Feature and Protocol Audit",
        "",
        "## Taxonomy gate",
        "",
        f"- Dataset columns: **{summary.n_columns}**",
        f"- Primary modeling features: **{summary.n_primary_features}**",
        f"- Unclassified dataset columns: **{summary.n_missing_taxonomy_columns}**",
        f"- Duplicate taxonomy assignments: **{summary.n_duplicate_taxonomy_columns}**",
        f"- Taxonomy columns absent from dataset: **{summary.n_unexpected_taxonomy_columns}**",
        "",
        "## Missingness review",
        "",
        f"Primary features with max per-wave missingness > {review_threshold:.0%}: "
        f"**{summary.n_primary_features_over_review_threshold}**",
        "",
        "| feature | overall_missing | max_wave_missing | median_wave_missing | waves_with_any_missing |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in flagged.itertuples(index=False):
        lines.append(
            f"| {row.feature} | {row.overall_missing_rate:.3f} | "
            f"{row.max_wave_missing_rate:.3f} | {row.median_wave_missing_rate:.3f} | "
            f"{row.waves_with_any_missing} |"
        )

    lines.extend(
        [
            "",
            "## Gate",
            "",
            "Modeling may begin only if taxonomy coverage is exact and every flagged",
            "primary feature has an explicit keep/drop/imputation decision in the experiment contract.",
            "",
        ]
    )
    return "\n".join(lines)
