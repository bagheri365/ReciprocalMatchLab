from __future__ import annotations

import pandas as pd

from reciprocal_match.data.preprocessing import build_directed_feature_frame


def build_unique_pair_frame(
    df: pd.DataFrame,
    taxonomy: dict,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build one canonical undirected row per observed pair.

    A/B participant columns are collapsed into symmetric numeric summaries so
    swapping the two people leaves the feature vector unchanged.
    """
    X_dir, _, y_match, meta = build_directed_feature_frame(df, taxonomy)
    full = pd.concat([meta, X_dir, y_match.rename("y_match")], axis=1)

    full["u"] = full[["iid", "pid"]].min(axis=1)
    full["v"] = full[["iid", "pid"]].max(axis=1)
    full = (
        full.sort_values(["wave", "u", "v", "iid", "pid"])
        .drop_duplicates(["wave", "u", "v"], keep="first")
        .reset_index(drop=True)
    )

    a_cols = [c for c in X_dir.columns if c.startswith("A__")]
    pair_cols = [
        c for c in X_dir.columns
        if not (c.startswith("A__") or c.startswith("B__"))
    ]

    features: dict[str, pd.Series] = {}
    for a_col in a_cols:
        base = a_col.split("__", 1)[1]
        b_col = f"B__{base}"
        if b_col not in full.columns:
            continue

        a = pd.to_numeric(full[a_col], errors="coerce")
        b = pd.to_numeric(full[b_col], errors="coerce")
        features[f"sym_absdiff__{base}"] = (a - b).abs()
        features[f"sym_mean__{base}"] = (a + b) / 2.0
        features[f"sym_min__{base}"] = pd.concat([a, b], axis=1).min(axis=1)
        features[f"sym_max__{base}"] = pd.concat([a, b], axis=1).max(axis=1)

    for col in pair_cols:
        features[f"pair__{col}"] = pd.to_numeric(full[col], errors="coerce")

    X_pair = pd.DataFrame(features, index=full.index)
    y_pair = full["y_match"].astype(int).copy()
    meta_pair = full[["wave", "u", "v"]].copy()
    return X_pair, y_pair, meta_pair
