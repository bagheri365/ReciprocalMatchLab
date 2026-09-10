from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from reciprocal_match.data.taxonomy import load_yaml, primary_model_features


@dataclass(frozen=True)
class FoldData:
    train: pd.DataFrame
    test: pd.DataFrame
    test_wave: int


def load_feature_decisions(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def primary_waves_from_contract(contract: dict) -> list[int]:
    return list(contract["protocol"]["primary_waves"])


def leave_one_wave_out(
    df: pd.DataFrame,
    primary_waves: Iterable[int],
):
    waves = list(primary_waves)
    scoped = df[df["wave"].isin(waves)].copy()
    for wave in waves:
        train = scoped[scoped["wave"] != wave].copy()
        test = scoped[scoped["wave"] == wave].copy()
        if train.empty or test.empty:
            raise ValueError(f"Invalid LOOW fold for wave {wave}")
        yield FoldData(train=train, test=test, test_wave=int(wave))


def _stable_participant_table(
    df: pd.DataFrame,
    participant_features: list[str],
) -> pd.DataFrame:
    cols = ["iid", *participant_features]
    available = [c for c in cols if c in df.columns]
    records = []

    for iid, group in df[available].groupby("iid", sort=False):
        row = {"iid": iid}
        for feature in participant_features:
            if feature not in group.columns:
                row[feature] = np.nan
                continue
            values = group[feature].dropna().unique()
            if len(values) > 1:
                raise ValueError(
                    f"Time-1 feature {feature!r} is not stable within iid={iid}: {values[:5]}"
                )
            row[feature] = values[0] if len(values) == 1 else np.nan
        records.append(row)

    return pd.DataFrame(records)


def build_directed_feature_frame(
    df: pd.DataFrame,
    taxonomy: dict,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Build leakage-safe A/B features for directed preference prediction.

    Participant Time-1 features are reconstructed via iid/pid joins so candidate
    features do not rely on convenience mirror columns such as *_o.
    """
    allowed = primary_model_features(taxonomy)
    pair_features = taxonomy["categories"].get("derived_preinteraction_pair", [])
    participant_features = [f for f in allowed if f not in pair_features]

    participants = _stable_participant_table(df, participant_features)

    a = participants.rename(
        columns={c: f"A__{c}" for c in participant_features}
    ).rename(columns={"iid": "iid"})
    b = participants.rename(
        columns={c: f"B__{c}" for c in participant_features}
    ).rename(columns={"iid": "pid"})

    base_cols = ["iid", "pid", "wave", "dec", "match", *pair_features]
    out = df[base_cols].merge(a, on="iid", how="left", validate="many_to_one")
    out = out.merge(b, on="pid", how="left", validate="many_to_one")

    pair_cols = [f for f in pair_features if f in out.columns]
    feature_cols = [
        *(f"A__{c}" for c in participant_features),
        *(f"B__{c}" for c in participant_features),
        *pair_cols,
    ]

    X = out[feature_cols].copy()
    y_like = out["dec"].astype(int).copy()
    y_match = out["match"].astype(int).copy()
    meta = out[["iid", "pid", "wave"]].copy()
    return X, y_like, y_match, meta


def feature_types(
    X: pd.DataFrame,
    decisions: dict,
) -> tuple[list[str], list[str]]:
    categorical_base = set(decisions.get("categorical_features", []))
    categorical = [
        c for c in X.columns
        if c.split("__", 1)[-1] in categorical_base
    ]
    numeric = [c for c in X.columns if c not in categorical]
    return numeric, categorical


def make_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value=-999999),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("categorical", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )
