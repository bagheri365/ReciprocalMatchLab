from pathlib import Path

import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from reciprocal_match.data.preprocessing import (
    build_directed_feature_frame,
    feature_types,
    leave_one_wave_out,
    load_feature_decisions,
    make_preprocessor,
)
from reciprocal_match.data.taxonomy import load_yaml


def test_primary_taxonomy_drops_entire_wave_missing_blocks():
    tax = load_yaml("configs/feature_taxonomy.yaml")
    primary = set(tax["categories"]["preinteraction_primary"])
    dropped = {
        "expnum",
        "attr4_1", "sinc4_1", "intel4_1", "fun4_1", "amb4_1", "shar4_1",
        "attr5_1", "sinc5_1", "intel5_1", "fun5_1", "amb5_1",
    }
    assert primary.isdisjoint(dropped)


def test_leave_one_wave_out_never_leaks_test_wave():
    df = pd.DataFrame({"wave": [1, 1, 2, 2, 3, 3], "x": range(6)})
    folds = list(leave_one_wave_out(df, [1, 2, 3]))
    assert len(folds) == 3
    for fold in folds:
        assert set(fold.test["wave"]) == {fold.test_wave}
        assert fold.test_wave not in set(fold.train["wave"])


def test_candidate_features_are_reconstructed_from_pid_join():
    df = pd.DataFrame(
        [
            {"iid": 1, "pid": 2, "wave": 1, "dec": 1, "match": 0, "age": 20, "int_corr": .2},
            {"iid": 2, "pid": 1, "wave": 1, "dec": 0, "match": 0, "age": 30, "int_corr": .2},
        ]
    )
    tax = {
        "categories": {
            "preinteraction_primary": ["age"],
            "derived_preinteraction_pair": ["int_corr"],
        },
        "modeling": {
            "allowed_primary_groups": [
                "preinteraction_primary",
                "derived_preinteraction_pair",
            ]
        },
    }
    X, y, ym, meta = build_directed_feature_frame(df, tax)
    assert X.loc[0, "A__age"] == 20
    assert X.loc[0, "B__age"] == 30
    assert X.loc[1, "A__age"] == 30
    assert X.loc[1, "B__age"] == 20


def test_preprocessor_fits_on_train_and_handles_unseen_category():
    X_train = pd.DataFrame({
        "A__age": [20.0, None, 30.0],
        "A__goal": [1, 1, 2],
    })
    X_test = pd.DataFrame({
        "A__age": [None],
        "A__goal": [99],
    })
    decisions = {"categorical_features": ["goal"]}
    num, cat = feature_types(X_train, decisions)
    pre = make_preprocessor(num, cat)
    Xt = pre.fit_transform(X_train)
    Xv = pre.transform(X_test)
    assert Xt.shape[0] == 3
    assert Xv.shape[0] == 1


def test_model_pipeline_can_fit_without_global_preprocessing():
    X = pd.DataFrame({
        "A__age": [20.0, None, 30.0, 40.0],
        "A__goal": [1, 1, 2, 2],
    })
    y = pd.Series([0, 0, 1, 1])
    decisions = {"categorical_features": ["goal"]}
    num, cat = feature_types(X, decisions)
    pipe = Pipeline([
        ("prep", make_preprocessor(num, cat)),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X, y)
    assert pipe.predict_proba(X).shape == (4, 2)
