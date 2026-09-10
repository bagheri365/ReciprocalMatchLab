from pathlib import Path

import yaml

from reciprocal_match.data.taxonomy import (
    primary_model_features,
    validate_primary_vs_blocked,
    validate_taxonomy_columns,
)


def _load_taxonomy():
    with Path("configs/feature_taxonomy.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_taxonomy_assignments_are_disjoint():
    config = _load_taxonomy()
    all_cols = [c for cols in config["categories"].values() for c in cols]
    validation = validate_taxonomy_columns(all_cols, config)
    assert validation.missing_columns == ()
    assert validation.unexpected_columns == ()
    assert validation.duplicate_columns == ()


def test_primary_features_do_not_overlap_blocked_groups():
    config = _load_taxonomy()
    validate_primary_vs_blocked(config)


def test_primary_features_exclude_known_post_interaction_columns():
    config = _load_taxonomy()
    primary = set(primary_model_features(config))
    forbidden = {"attr_o", "like_o", "attr", "like", "attr1_s", "satis_2", "you_call"}
    assert primary.isdisjoint(forbidden)
