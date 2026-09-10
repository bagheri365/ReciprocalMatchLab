from pathlib import Path
import yaml


def test_feature_decisions_are_frozen_in_contract():
    with Path("configs/experiment_contract.yaml").open() as f:
        contract = yaml.safe_load(f)
    assert contract["preprocessing"]["feature_decisions_file"] == "configs/feature_decisions.yaml"
    assert contract["research_policy"]["freeze_before_modeling"] is True


def test_int_corr_kept_with_fold_imputation():
    with Path("configs/feature_decisions.yaml").open() as f:
        decisions = yaml.safe_load(f)
    spec = decisions["keep_with_imputation"]["int_corr"]
    assert spec["strategy"] == "training-fold median"
    assert spec["add_missing_indicator"] is True
