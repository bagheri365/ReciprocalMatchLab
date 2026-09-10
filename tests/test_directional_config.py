from pathlib import Path
import yaml


def test_directional_baseline_has_no_hyperparameter_search():
    with Path("configs/model_directional_logistic.yaml").open() as f:
        cfg = yaml.safe_load(f)
    assert cfg["model"]["params"]["C"] == 1.0
    assert "search" not in cfg["model"]


def test_contract_primary_metric_is_log_loss():
    with Path("configs/experiment_contract.yaml").open() as f:
        contract = yaml.safe_load(f)
    assert contract["evaluation"]["primary_directional_metric"] == "log_loss"
