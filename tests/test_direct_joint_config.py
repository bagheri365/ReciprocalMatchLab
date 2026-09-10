from pathlib import Path
import yaml


def test_joint_model_config_frozen():
    cfg = yaml.safe_load(Path("configs/model_direct_joint.yaml").read_text())
    assert cfg["evaluation"]["comparison_policies"] == [
        "requester", "candidate", "product", "minimum", "direct_joint"
    ]
    assert cfg["evaluation"]["primary_metric"] == "mutual_outcome_yield_at_k"
