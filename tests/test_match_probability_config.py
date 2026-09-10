from pathlib import Path
import yaml


def test_match_probability_evaluation_uses_unique_pairs():
    config = yaml.safe_load(Path("configs/match_probability_evaluation.yaml").read_text())
    assert config["unit"] == "unique_undirected_pair"
    assert config["metrics"]["primary"] == ["log_loss", "brier"]


def test_probability_comparison_is_only_product_vs_direct_joint():
    config = yaml.safe_load(Path("configs/match_probability_evaluation.yaml").read_text())
    assert set(config["models"]) == {"product", "direct_joint"}
    assert config["comparison"]["paired_unit"] == "wave"
