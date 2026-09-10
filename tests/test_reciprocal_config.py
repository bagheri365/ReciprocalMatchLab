from pathlib import Path
import yaml

def test_calibration_scheme_frozen():
    c=yaml.safe_load(Path("configs/reciprocal_scoring.yaml").read_text()); assert c["calibration"]["scheme"]=="outer_training_scores_only"

def test_scores_frozen():
    c=yaml.safe_load(Path("configs/reciprocal_scoring.yaml").read_text()); assert c["ranking"]["scores"]==["requester","candidate","product","minimum"]
