from pathlib import Path
import yaml

def test_marketplace_contract_is_pair_level_and_frozen():
    c=yaml.safe_load(Path("configs/marketplace_allocation.yaml").read_text())
    assert c["allocation"]["unit"]=="undirected_pair"
    assert c["allocation"]["value_column"]=="p_direct_joint"
    assert c["allocation"]["lambda_grid"]==[0.0,0.01,0.05,0.10,0.20,0.50]
    assert c["exposure"]["opportunity_baseline"]=="uniform_random_eligible_edge"
