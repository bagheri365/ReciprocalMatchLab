from pathlib import Path
import argparse
import pandas as pd
import yaml

from reciprocal_match.evaluation.marketplace import render_marketplace_summary
from reciprocal_match.experiments.marketplace_allocation import run_marketplace_allocation


def main():
    p=argparse.ArgumentParser(description="Run pair-level marketplace allocation trade-off experiment.")
    p.add_argument("--predictions",default="reports/tables/direct_joint_match_predictions.csv")
    p.add_argument("--config",default="configs/marketplace_allocation.yaml")
    a=p.parse_args()

    predictions=pd.read_csv(a.predictions)
    cfg=yaml.safe_load(Path(a.config).read_text())
    acfg=cfg["allocation"]; rcfg=cfg["random_baseline"]
    result=run_marketplace_allocation(
        predictions=predictions,
        capacity=int(acfg["per_user_capacity"]),
        target_capacity_fraction=float(acfg["target_capacity_fraction"]),
        lambda_grid=[float(x) for x in acfg["lambda_grid"]],
        random_repeats=int(rcfg["repeats"]),
        seed=int(rcfg["seed"]),
    )
    out=cfg["outputs"]
    paths={k:Path(v) for k,v in out.items()}
    for path in paths.values(): path.parent.mkdir(parents=True,exist_ok=True)
    result.per_wave.to_csv(paths["per_wave_csv"],index=False)
    result.tradeoff.to_csv(paths["tradeoff_csv"],index=False)
    result.paired.to_csv(paths["paired_csv"],index=False)
    result.user_exposure.to_csv(paths["exposure_csv"],index=False)
    paths["summary_md"].write_text(render_marketplace_summary(result.per_wave,result.tradeoff,result.paired))
    for path in paths.values(): print(f"Wrote {path}")
    print("\nOptimized trade-off:")
    print(result.tradeoff.to_string(index=False))
    print("\nPaired differences versus lambda=0:")
    print(result.paired.to_string(index=False))

if __name__=="__main__": main()
