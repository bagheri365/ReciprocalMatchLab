from pathlib import Path
import argparse
import pandas as pd
import yaml

from reciprocal_match.data.audit import load_speed_dating_csv
from reciprocal_match.data.taxonomy import load_yaml
from reciprocal_match.evaluation.direct_joint import render_joint_summary
from reciprocal_match.experiments.direct_joint_match import add_direct_joint_scores
from reciprocal_match.models.joint_match import JointMatchModelSpec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--reciprocal-predictions",
        default="reports/tables/reciprocal_scores_predictions.csv",
    )
    parser.add_argument("--taxonomy", default="configs/feature_taxonomy.yaml")
    parser.add_argument("--contract", default="configs/experiment_contract.yaml")
    parser.add_argument("--config", default="configs/model_direct_joint.yaml")
    args = parser.parse_args()

    df = load_speed_dating_csv(args.input)
    reciprocal = pd.read_csv(args.reciprocal_predictions)
    taxonomy = load_yaml(args.taxonomy)
    contract = yaml.safe_load(Path(args.contract).read_text())
    cfg = yaml.safe_load(Path(args.config).read_text())

    params = cfg["model"]["params"]
    spec = JointMatchModelSpec(
        C=float(params["C"]),
        solver=str(params["solver"]),
        max_iter=int(params["max_iter"]),
        random_state=int(params["random_state"]),
    )

    result = add_direct_joint_scores(
        df,
        reciprocal,
        taxonomy,
        list(contract["protocol"]["primary_waves"]),
        spec=spec,
        k_values=list(cfg["evaluation"]["k_values"]),
    )

    outputs = cfg["outputs"]
    pred_path = Path(outputs["predictions_csv"])
    metric_path = Path(outputs["per_wave_csv"])
    paired_path = Path(outputs["paired_differences_csv"])
    summary_path = Path(outputs["summary_md"])
    for path in [pred_path, metric_path, paired_path, summary_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    result.predictions.to_csv(pred_path, index=False)
    result.per_wave_metrics.to_csv(metric_path, index=False)
    result.paired_differences.to_csv(paired_path, index=False)
    summary_path.write_text(
        render_joint_summary(result.per_wave_metrics, result.paired_differences)
    )

    print(f"Wrote {pred_path}")
    print(f"Wrote {metric_path}")
    print(f"Wrote {paired_path}")
    print(f"Wrote {summary_path}")
    print(
        result.per_wave_metrics.groupby(["policy", "k"])[
            ["mutual_outcome_yield_at_k", "ndcg_match_at_k"]
        ].mean().reset_index().to_string(index=False)
    )
    print()
    print(result.paired_differences.to_string(index=False))


if __name__ == "__main__":
    main()
