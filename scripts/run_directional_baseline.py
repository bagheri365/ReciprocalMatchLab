from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from reciprocal_match.data.audit import load_speed_dating_csv
from reciprocal_match.data.preprocessing import load_feature_decisions
from reciprocal_match.data.taxonomy import load_yaml
from reciprocal_match.evaluation.directional import render_directional_summary
from reciprocal_match.experiments.directional_baseline import run_directional_loow
from reciprocal_match.models.directional import DirectionalModelSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe LOOW directional logistic regression."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--taxonomy",
        default="configs/feature_taxonomy.yaml",
    )
    parser.add_argument(
        "--decisions",
        default="configs/feature_decisions.yaml",
    )
    parser.add_argument(
        "--contract",
        default="configs/experiment_contract.yaml",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model_directional_logistic.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_speed_dating_csv(args.input)
    taxonomy = load_yaml(args.taxonomy)
    decisions = load_feature_decisions(args.decisions)

    with Path(args.contract).open("r", encoding="utf-8") as f:
        contract = yaml.safe_load(f)
    with Path(args.model_config).open("r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    params = model_cfg["model"]["params"]
    spec = DirectionalModelSpec(
        C=float(params["C"]),
        solver=str(params["solver"]),
        max_iter=int(params["max_iter"]),
        random_state=int(params["random_state"]),
    )

    result = run_directional_loow(
        df=df,
        taxonomy=taxonomy,
        decisions=decisions,
        primary_waves=list(contract["protocol"]["primary_waves"]),
        spec=spec,
    )

    outputs = model_cfg["outputs"]
    metrics_path = Path(outputs["metrics_csv"])
    predictions_path = Path(outputs["predictions_csv"])
    summary_path = Path(outputs["summary_md"])

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    result.per_wave.to_csv(metrics_path, index=False)
    result.predictions.to_csv(predictions_path, index=False)
    summary_path.write_text(
        render_directional_summary(result.per_wave, result.predictions),
        encoding="utf-8",
    )

    print(f"Wrote {metrics_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {summary_path}")
    print(result.per_wave.to_string(index=False))


if __name__ == "__main__":
    main()
