from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from reciprocal_match.evaluation.match_probability import (
    fixed_width_reliability_table,
    paired_probability_differences,
    per_wave_probability_metrics,
    render_match_probability_summary,
    unique_pair_probabilities,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate held-out match probabilities on unique undirected pairs."
    )
    parser.add_argument(
        "--predictions",
        default="reports/tables/direct_joint_match_predictions.csv",
    )
    parser.add_argument(
        "--config",
        default="configs/match_probability_evaluation.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    config = yaml.safe_load(Path(args.config).read_text())

    pairs = unique_pair_probabilities(predictions)
    per_wave = per_wave_probability_metrics(pairs)
    reliability = fixed_width_reliability_table(
        pairs,
        n_bins=int(config["metrics"]["calibration_bins"]),
    )
    paired = paired_probability_differences(per_wave)

    outputs = config["outputs"]
    per_wave_path = Path(outputs["per_wave_csv"])
    reliability_path = Path(outputs["reliability_csv"])
    paired_path = Path(outputs["paired_csv"])
    summary_path = Path(outputs["summary_md"])
    for path in [per_wave_path, reliability_path, paired_path, summary_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    per_wave.to_csv(per_wave_path, index=False)
    reliability.to_csv(reliability_path, index=False)
    paired.to_csv(paired_path, index=False)
    summary_path.write_text(
        render_match_probability_summary(pairs, per_wave, reliability, paired),
        encoding="utf-8",
    )

    print(f"Wrote {per_wave_path}")
    print(f"Wrote {reliability_path}")
    print(f"Wrote {paired_path}")
    print(f"Wrote {summary_path}")
    print()
    print(per_wave.groupby("model")[["log_loss", "brier"]].mean().reset_index().to_string(index=False))
    print()
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
