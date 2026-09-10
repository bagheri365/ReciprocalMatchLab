from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from reciprocal_match.data.audit import load_speed_dating_csv
from reciprocal_match.data.feature_audit import (
    build_feature_audit_summary,
    feature_missingness_summary,
    primary_missingness_by_wave,
    render_feature_audit_markdown,
)
from reciprocal_match.data.protocol import load_protocol_config, validate_protocol_groups
from reciprocal_match.data.taxonomy import load_yaml, validate_taxonomy_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit protocol groups and feature safety.")
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--taxonomy",
        default="configs/feature_taxonomy.yaml",
    )
    parser.add_argument(
        "--protocols",
        default="configs/protocol_groups.yaml",
    )
    parser.add_argument(
        "--contract",
        default="configs/experiment_contract.yaml",
    )
    parser.add_argument(
        "--report",
        default="reports/feature_audit.md",
    )
    parser.add_argument(
        "--summary-table",
        default="reports/tables/primary_feature_missingness.csv",
    )
    parser.add_argument(
        "--wave-table",
        default="reports/tables/primary_feature_missingness_by_wave.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_speed_dating_csv(args.input)
    taxonomy = load_yaml(args.taxonomy)
    protocols = load_protocol_config(args.protocols)
    validate_protocol_groups(protocols)

    with Path(args.contract).open("r", encoding="utf-8") as f:
        contract = yaml.safe_load(f)
    threshold = float(
        contract["preprocessing"]["primary_max_missing_rate_review_threshold"]
    )

    validation = validate_taxonomy_columns(df.columns, taxonomy)
    if not validation.ok:
        raise SystemExit(
            "Feature taxonomy is not exact: "
            f"missing={list(validation.missing_columns)}, "
            f"duplicates={list(validation.duplicate_columns)}, "
            f"unexpected={list(validation.unexpected_columns)}"
        )

    summary = build_feature_audit_summary(df, taxonomy, threshold)
    missingness = feature_missingness_summary(df, taxonomy)
    by_wave = primary_missingness_by_wave(df, taxonomy)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_feature_audit_markdown(summary, missingness, threshold),
        encoding="utf-8",
    )

    summary_path = Path(args.summary_table)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    missingness.to_csv(summary_path, index=False)

    wave_path = Path(args.wave_table)
    wave_path.parent.mkdir(parents=True, exist_ok=True)
    by_wave.to_csv(wave_path, index=False)

    print(f"Wrote {report_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {wave_path}")
    print(summary)


if __name__ == "__main__":
    main()
