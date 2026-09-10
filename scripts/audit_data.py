from __future__ import annotations

import argparse
from pathlib import Path

from reciprocal_match.data.audit import (
    build_audit_summary,
    load_speed_dating_csv,
    render_markdown,
    wave_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the raw Speed Dating dataset.")
    parser.add_argument("--input", required=True, help="Path to Speed Dating Data.csv")
    parser.add_argument(
        "--report",
        default="reports/data_audit.md",
        help="Markdown report output path",
    )
    parser.add_argument(
        "--wave-table",
        default="reports/tables/wave_summary.csv",
        help="Wave summary CSV output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_speed_dating_csv(args.input)
    summary = build_audit_summary(df)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(summary, df), encoding="utf-8")

    wave_path = Path(args.wave_table)
    wave_path.parent.mkdir(parents=True, exist_ok=True)
    wave_summary(df).to_csv(wave_path, index=False)

    print(f"Wrote {report_path}")
    print(f"Wrote {wave_path}")
    print(summary)


if __name__ == "__main__":
    main()
