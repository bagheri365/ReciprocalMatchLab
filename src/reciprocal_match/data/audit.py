from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from reciprocal_match.data.pairs import (
    add_pair_id,
    match_identity_violations,
    reverse_pair_consistency,
    validate_required_columns,
)


@dataclass(frozen=True)
class AuditSummary:
    n_rows: int
    n_participants: int
    n_waves: int
    n_unique_pairs: int
    n_match_identity_violations: int
    n_reverse_pair_problems: int


def load_speed_dating_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding_errors="replace")
    validate_required_columns(df)
    return df


def build_audit_summary(df: pd.DataFrame) -> AuditSummary:
    with_ids = add_pair_id(df)

    participants = pd.concat(
        [df["iid"], df["pid"]], ignore_index=True
    ).dropna().nunique()

    return AuditSummary(
        n_rows=len(df),
        n_participants=int(participants),
        n_waves=int(df["wave"].dropna().nunique()),
        n_unique_pairs=int(with_ids["pair_id"].dropna().nunique()),
        n_match_identity_violations=len(match_identity_violations(df)),
        n_reverse_pair_problems=len(reverse_pair_consistency(df)),
    )


def wave_summary(df: pd.DataFrame) -> pd.DataFrame:
    with_ids = add_pair_id(df)

    grouped = with_ids.groupby("wave", dropna=False)
    rows = []
    for wave, g in grouped:
        participants = pd.concat([g["iid"], g["pid"]], ignore_index=True).dropna().nunique()
        rows.append(
            {
                "wave": wave,
                "directed_rows": len(g),
                "participants": int(participants),
                "unique_pairs": int(g["pair_id"].dropna().nunique()),
                "like_rate": float(g["dec"].mean()),
                "mutual_match_row_rate": float(g["match"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("wave", na_position="last")


def missingness_by_wave(df: pd.DataFrame) -> pd.DataFrame:
    rates = df.groupby("wave", dropna=False).apply(lambda x: x.isna().mean(), include_groups=False)
    rates.index.name = "wave"
    return rates.reset_index()


def render_markdown(summary: AuditSummary, df: pd.DataFrame) -> str:
    missing = df.isna().mean().sort_values(ascending=False)
    top_missing = missing.head(20)

    lines = [
        "# Data Audit",
        "",
        "## Structural summary",
        "",
        f"- Directed rows: **{summary.n_rows:,}**",
        f"- Unique participants: **{summary.n_participants:,}**",
        f"- Waves: **{summary.n_waves:,}**",
        f"- Unique undirected pairs: **{summary.n_unique_pairs:,}**",
        f"- `match == dec & dec_o` violations: **{summary.n_match_identity_violations:,}**",
        f"- Reverse-pair consistency problems: **{summary.n_reverse_pair_problems:,}**",
        "",
        "## Top missing columns",
        "",
        "| column | missing_rate |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {rate:.3f} |" for name, rate in top_missing.items())
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "Do not begin predictive modeling until protocol groups, leakage policy,",
            "and any reverse-pair inconsistencies are manually reviewed.",
            "",
        ]
    )
    return "\n".join(lines)
