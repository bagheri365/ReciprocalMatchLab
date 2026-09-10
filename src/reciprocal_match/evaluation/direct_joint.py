from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


def _ndcg_match_at_k(df: pd.DataFrame, score_col: str, k: int) -> float:
    values: list[float] = []
    for _, group in df.groupby(["wave", "iid"], sort=False):
        if len(group) < 2:
            continue
        y = group["y_match"].to_numpy(dtype=float)
        if y.sum() == 0:
            continue
        scores = group[score_col].to_numpy(dtype=float)
        values.append(
            float(
                ndcg_score(
                    y.reshape(1, -1),
                    scores.reshape(1, -1),
                    k=k,
                )
            )
        )
    return float(np.mean(values)) if values else float("nan")


def _topk_mean(df: pd.DataFrame, score_col: str, k: int, label: str) -> float:
    selected = []
    for _, group in df.groupby(["wave", "iid"], sort=False):
        selected.append(
            group.sort_values(
                [score_col, "pid"],
                ascending=[False, True],
                kind="mergesort",
            ).head(k)
        )
    if not selected:
        return float("nan")
    return float(pd.concat(selected, ignore_index=True)[label].mean())


def evaluate_direct_joint_vs_baselines(
    scored: pd.DataFrame,
    k_values: list[int],
) -> pd.DataFrame:
    score_cols = {
        "requester": "score_requester",
        "candidate": "score_candidate",
        "product": "score_product",
        "minimum": "score_minimum",
        "direct_joint": "score_direct_joint",
    }
    rows = []
    for wave, wave_df in scored.groupby("wave", sort=True):
        for policy, score_col in score_cols.items():
            for k in k_values:
                rows.append(
                    {
                        "wave": int(wave),
                        "policy": policy,
                        "k": int(k),
                        "mutual_outcome_yield_at_k": _topk_mean(
                            wave_df, score_col, k, "y_match"
                        ),
                        "ndcg_match_at_k": _ndcg_match_at_k(
                            wave_df, score_col, k
                        ),
                    }
                )
    return (
        pd.DataFrame(rows)
        .sort_values(["wave", "policy", "k"])
        .reset_index(drop=True)
    )


def paired_policy_differences(
    metrics: pd.DataFrame,
    reference_policy: str = "direct_joint",
) -> pd.DataFrame:
    rows = []
    for k, kdf in metrics.groupby("k"):
        ref = kdf[kdf["policy"] == reference_policy][
            ["wave", "mutual_outcome_yield_at_k", "ndcg_match_at_k"]
        ].rename(
            columns={
                "mutual_outcome_yield_at_k": "ref_yield",
                "ndcg_match_at_k": "ref_ndcg",
            }
        )

        for policy in sorted(set(kdf["policy"]) - {reference_policy}):
            cur = kdf[kdf["policy"] == policy][
                ["wave", "mutual_outcome_yield_at_k", "ndcg_match_at_k"]
            ]
            merged = ref.merge(cur, on="wave", validate="one_to_one")
            merged["delta_yield"] = (
                merged["ref_yield"] - merged["mutual_outcome_yield_at_k"]
            )
            merged["delta_ndcg"] = merged["ref_ndcg"] - merged["ndcg_match_at_k"]

            tol = 1e-12
            rows.append(
                {
                    "reference_policy": reference_policy,
                    "comparison_policy": policy,
                    "k": int(k),
                    "mean_delta_yield": float(merged["delta_yield"].mean()),
                    "median_delta_yield": float(merged["delta_yield"].median()),
                    "mean_delta_ndcg": float(merged["delta_ndcg"].mean()),
                    "wins": int((merged["delta_yield"] > tol).sum()),
                    "ties": int((merged["delta_yield"].abs() <= tol).sum()),
                    "losses": int((merged["delta_yield"] < -tol).sum()),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["k", "comparison_policy"])
        .reset_index(drop=True)
    )


def render_joint_summary(metrics: pd.DataFrame, paired: pd.DataFrame) -> str:
    agg = (
        metrics.groupby(["policy", "k"])[
            ["mutual_outcome_yield_at_k", "ndcg_match_at_k"]
        ]
        .mean()
        .reset_index()
    )

    lines = [
        "# Direct Symmetric Joint-Match Model",
        "",
        "The direct model is trained on unique undirected pairs with symmetric features.",
        "",
        "## Mean of per-wave metrics",
        "",
        "| policy | K | MutualOutcomeYield@K | NDCG(match)@K |",
        "|---|---:|---:|---:|",
    ]
    for row in agg.itertuples(index=False):
        lines.append(
            f"| {row.policy} | {int(row.k)} | "
            f"{row.mutual_outcome_yield_at_k:.4f} | {row.ndcg_match_at_k:.4f} |"
        )

    lines += [
        "",
        "## Direct-joint paired differences",
        "",
        "| comparison | K | mean Δyield | median Δyield | mean ΔNDCG | wins | ties | losses |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired.itertuples(index=False):
        lines.append(
            f"| {row.comparison_policy} | {int(row.k)} | "
            f"{row.mean_delta_yield:.4f} | {row.median_delta_yield:.4f} | "
            f"{row.mean_delta_ndcg:.4f} | {int(row.wins)} | "
            f"{int(row.ties)} | {int(row.losses)} |"
        )

    lines += [
        "",
        "Positive Δ means direct-joint outperformed the comparison policy.",
        "",
        "These are paired wave-level comparisons; they do not establish conditional dependence.",
        "",
    ]
    return "\n".join(lines)
