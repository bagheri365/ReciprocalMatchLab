from __future__ import annotations

import numpy as np
import pandas as pd

from reciprocal_match.marketplace.allocation import gini


def allocation_metrics(selected: pd.DataFrame, exposure: pd.DataFrame) -> dict[str, float]:
    if selected.empty:
        raise ValueError("Cannot evaluate an empty allocation")
    eligible_users = set(exposure["user"].astype(int))
    successful = set()
    for r in selected[selected["y_match"] == 1].itertuples(index=False):
        successful.add(int(r.u)); successful.add(int(r.v))
    covered = set(exposure.loc[exposure["exposure"] > 0, "user"].astype(int))
    return {
        "pair_match_rate": float(selected["y_match"].mean()),
        "successful_user_rate": float(len(successful) / len(eligible_users)),
        "pair_coverage": float(len(covered) / len(eligible_users)),
        "raw_exposure_gini": gini(exposure["exposure"]),
        "relative_exposure_gini": gini(exposure["relative_exposure"]),
        "mean_selected_score": float(selected["score"].mean()),
    }


def aggregate_random_repeats(rows: list[dict]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    metrics = [
        "pair_match_rate", "successful_user_rate", "pair_coverage",
        "raw_exposure_gini", "relative_exposure_gini", "mean_selected_score",
    ]
    out = {m: float(frame[m].mean()) for m in metrics}
    out.update({f"{m}_sd": float(frame[m].std(ddof=1)) for m in metrics})
    return out


def tradeoff_summary(per_wave: pd.DataFrame) -> pd.DataFrame:
    optimized = per_wave[per_wave["policy"] == "optimized"].copy()
    return optimized.groupby("lambda", as_index=False)[
        ["pair_match_rate","successful_user_rate","pair_coverage",
         "raw_exposure_gini","relative_exposure_gini","mean_selected_score"]
    ].mean().sort_values("lambda").reset_index(drop=True)


def paired_tradeoff_differences(per_wave: pd.DataFrame) -> pd.DataFrame:
    optimized = per_wave[per_wave["policy"] == "optimized"].copy()
    baseline = optimized[np.isclose(optimized["lambda"], 0.0)].set_index("wave")
    rows=[]
    for lam, group in optimized.groupby("lambda", sort=True):
        if np.isclose(lam, 0.0):
            continue
        cur = group.set_index("wave")
        joined = baseline[["pair_match_rate","relative_exposure_gini"]].join(
            cur[["pair_match_rate","relative_exposure_gini"]],
            lsuffix="_base", rsuffix="_cur", how="inner"
        )
        delta_yield = joined["pair_match_rate_cur"] - joined["pair_match_rate_base"]
        delta_conc = joined["relative_exposure_gini_cur"] - joined["relative_exposure_gini_base"]
        tol=1e-12
        rows.append({
            "lambda": float(lam),
            "mean_delta_pair_match_rate": float(delta_yield.mean()),
            "median_delta_pair_match_rate": float(delta_yield.median()),
            "mean_delta_relative_exposure_gini": float(delta_conc.mean()),
            "median_delta_relative_exposure_gini": float(delta_conc.median()),
            "yield_wins": int((delta_yield > tol).sum()),
            "yield_ties": int((delta_yield.abs() <= tol).sum()),
            "yield_losses": int((delta_yield < -tol).sum()),
            "concentration_improved_waves": int((delta_conc < -tol).sum()),
        })
    return pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)


def render_marketplace_summary(per_wave: pd.DataFrame, tradeoff: pd.DataFrame, paired: pd.DataFrame) -> str:
    random_mean = per_wave[per_wave["policy"] == "random"].mean(numeric_only=True)
    lines=[
        "# Pair-Level Marketplace Allocation", "",
        "Each selected undirected edge represents a mutual interaction opportunity for both participants.",
        "Historical outcomes are reused under a static replay assumption; this is not a causal policy estimate.", "",
        "## Random eligible-edge baseline", "",
        f"Mean PairMatchRate: **{random_mean['pair_match_rate']:.4f}**", "",
        f"Mean SuccessfulUserRate: **{random_mean['successful_user_rate']:.4f}**", "",
        f"Mean relative-exposure Gini: **{random_mean['relative_exposure_gini']:.4f}**", "",
        "## Optimized exposure trade-off", "",
        "| lambda | PairMatchRate | SuccessfulUserRate | Coverage | raw Gini | relative Gini | mean score |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in tradeoff.to_dict(orient="records"):
        lines.append(
            f"| {r['lambda']:.2f} | {r['pair_match_rate']:.4f} | {r['successful_user_rate']:.4f} | "
            f"{r['pair_coverage']:.4f} | {r['raw_exposure_gini']:.4f} | {r['relative_exposure_gini']:.4f} | {r['mean_selected_score']:.4f} |"
        )
    lines += ["", "## Paired change versus lambda=0", "",
              "Negative delta relative-exposure Gini means exposure became less concentrated.", "",
              "| lambda | mean ΔPairMatchRate | median ΔPairMatchRate | mean Δrelative Gini | improved-concentration waves |",
              "|---:|---:|---:|---:|---:|"]
    for r in paired.to_dict(orient="records"):
        lines.append(
            f"| {r['lambda']:.2f} | {r['mean_delta_pair_match_rate']:.4f} | {r['median_delta_pair_match_rate']:.4f} | "
            f"{r['mean_delta_relative_exposure_gini']:.4f} | {int(r['concentration_improved_waves'])} |"
        )
    lines += ["", "## Interpretation limits", "",
              "Relative exposure is normalized by uniform random eligible-edge opportunity, not a fairness definition.",
              "The allocation objective is batch-optimized and order-independent; the random baseline is Monte Carlo.", ""]
    return "\n".join(lines)
