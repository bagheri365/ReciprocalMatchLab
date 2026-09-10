from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


MODEL_COLUMNS = {
    "product": "score_product",
    "direct_joint": "p_match_direct",
}


def unique_pair_probabilities(predictions: pd.DataFrame) -> pd.DataFrame:
    """Collapse directed A->B/B->A rows to one undirected pair.

    Reciprocal product and direct-joint probabilities must agree across reverse
    directed rows before deduplication. This prevents silently evaluating a
    duplicated or misaligned probability table.
    """
    required = {"wave", "iid", "pid", "y_match", *MODEL_COLUMNS.values()}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing probability-evaluation columns: {missing}")

    frame = predictions.copy()
    frame["u"] = frame[["iid", "pid"]].min(axis=1)
    frame["v"] = frame[["iid", "pid"]].max(axis=1)

    rows = []
    for (wave, u, v), group in frame.groupby(["wave", "u", "v"], sort=True):
        if len(group) != 2:
            raise ValueError(
                f"Expected exactly two directed rows for pair {(int(wave), int(u), int(v))}; "
                f"found {len(group)}"
            )

        y_values = group["y_match"].dropna().unique()
        if len(y_values) != 1:
            raise ValueError(f"Reverse rows disagree on y_match for pair {(wave, u, v)}")

        row = {
            "wave": int(wave),
            "u": int(u),
            "v": int(v),
            "y_match": int(y_values[0]),
        }
        for model, column in MODEL_COLUMNS.items():
            values = group[column].to_numpy(dtype=float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError(f"{column} must contain finite probabilities in [0, 1]")
            if not np.allclose(values, values[0], atol=1e-12, rtol=0.0):
                raise ValueError(
                    f"Reverse rows disagree on {column} for pair {(wave, u, v)}"
                )
            row[f"p_{model}"] = float(values[0])
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["wave", "u", "v"]).reset_index(drop=True)


def probability_metrics(y_true: pd.Series, probability: pd.Series) -> dict[str, float]:
    y = pd.Series(y_true).astype(int)
    p = np.asarray(probability, dtype=float)
    if len(y) == 0:
        raise ValueError("Cannot score an empty probability table")
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def fixed_width_reliability_table(
    pairs: pd.DataFrame,
    n_bins: int = 10,
) -> pd.DataFrame:
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for model in MODEL_COLUMNS:
        p_col = f"p_{model}"
        bins = pd.cut(
            pairs[p_col],
            bins=edges,
            labels=False,
            include_lowest=True,
            right=True,
        )
        temp = pairs.assign(_bin=bins)
        for bin_id, group in temp.groupby("_bin", observed=True, sort=True):
            if pd.isna(bin_id):
                continue
            idx = int(bin_id)
            mean_pred = float(group[p_col].mean())
            observed = float(group["y_match"].mean())
            rows.append(
                {
                    "model": model,
                    "bin": idx,
                    "bin_lower": float(edges[idx]),
                    "bin_upper": float(edges[idx + 1]),
                    "n_pairs": int(len(group)),
                    "mean_predicted": mean_pred,
                    "observed_match_rate": observed,
                    "absolute_gap": abs(mean_pred - observed),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "bin"]).reset_index(drop=True)


def weighted_absolute_reliability_gap(reliability: pd.DataFrame, model: str) -> float:
    part = reliability[reliability["model"] == model]
    if part.empty:
        return float("nan")
    weights = part["n_pairs"].to_numpy(dtype=float)
    gaps = part["absolute_gap"].to_numpy(dtype=float)
    return float(np.average(gaps, weights=weights))


def per_wave_probability_metrics(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wave, group in pairs.groupby("wave", sort=True):
        for model in MODEL_COLUMNS:
            metrics = probability_metrics(group["y_match"], group[f"p_{model}"])
            rows.append(
                {
                    "wave": int(wave),
                    "model": model,
                    "n_pairs": int(len(group)),
                    "match_rate": float(group["y_match"].mean()),
                    **metrics,
                }
            )
    return pd.DataFrame(rows).sort_values(["wave", "model"]).reset_index(drop=True)


def paired_probability_differences(per_wave: pd.DataFrame) -> pd.DataFrame:
    direct = per_wave[per_wave["model"] == "direct_joint"].set_index("wave")
    product = per_wave[per_wave["model"] == "product"].set_index("wave")
    if set(direct.index) != set(product.index):
        raise ValueError("Direct and product metrics must cover the same waves")

    joined = direct[["log_loss", "brier"]].join(
        product[["log_loss", "brier"]],
        lsuffix="_direct",
        rsuffix="_product",
        how="inner",
    )
    # Positive delta means the direct joint model has lower (better) loss.
    joined["delta_log_loss"] = joined["log_loss_product"] - joined["log_loss_direct"]
    joined["delta_brier"] = joined["brier_product"] - joined["brier_direct"]
    tol = 1e-12

    return pd.DataFrame(
        [
            {
                "reference_model": "direct_joint",
                "comparison_model": "product",
                "metric": metric,
                "mean_delta": float(joined[f"delta_{metric}"].mean()),
                "median_delta": float(joined[f"delta_{metric}"].median()),
                "wins": int((joined[f"delta_{metric}"] > tol).sum()),
                "ties": int((joined[f"delta_{metric}"].abs() <= tol).sum()),
                "losses": int((joined[f"delta_{metric}"] < -tol).sum()),
            }
            for metric in ["log_loss", "brier"]
        ]
    )


def render_match_probability_summary(
    pairs: pd.DataFrame,
    per_wave: pd.DataFrame,
    reliability: pd.DataFrame,
    paired: pd.DataFrame,
) -> str:
    pooled_rows = []
    for model in MODEL_COLUMNS:
        metrics = probability_metrics(pairs["y_match"], pairs[f"p_{model}"])
        pooled_rows.append(
            {
                "model": model,
                **metrics,
                "reliability_gap": weighted_absolute_reliability_gap(reliability, model),
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    wave_mean = per_wave.groupby("model")[["log_loss", "brier"]].mean().reset_index()

    lines = [
        "# Match Probability Evaluation",
        "",
        "Probability evaluation uses one unique undirected row per historical pair.",
        "",
        "## Pooled held-out probability metrics",
        "",
        "| model | log loss | Brier | weighted reliability gap |",
        "|---|---:|---:|---:|",
    ]
    for row in pooled.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.log_loss:.4f} | {row.brier:.4f} | "
            f"{row.reliability_gap:.4f} |"
        )

    lines += [
        "",
        "## Mean of per-wave probability metrics",
        "",
        "| model | log loss | Brier |",
        "|---|---:|---:|",
    ]
    for row in wave_mean.itertuples(index=False):
        lines.append(f"| {row.model} | {row.log_loss:.4f} | {row.brier:.4f} |")

    lines += [
        "",
        "## Paired wave-level comparison: direct joint minus product",
        "",
        "Positive delta means direct joint has lower loss and therefore performs better.",
        "",
        "| metric | mean delta | median delta | wins | ties | losses |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in paired.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.mean_delta:.4f} | {row.median_delta:.4f} | "
            f"{int(row.wins)} | {int(row.ties)} | {int(row.losses)} |"
        )

    lines += [
        "",
        "## Interpretation limits",
        "",
        "The product is an independence-based construction from directional probabilities;",
        "the direct joint score is a separately trained match probability model.",
        "Performance differences do not establish or refute conditional independence.",
        "",
        "The directional Platt calibrator used upstream is fit on outer-training scores.",
        "That avoids held-out-wave leakage but can be optimistic, so reliability results",
        "should be read as held-out diagnostics of the current pipeline rather than a",
        "claim that the directional marginals are perfectly calibrated.",
        "",
    ]
    return "\n".join(lines)
