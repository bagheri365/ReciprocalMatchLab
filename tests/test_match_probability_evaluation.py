import pandas as pd
import pytest

from reciprocal_match.evaluation.match_probability import (
    fixed_width_reliability_table,
    paired_probability_differences,
    per_wave_probability_metrics,
    unique_pair_probabilities,
)


def _directed_predictions():
    rows = []
    for wave, pairs in {
        1: [(1, 2, 1, .30, .25), (1, 3, 0, .10, .08)],
        2: [(4, 5, 1, .40, .50), (4, 6, 0, .20, .15)],
    }.items():
        for u, v, y, product, direct in pairs:
            for iid, pid in [(u, v), (v, u)]:
                rows.append({
                    "wave": wave,
                    "iid": iid,
                    "pid": pid,
                    "y_match": y,
                    "score_product": product,
                    "p_match_direct": direct,
                })
    return pd.DataFrame(rows)


def test_unique_pair_probability_frame_deduplicates_reverse_rows():
    pairs = unique_pair_probabilities(_directed_predictions())
    assert len(pairs) == 4
    assert set(pairs.columns) == {
        "wave", "u", "v", "y_match", "p_product", "p_direct_joint"
    }


def test_unique_pair_probability_frame_rejects_reverse_disagreement():
    df = _directed_predictions()
    df.loc[1, "score_product"] = 0.99
    with pytest.raises(ValueError):
        unique_pair_probabilities(df)


def test_reliability_table_counts_each_pair_once():
    pairs = unique_pair_probabilities(_directed_predictions())
    table = fixed_width_reliability_table(pairs, n_bins=5)
    for model in ["product", "direct_joint"]:
        assert table.query("model == @model")["n_pairs"].sum() == len(pairs)


def test_per_wave_metrics_have_two_models_per_wave():
    pairs = unique_pair_probabilities(_directed_predictions())
    metrics = per_wave_probability_metrics(pairs)
    assert len(metrics) == 4
    assert set(metrics["model"]) == {"product", "direct_joint"}
    assert metrics["log_loss"].gt(0).all()
    assert metrics["brier"].between(0, 1).all()


def test_paired_difference_positive_when_direct_is_better():
    per_wave = pd.DataFrame([
        {"wave": 1, "model": "direct_joint", "log_loss": .40, "brier": .15},
        {"wave": 1, "model": "product", "log_loss": .50, "brier": .20},
        {"wave": 2, "model": "direct_joint", "log_loss": .30, "brier": .10},
        {"wave": 2, "model": "product", "log_loss": .35, "brier": .12},
    ])
    paired = paired_probability_differences(per_wave)
    assert (paired["mean_delta"] > 0).all()
    assert (paired["wins"] == 2).all()
