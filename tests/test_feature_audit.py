import pandas as pd

from reciprocal_match.data.feature_audit import feature_missingness_summary


def test_feature_missingness_uses_max_per_wave_not_only_pooled_rate():
    df = pd.DataFrame(
        {
            "wave": [1, 1, 2, 2],
            "age": [20.0, 21.0, None, None],
        }
    )
    taxonomy = {
        "categories": {
            "preinteraction_primary": ["age"],
            "derived_preinteraction_pair": [],
        },
        "modeling": {
            "allowed_primary_groups": [
                "preinteraction_primary",
                "derived_preinteraction_pair",
            ],
            "explicitly_blocked_groups": [],
        },
    }

    out = feature_missingness_summary(df, taxonomy)

    assert out.iloc[0]["overall_missing_rate"] == 0.5
    assert out.iloc[0]["max_wave_missing_rate"] == 1.0
