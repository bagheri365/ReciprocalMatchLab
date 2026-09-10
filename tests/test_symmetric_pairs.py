import pandas as pd
from reciprocal_match.data.symmetric_pairs import build_unique_pair_frame


def test_unique_pair_frame_has_one_row_per_pair_and_symmetric_features():
    df = pd.DataFrame([
        {"iid":1,"pid":2,"wave":1,"dec":1,"match":1,"age":20,"goal":1,"int_corr":.5},
        {"iid":2,"pid":1,"wave":1,"dec":1,"match":1,"age":30,"goal":2,"int_corr":.5},
    ])
    taxonomy = {
        "categories": {
            "preinteraction_primary": ["age", "goal"],
            "derived_preinteraction_pair": ["int_corr"],
        },
        "modeling": {"allowed_primary_groups": ["preinteraction_primary", "derived_preinteraction_pair"]},
    }
    X, y, meta = build_unique_pair_frame(df, taxonomy)
    assert len(X) == 1 and len(y) == 1
    assert meta.iloc[0].u == 1 and meta.iloc[0].v == 2
    assert X.iloc[0]["sym_absdiff__age"] == 10
