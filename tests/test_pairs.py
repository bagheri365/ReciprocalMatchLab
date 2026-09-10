import pandas as pd

from reciprocal_match.data.pairs import (
    add_pair_id,
    match_identity_violations,
    reverse_pair_consistency,
)


def _valid_pair_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"iid": 1, "pid": 2, "wave": 1, "dec": 1, "dec_o": 0, "match": 0},
            {"iid": 2, "pid": 1, "wave": 1, "dec": 0, "dec_o": 1, "match": 0},
            {"iid": 1, "pid": 3, "wave": 1, "dec": 1, "dec_o": 1, "match": 1},
            {"iid": 3, "pid": 1, "wave": 1, "dec": 1, "dec_o": 1, "match": 1},
        ]
    )


def test_pair_id_is_symmetric():
    out = add_pair_id(_valid_pair_df())
    assert out.loc[0, "pair_id"] == out.loc[1, "pair_id"]
    assert out.loc[2, "pair_id"] == out.loc[3, "pair_id"]


def test_match_identity_has_no_violations_for_valid_data():
    assert match_identity_violations(_valid_pair_df()).empty


def test_match_identity_detects_bad_label():
    df = _valid_pair_df()
    df.loc[0, "match"] = 1
    assert len(match_identity_violations(df)) == 1


def test_reverse_pair_consistency_accepts_valid_pairs():
    assert reverse_pair_consistency(_valid_pair_df()).empty


def test_reverse_pair_consistency_detects_mismatch():
    df = _valid_pair_df()
    df.loc[1, "dec_o"] = 0
    problems = reverse_pair_consistency(df)
    assert len(problems) == 1
    assert problems.iloc[0]["problem"] == "reverse_inconsistency"
