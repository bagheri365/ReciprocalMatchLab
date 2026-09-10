import pandas as pd

from reciprocal_match.data.audit import build_audit_summary, wave_summary


def test_audit_summary_counts_unique_pairs_not_directed_rows():
    df = pd.DataFrame(
        [
            {"iid": 1, "pid": 2, "wave": 1, "dec": 1, "dec_o": 1, "match": 1},
            {"iid": 2, "pid": 1, "wave": 1, "dec": 1, "dec_o": 1, "match": 1},
        ]
    )

    summary = build_audit_summary(df)

    assert summary.n_rows == 2
    assert summary.n_participants == 2
    assert summary.n_waves == 1
    assert summary.n_unique_pairs == 1
    assert summary.n_match_identity_violations == 0
    assert summary.n_reverse_pair_problems == 0


def test_wave_summary_uses_unique_pair_count():
    df = pd.DataFrame(
        [
            {"iid": 1, "pid": 2, "wave": 1, "dec": 1, "dec_o": 0, "match": 0},
            {"iid": 2, "pid": 1, "wave": 1, "dec": 0, "dec_o": 1, "match": 0},
        ]
    )

    out = wave_summary(df)

    assert out.iloc[0]["directed_rows"] == 2
    assert out.iloc[0]["unique_pairs"] == 1
