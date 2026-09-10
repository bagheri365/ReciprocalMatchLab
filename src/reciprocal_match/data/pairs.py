from __future__ import annotations

import pandas as pd


REQUIRED_PAIR_COLUMNS = {"iid", "pid", "wave", "dec", "dec_o", "match"}


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_PAIR_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def add_pair_id(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a canonical undirected pair identifier.

    pair_id is a tuple: (wave, min(iid, pid), max(iid, pid)).
    Rows with missing iid/pid/wave receive a missing pair_id.
    """
    validate_required_columns(df)
    out = df.copy()

    valid = out[["wave", "iid", "pid"]].notna().all(axis=1)
    pair_id = pd.Series(pd.NA, index=out.index, dtype="object")

    pair_id.loc[valid] = [
        (wave, min(iid, pid), max(iid, pid))
        for wave, iid, pid in out.loc[valid, ["wave", "iid", "pid"]].itertuples(
            index=False, name=None
        )
    ]
    out["pair_id"] = pair_id
    return out


def match_identity_violations(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where match != (dec == 1 and dec_o == 1), excluding missing labels."""
    validate_required_columns(df)
    mask = df[["dec", "dec_o", "match"]].notna().all(axis=1)
    expected = ((df.loc[mask, "dec"] == 1) & (df.loc[mask, "dec_o"] == 1)).astype(int)
    bad = df.loc[mask].loc[df.loc[mask, "match"].astype(int) != expected]
    return bad.copy()


def reverse_pair_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Return pair-level rows whose two directed records disagree.

    A fully represented pair should have exactly two directed rows:
    A->B and B->A. For such pairs, dec(A,B) should equal dec_o(B,A),
    dec_o(A,B) should equal dec(B,A), and match should agree.
    """
    with_ids = add_pair_id(df)
    problems: list[dict] = []

    for pair_id, group in with_ids.dropna(subset=["pair_id"]).groupby("pair_id"):
        if len(group) != 2:
            problems.append(
                {"pair_id": pair_id, "problem": "directed_row_count", "row_count": len(group)}
            )
            continue

        r1, r2 = list(group.itertuples(index=False))
        ok_reverse_ids = (r1.iid == r2.pid) and (r1.pid == r2.iid)
        ok_decisions = (
            r1.dec == r2.dec_o
            and r1.dec_o == r2.dec
            and r1.match == r2.match
        )

        if not (ok_reverse_ids and ok_decisions):
            problems.append(
                {
                    "pair_id": pair_id,
                    "problem": "reverse_inconsistency",
                    "row_count": 2,
                }
            )

    return pd.DataFrame(problems, columns=["pair_id", "problem", "row_count"])
