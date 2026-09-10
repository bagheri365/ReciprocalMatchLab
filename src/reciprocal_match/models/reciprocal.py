from __future__ import annotations
import numpy as np
import pandas as pd

REQUIRED={"iid","pid","wave","y_like","y_match","p_like_raw","p_like_calibrated"}

def add_reverse_probabilities(predictions: pd.DataFrame) -> pd.DataFrame:
    missing=REQUIRED-set(predictions.columns)
    if missing:
        raise ValueError(f"Missing directional columns: {sorted(missing)}")
    rev=predictions[["iid","pid","wave","p_like_raw","p_like_calibrated"]].rename(
        columns={"iid":"pid","pid":"iid","p_like_raw":"p_ba_raw","p_like_calibrated":"p_ba"}
    )
    out=predictions.merge(rev,on=["iid","pid","wave"],how="left",validate="one_to_one")
    if out[["p_ba_raw","p_ba"]].isna().any().any():
        raise ValueError("Missing reverse-row prediction")
    out["score_requester"]=out["p_like_calibrated"]
    out["score_candidate"]=out["p_ba"]
    out["score_product"]=out["p_like_calibrated"]*out["p_ba"]
    out["score_minimum"]=np.minimum(out["p_like_calibrated"],out["p_ba"])
    return out

def validate_reverse_alignment(scored: pd.DataFrame, atol=1e-12):
    rev=scored[["iid","pid","wave","p_like_calibrated","y_match"]].rename(
        columns={"iid":"pid","pid":"iid","p_like_calibrated":"reverse_p_like","y_match":"reverse_match"}
    )
    chk=scored.merge(rev,on=["iid","pid","wave"],how="left",validate="one_to_one")
    if chk[["reverse_p_like","reverse_match"]].isna().any().any():
        raise AssertionError("Missing reverse row during validation")
    if not np.allclose(chk["p_ba"].to_numpy(float),chk["reverse_p_like"].to_numpy(float),atol=atol,rtol=0):
        raise AssertionError("Reverse probability misalignment")
    if not (chk["y_match"].to_numpy()==chk["reverse_match"].to_numpy()).all():
        raise AssertionError("Reverse rows disagree on mutual outcome")
