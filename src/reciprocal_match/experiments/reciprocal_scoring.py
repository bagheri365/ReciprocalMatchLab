from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from reciprocal_match.data.preprocessing import build_directed_feature_frame
from reciprocal_match.evaluation.reciprocal import evaluate_reciprocal_policies
from reciprocal_match.models.calibration import fit_platt_calibrator
from reciprocal_match.models.directional import DirectionalModelSpec,make_directional_logistic_pipeline,predict_positive_probability
from reciprocal_match.models.reciprocal import add_reverse_probabilities,validate_reverse_alignment

@dataclass(frozen=True)
class ReciprocalExperimentResult:
    predictions: pd.DataFrame
    per_wave_metrics: pd.DataFrame

def run_calibrated_reciprocal_loow(df,taxonomy,decisions,primary_waves,k_values,spec=None,calibration_eps=1e-6,random_state=365):
    spec=spec or DirectionalModelSpec()
    scoped=df[df["wave"].isin(primary_waves)].copy()
    X,y_like,y_match,meta=build_directed_feature_frame(scoped,taxonomy)
    frames=[]
    for test_wave in primary_waves:
        test_mask=meta["wave"].astype(int).eq(int(test_wave))
        train_mask=~test_mask
        Xtr=X.loc[train_mask]; ytr=y_like.loc[train_mask]; Xte=X.loc[test_mask]
        base=make_directional_logistic_pipeline(Xtr,decisions,spec)
        base.fit(Xtr,ytr)
        raw_train=predict_positive_probability(base,Xtr)
        cal=fit_platt_calibrator(raw_train,ytr.to_numpy(),eps=calibration_eps,random_state=random_state)
        raw_test=predict_positive_probability(base,Xte)
        p=meta.loc[test_mask].copy()
        p["y_like"]=y_like.loc[test_mask].to_numpy(); p["y_match"]=y_match.loc[test_mask].to_numpy()
        p["p_like_raw"]=raw_test; p["p_like_calibrated"]=cal.predict(raw_test)
        frames.append(p)
    pred=pd.concat(frames,ignore_index=True).sort_values(["wave","iid","pid"]).reset_index(drop=True)
    scored=add_reverse_probabilities(pred); validate_reverse_alignment(scored)
    if len(scored)!=len(scoped):
        raise AssertionError("Missing held-out reciprocal scores")
    return ReciprocalExperimentResult(scored,evaluate_reciprocal_policies(scored,k_values))
