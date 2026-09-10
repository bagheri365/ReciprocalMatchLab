from pathlib import Path
import argparse,yaml
from reciprocal_match.data.audit import load_speed_dating_csv
from reciprocal_match.data.preprocessing import load_feature_decisions
from reciprocal_match.data.taxonomy import load_yaml
from reciprocal_match.evaluation.reciprocal import render_reciprocal_summary
from reciprocal_match.experiments.reciprocal_scoring import run_calibrated_reciprocal_loow
from reciprocal_match.models.directional import DirectionalModelSpec

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True)
    p.add_argument("--taxonomy",default="configs/feature_taxonomy.yaml"); p.add_argument("--decisions",default="configs/feature_decisions.yaml")
    p.add_argument("--contract",default="configs/experiment_contract.yaml"); p.add_argument("--model-config",default="configs/model_directional_logistic.yaml")
    p.add_argument("--reciprocal-config",default="configs/reciprocal_scoring.yaml"); a=p.parse_args()
    df=load_speed_dating_csv(a.input); tax=load_yaml(a.taxonomy); dec=load_feature_decisions(a.decisions)
    contract=yaml.safe_load(Path(a.contract).read_text()); mc=yaml.safe_load(Path(a.model_config).read_text()); rc=yaml.safe_load(Path(a.reciprocal_config).read_text())
    q=mc["model"]["params"]; spec=DirectionalModelSpec(C=float(q["C"]),solver=str(q["solver"]),max_iter=int(q["max_iter"]),random_state=int(q["random_state"]))
    r=run_calibrated_reciprocal_loow(df,tax,dec,list(contract["protocol"]["primary_waves"]),list(rc["ranking"]["k_values"]),spec=spec,calibration_eps=float(rc["calibration"]["clip_epsilon"]),random_state=int(rc["calibration"]["random_state"]))
    o=rc["outputs"]; pred=Path(o["predictions_csv"]); met=Path(o["per_wave_csv"]); summ=Path(o["summary_md"])
    for x in [pred,met,summ]: x.parent.mkdir(parents=True,exist_ok=True)
    r.predictions.to_csv(pred,index=False); r.per_wave_metrics.to_csv(met,index=False); summ.write_text(render_reciprocal_summary(r.per_wave_metrics))
    print(f"Wrote {pred}\nWrote {met}\nWrote {summ}")
    print(r.per_wave_metrics.groupby(["policy","k"])[["like_rate_at_k","mutual_outcome_yield_at_k","ndcg_match_at_k"]].mean().reset_index().to_string(index=False))
if __name__=="__main__": main()
