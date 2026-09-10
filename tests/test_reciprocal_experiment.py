import pandas as pd
from reciprocal_match.experiments.reciprocal_scoring import run_calibrated_reciprocal_loow


def _toy_data():
    rows=[]
    for wave in [1,2,3]:
        ids=[wave*10+i for i in [1,2,3,4]]
        ages=dict(zip(ids,[20+wave,25+wave,30+wave,35+wave]))
        goals=dict(zip(ids,[1,1,2,2]))
        for i in ids:
            for j in ids:
                if i==j: continue
                rows.append({
                    "iid":i,"pid":j,"wave":wave,
                    "dec":int(ages[j]>ages[i]),"match":0,
                    "age":ages[i],"goal":goals[i],"int_corr":0.2,
                })
    return pd.DataFrame(rows)


def test_reciprocal_experiment_scores_every_outer_row():
    taxonomy={
        "categories":{"preinteraction_primary":["age","goal"],"derived_preinteraction_pair":["int_corr"]},
        "modeling":{"allowed_primary_groups":["preinteraction_primary","derived_preinteraction_pair"]},
    }
    df=_toy_data()
    result=run_calibrated_reciprocal_loow(
        df,taxonomy,{"categorical_features":["goal"]},[1,2,3],[1]
    )
    assert len(result.predictions)==len(df)
    assert result.predictions["p_ba"].notna().all()
    assert set(result.per_wave_metrics["wave"])=={1,2,3}
