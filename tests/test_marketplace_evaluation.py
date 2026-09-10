import pandas as pd
from reciprocal_match.evaluation.marketplace import allocation_metrics,paired_tradeoff_differences


def test_allocation_metrics_pair_and_user_semantics():
    selected=pd.DataFrame([
        {"u":1,"v":2,"y_match":1,"score":.8},
        {"u":3,"v":4,"y_match":0,"score":.7},
    ])
    exposure=pd.DataFrame({"user":[1,2,3,4],"eligible_degree":[3]*4,"q_random":[1.]*4,
                           "exposure":[1,1,1,1],"relative_exposure":[1.,1.,1.,1.]})
    m=allocation_metrics(selected,exposure)
    assert m["pair_match_rate"]==.5
    assert m["successful_user_rate"]==.5
    assert m["pair_coverage"]==1.0
    assert m["raw_exposure_gini"]==0.0


def test_paired_tradeoff_signs():
    df=pd.DataFrame([
        {"wave":1,"policy":"optimized","lambda":0.,"pair_match_rate":.3,"relative_exposure_gini":.4},
        {"wave":2,"policy":"optimized","lambda":0.,"pair_match_rate":.2,"relative_exposure_gini":.5},
        {"wave":1,"policy":"optimized","lambda":.1,"pair_match_rate":.2,"relative_exposure_gini":.2},
        {"wave":2,"policy":"optimized","lambda":.1,"pair_match_rate":.2,"relative_exposure_gini":.3},
    ])
    out=paired_tradeoff_differences(df).iloc[0]
    assert out.mean_delta_pair_match_rate < 0
    assert out.mean_delta_relative_exposure_gini < 0
    assert out.concentration_improved_waves == 2
