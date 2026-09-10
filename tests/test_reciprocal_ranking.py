import pandas as pd
from reciprocal_match.evaluation.reciprocal import evaluate_reciprocal_policies

def test_ranking_schema():
    df=pd.DataFrame([{"iid":1,"pid":2,"wave":1,"y_like":1,"y_match":1,"score_requester":.9,"score_candidate":.8,"score_product":.72,"score_minimum":.8},{"iid":1,"pid":3,"wave":1,"y_like":1,"y_match":0,"score_requester":.8,"score_candidate":.1,"score_product":.08,"score_minimum":.1},{"iid":2,"pid":1,"wave":1,"y_like":1,"y_match":1,"score_requester":.8,"score_candidate":.9,"score_product":.72,"score_minimum":.8},{"iid":3,"pid":1,"wave":1,"y_like":0,"y_match":0,"score_requester":.1,"score_candidate":.8,"score_product":.08,"score_minimum":.1}])
    assert set(evaluate_reciprocal_policies(df,[1]).policy)=={"requester","candidate","product","minimum"}
