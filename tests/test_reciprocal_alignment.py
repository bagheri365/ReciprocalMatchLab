import pandas as pd,pytest
from reciprocal_match.models.reciprocal import add_reverse_probabilities,validate_reverse_alignment

def _p():
    return pd.DataFrame([{"iid":1,"pid":2,"wave":1,"y_like":1,"y_match":0,"p_like_raw":.8,"p_like_calibrated":.7},{"iid":2,"pid":1,"wave":1,"y_like":0,"y_match":0,"p_like_raw":.3,"p_like_calibrated":.2}])

def test_reverse_alignment():
    s=add_reverse_probabilities(_p()); assert s.iloc[0].p_ba==.2; validate_reverse_alignment(s)

def test_missing_reverse_rejected():
    with pytest.raises(ValueError): add_reverse_probabilities(_p().iloc[[0]])
