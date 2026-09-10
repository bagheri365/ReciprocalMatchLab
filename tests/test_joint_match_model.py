import pandas as pd
from reciprocal_match.models.joint_match import make_joint_match_pipeline, predict_match_probability


def test_joint_model_probability_range():
    X = pd.DataFrame({
        "sym_absdiff__age": [1.0, 2.0, 8.0, 9.0],
        "sym_mean__age": [25.0, 26.0, 35.0, 36.0],
    })
    y = pd.Series([1, 1, 0, 0])
    model = make_joint_match_pipeline()
    model.fit(X, y)
    p = predict_match_probability(model, X)
    assert ((p >= 0) & (p <= 1)).all()
