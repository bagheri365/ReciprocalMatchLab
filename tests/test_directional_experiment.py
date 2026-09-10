import pandas as pd

from reciprocal_match.experiments.directional_baseline import run_directional_loow


def _toy_data():
    rows = []
    # Three waves; each has four participants and all cross-pairs represented
    # in both directions. Stable participant features within wave are sufficient
    # for the synthetic leakage/determinism test.
    for wave in [1, 2, 3]:
        ages = {1: 20 + wave, 2: 25 + wave, 3: 30 + wave, 4: 35 + wave}
        goals = {1: 1, 2: 1, 3: 2, 4: 2}
        for i in ages:
            for j in ages:
                if i == j:
                    continue
                dec = int((ages[j] - ages[i]) > 0)
                match = 0
                rows.append({
                    "iid": wave * 10 + i,
                    "pid": wave * 10 + j,
                    "wave": wave,
                    "dec": dec,
                    "match": match,
                    "age": ages[i],
                    "goal": goals[i],
                    "int_corr": 0.1 * (i + j),
                })
    return pd.DataFrame(rows)


def _taxonomy():
    return {
        "categories": {
            "preinteraction_primary": ["age", "goal"],
            "derived_preinteraction_pair": ["int_corr"],
        },
        "modeling": {
            "allowed_primary_groups": [
                "preinteraction_primary", "derived_preinteraction_pair"
            ]
        },
    }


def _decisions():
    return {"categorical_features": ["goal"]}


def test_loow_outputs_one_prediction_per_primary_row_and_is_deterministic():
    df = _toy_data()
    r1 = run_directional_loow(df, _taxonomy(), _decisions(), [1, 2, 3])
    r2 = run_directional_loow(df, _taxonomy(), _decisions(), [1, 2, 3])

    assert len(r1.predictions) == len(df)
    assert list(r1.per_wave["wave"]) == [1, 2, 3]
    pd.testing.assert_frame_equal(r1.per_wave, r2.per_wave)
    pd.testing.assert_frame_equal(r1.predictions, r2.predictions)


def test_prediction_probabilities_are_in_unit_interval():
    r = run_directional_loow(_toy_data(), _taxonomy(), _decisions(), [1, 2, 3])
    assert r.predictions["p_like"].between(0, 1).all()
