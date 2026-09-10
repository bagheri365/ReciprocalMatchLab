import numpy as np
import pandas as pd
from reciprocal_match.marketplace.allocation import (
    expected_random_exposure,gini,solve_exposure_controlled_allocation,target_pair_count
)


def _pairs():
    rows=[]
    users=[1,2,3,4]
    for i,u in enumerate(users):
        for v in users[i+1:]:
            rows.append({"wave":1,"u":u,"v":v,"y_match":int({u,v}=={1,2}),"score":1.0/(u+v)})
    return pd.DataFrame(rows)


def test_gini_zero_for_equal_exposure():
    assert gini(np.array([2,2,2,2])) == 0.0


def test_expected_random_exposure_sums_to_twice_pair_budget():
    q=expected_random_exposure(_pairs(),3)
    assert np.isclose(q.q_random.sum(),6.0)


def test_milp_respects_pair_budget_and_user_capacity():
    result=solve_exposure_controlled_allocation(_pairs(),capacity=2,target_pairs=3,lambda_value=.1)
    assert len(result.selected_pairs)==3
    assert result.user_exposure.exposure.max()<=2
    assert result.user_exposure.exposure.sum()==6


def test_target_pair_count_uses_fraction_of_user_capacity():
    assert target_pair_count(n_users=10,capacity=3,fraction=.6,n_edges=100)==9
