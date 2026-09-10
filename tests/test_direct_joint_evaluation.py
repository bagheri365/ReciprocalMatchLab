import pandas as pd
from reciprocal_match.evaluation.direct_joint import paired_policy_differences


def test_paired_differences_counts_wins_ties_losses():
    rows = []
    for wave, direct, candidate in [(1,.3,.2),(2,.1,.2),(3,.2,.2)]:
        rows += [
            {"wave":wave,"policy":"direct_joint","k":1,"mutual_outcome_yield_at_k":direct,"ndcg_match_at_k":direct},
            {"wave":wave,"policy":"candidate","k":1,"mutual_outcome_yield_at_k":candidate,"ndcg_match_at_k":candidate},
        ]
    out = paired_policy_differences(pd.DataFrame(rows), "direct_joint")
    row = out.iloc[0]
    assert row.wins == 1
    assert row.ties == 1
    assert row.losses == 1
