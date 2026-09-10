# Calibrated Reciprocal Scoring

Platt calibration is fit only on each outer model's training scores and labels.
This avoids held-out-wave leakage but may be optimistic; it is used for reciprocal score construction rather than definitive calibration claims.

| policy | K | LikeRate@K | MutualOutcomeYield@K | NDCG(match)@K |
|---|---:|---:|---:|---:|
| candidate | 1 | 0.3848 | 0.2211 | 0.2653 |
| candidate | 3 | 0.4056 | 0.1846 | 0.2882 |
| candidate | 5 | 0.4282 | 0.2152 | 0.3939 |
| minimum | 1 | 0.3792 | 0.2069 | 0.2525 |
| minimum | 3 | 0.4574 | 0.2072 | 0.2976 |
| minimum | 5 | 0.4486 | 0.1947 | 0.3567 |
| product | 1 | 0.3502 | 0.1646 | 0.1950 |
| product | 3 | 0.4436 | 0.1989 | 0.2794 |
| product | 5 | 0.4646 | 0.2009 | 0.3512 |
| requester | 1 | 0.4432 | 0.1780 | 0.2147 |
| requester | 3 | 0.4893 | 0.2043 | 0.2759 |
| requester | 5 | 0.4589 | 0.1806 | 0.3135 |

`MutualOutcomeYield@K` is a directed static-replay metric, not a realized match count.
