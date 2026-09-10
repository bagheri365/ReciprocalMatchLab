# Match Probability Evaluation

Probability evaluation uses one unique undirected row per historical pair.

## Pooled held-out probability metrics

| model | log loss | Brier | weighted reliability gap |
|---|---:|---:|---:|
| product | 0.5658 | 0.1633 | 0.1151 |
| direct_joint | 0.5380 | 0.1599 | 0.1074 |

## Mean of per-wave probability metrics

| model | log loss | Brier |
|---|---:|---:|
| direct_joint | 0.5645 | 0.1667 |
| product | 0.6134 | 0.1760 |

## Paired wave-level comparison: direct joint minus product

Positive delta means direct joint has lower loss and therefore performs better.

| metric | mean delta | median delta | wins | ties | losses |
|---|---:|---:|---:|---:|---:|
| log_loss | 0.0489 | 0.0355 | 6 | 0 | 3 |
| brier | 0.0093 | 0.0088 | 5 | 0 | 4 |

## Interpretation limits

The product is an independence-based construction from directional probabilities;
the direct joint score is a separately trained match probability model.
Performance differences do not establish or refute conditional independence.

The directional Platt calibrator used upstream is fit on outer-training scores.
That avoids held-out-wave leakage but can be optimistic, so reliability results
should be read as held-out diagnostics of the current pipeline rather than a
claim that the directional marginals are perfectly calibrated.
