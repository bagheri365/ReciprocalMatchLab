# Directional Logistic Regression Baseline

## Scope

Leakage-safe leave-one-wave-out evaluation on the frozen primary protocol group.
Preprocessing is fit separately inside every training fold.

## Aggregate metrics

| aggregation | log_loss | roc_auc | pr_auc | brier |
|---|---:|---:|---:|---:|
| mean of wave metrics | 0.8921 | 0.5591 | 0.4996 | 0.2967 |
| pooled held-out predictions | 0.8507 | 0.5647 | 0.4754 | 0.2872 |

## Per-wave results

| wave | n_test | positive_rate | log_loss | roc_auc | pr_auc | brier |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 0.5550 | 1.1895 | 0.4857 | 0.5505 | 0.3739 |
| 2 | 608 | 0.3734 | 0.7573 | 0.5947 | 0.4337 | 0.2683 |
| 3 | 200 | 0.3350 | 0.8381 | 0.5525 | 0.4331 | 0.2749 |
| 4 | 648 | 0.4290 | 0.7221 | 0.5944 | 0.5122 | 0.2564 |
| 10 | 162 | 0.4877 | 0.8701 | 0.6157 | 0.5801 | 0.2837 |
| 11 | 882 | 0.4036 | 0.7881 | 0.6046 | 0.5121 | 0.2667 |
| 15 | 684 | 0.4518 | 1.0082 | 0.5193 | 0.4586 | 0.3283 |
| 16 | 96 | 0.4583 | 0.9804 | 0.5079 | 0.5477 | 0.3148 |
| 17 | 280 | 0.4321 | 0.8746 | 0.5567 | 0.4685 | 0.3034 |

## Interpretation rule

This milestone establishes a directional probability baseline only.
Do not interpret these results as reciprocal-match performance.
Held-out probabilities are saved for later calibration and reciprocal-scoring milestones.
