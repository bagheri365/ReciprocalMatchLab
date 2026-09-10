# Pair-Level Marketplace Allocation

Each selected undirected edge represents a mutual interaction opportunity for both participants.
Historical outcomes are reused under a static replay assumption; this is not a causal policy estimate.

## Random eligible-edge baseline

Mean PairMatchRate: **0.1889**

Mean SuccessfulUserRate: **0.2761**

Mean relative-exposure Gini: **0.3089**

## Optimized exposure trade-off

| lambda | PairMatchRate | SuccessfulUserRate | Coverage | raw Gini | relative Gini | mean score |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.1707 | 0.2437 | 0.7128 | 0.3884 | 0.3934 | 0.3672 |
| 0.01 | 0.1707 | 0.2437 | 0.7221 | 0.3852 | 0.3901 | 0.3671 |
| 0.05 | 0.1669 | 0.2471 | 0.8183 | 0.3480 | 0.3464 | 0.3630 |
| 0.10 | 0.1622 | 0.2342 | 0.9136 | 0.2969 | 0.2905 | 0.3529 |
| 0.20 | 0.1621 | 0.2378 | 0.9944 | 0.2159 | 0.2136 | 0.3334 |
| 0.50 | 0.1584 | 0.2403 | 1.0000 | 0.1376 | 0.1396 | 0.3056 |

## Paired change versus lambda=0

Negative delta relative-exposure Gini means exposure became less concentrated.

| lambda | mean ΔPairMatchRate | median ΔPairMatchRate | mean Δrelative Gini | improved-concentration waves |
|---:|---:|---:|---:|---:|
| 0.01 | 0.0000 | 0.0000 | -0.0033 | 3 |
| 0.05 | -0.0038 | 0.0000 | -0.0470 | 9 |
| 0.10 | -0.0084 | 0.0000 | -0.1029 | 9 |
| 0.20 | -0.0085 | 0.0000 | -0.1798 | 9 |
| 0.50 | -0.0123 | 0.0000 | -0.2537 | 9 |

## Interpretation limits

Relative exposure is normalized by uniform random eligible-edge opportunity, not a fairness definition.
The allocation objective is batch-optimized and order-independent; the random baseline is Monte Carlo.
