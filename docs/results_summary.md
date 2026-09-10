# ReciprocalMatchLab — Results Summary

This document consolidates the three research questions without repeating implementation detail from the codebase.

## RQ1 — Directional preference generalization

**Question:** How well do leakage-safe preference models generalize directional attraction to unseen speed-dating events?

**Design:** Leave-One-Wave-Out evaluation using pre-interaction features only. Every learned preprocessing transform is fit inside the training fold.

**Finding:** Logistic-regression ROC-AUC varies roughly from 0.49 to 0.62 across held-out primary waves. Directional attraction is learnable only weakly to moderately across events, and event-to-event heterogeneity is substantial.

**Implication:** Reciprocal scoring is operating on noisy marginal estimates. Multiplying two such estimates can compound error rather than improve ranking.

## RQ2 — Reciprocal construction vs direct joint modeling

**Question:** For mutual-match ranking, do calibrated reciprocal constructions outperform one-sided ranking, and does direct joint modeling add predictive value?

### Ranking

Mean held-out metrics:

| Policy | K | MutualOutcomeYield@K | NDCG(match)@K |
|---|---:|---:|---:|
| Candidate-only | 1 | 0.2211 | 0.2653 |
| Minimum | 1 | 0.2069 | 0.2525 |
| Product | 1 | 0.1646 | 0.1950 |
| Requester-only | 1 | 0.1780 | 0.2147 |
| Direct joint | 1 | **0.2269** | **0.2762** |
| Candidate-only | 3 | 0.1846 | 0.2882 |
| Minimum | 3 | **0.2072** | **0.2976** |
| Product | 3 | 0.1989 | 0.2794 |
| Requester-only | 3 | 0.2043 | 0.2759 |
| Direct joint | 3 | 0.1877 | 0.2938 |
| Candidate-only | 5 | **0.2152** | **0.3939** |
| Minimum | 5 | 0.1947 | 0.3567 |
| Product | 5 | 0.2009 | 0.3512 |
| Requester-only | 5 | 0.1806 | 0.3135 |
| Direct joint | 5 | 0.1895 | 0.3665 |

At K=1, direct joint vs candidate-only has mean Δyield +0.0057 and a 5/1/3 win/tie/loss record. At K=5, direct joint loses to candidate-only on mean yield by 0.0258 and wins only 2 of 9 waves.

### Match-probability quality

Evaluation is on unique undirected held-out pairs.

| Model | Log loss ↓ | Brier ↓ |
|---|---:|---:|
| Direct joint | **0.5645** | **0.1667** |
| Calibrated product | 0.6134 | 0.1760 |

Direct joint wins 6/9 waves on log loss and 5/9 on Brier.

**Conclusion:** Direct joint modeling gives better held-out match probabilities and the best K=1 ranking result, but it does not uniformly dominate simpler policies at larger K. The calibrated product is not a strong baseline in this setting.

**Non-claim:** These comparisons do not identify the conditional dependence structure of the two participants' decisions.

## RQ3 — Match yield vs exposure concentration

**Question:** What pair-level match-yield trade-off results from relative-exposure control under a fully specified allocation rule?

Allocation uses unique undirected pairs, per-user capacity constraints, a fixed pair budget, and a batch optimization objective. Relative exposure is normalized to the analytic uniform-random eligible-edge expectation.

| λ | PairMatchRate | SuccessfulUserRate | Coverage | Raw exposure Gini | Relative exposure Gini |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.1707 | 0.2437 | 0.7128 | 0.3884 | 0.3934 |
| 0.01 | 0.1707 | 0.2437 | 0.7221 | 0.3852 | 0.3901 |
| 0.05 | 0.1669 | 0.2471 | 0.8183 | 0.3480 | 0.3464 |
| 0.10 | 0.1622 | 0.2342 | 0.9136 | 0.2969 | 0.2905 |
| 0.20 | 0.1621 | 0.2379 | 0.9944 | 0.2159 | 0.2136 |
| 0.50 | 0.1584 | 0.2403 | 1.0000 | 0.1376 | 0.1396 |

At λ=0.20, relative-exposure Gini decreases by about 46% versus λ=0 while PairMatchRate decreases by about 5% relative. Concentration improves in all 9 primary waves.

A notable negative result is that the uniform-random eligible-edge baseline can outperform the score-maximizing allocator on historical pair-match rate. This reinforces that prediction quality and constrained marketplace policy quality are different objectives.

## Experimental guardrails

- primary unit of inference: wave
- primary split: Leave-One-Wave-Out
- primary ranking features: pre-interaction only
- pair metrics: unique undirected pairs
- calibration: no outer held-out wave access
- preprocessing: training-fold only
- static replay: explicitly non-causal
- exposure concentration: not labeled fairness

## Overall conclusion

Reciprocal recommendation is not solved by multiplying two marginal preference estimates. Direct joint modeling improves held-out probability quality and very-top-of-list ranking, but the strongest ranking policy depends on slate depth. Once the problem is formulated as pair-level marketplace allocation, a separate exposure-versus-yield frontier appears, and substantial concentration reduction is possible with a comparatively modest static-replay yield cost.
