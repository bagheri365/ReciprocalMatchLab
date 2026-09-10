# Direct Symmetric Joint-Match Model

The direct model is trained on unique undirected pairs with symmetric features.

## Mean of per-wave metrics

| policy | K | MutualOutcomeYield@K | NDCG(match)@K |
|---|---:|---:|---:|
| candidate | 1 | 0.2211 | 0.2653 |
| candidate | 3 | 0.1846 | 0.2882 |
| candidate | 5 | 0.2152 | 0.3939 |
| direct_joint | 1 | 0.2269 | 0.2762 |
| direct_joint | 3 | 0.1877 | 0.2938 |
| direct_joint | 5 | 0.1895 | 0.3665 |
| minimum | 1 | 0.2069 | 0.2525 |
| minimum | 3 | 0.2072 | 0.2976 |
| minimum | 5 | 0.1947 | 0.3567 |
| product | 1 | 0.1646 | 0.1950 |
| product | 3 | 0.1989 | 0.2794 |
| product | 5 | 0.2009 | 0.3512 |
| requester | 1 | 0.1780 | 0.2147 |
| requester | 3 | 0.2043 | 0.2759 |
| requester | 5 | 0.1806 | 0.3135 |

## Direct-joint paired differences

| comparison | K | mean Δyield | median Δyield | mean ΔNDCG | wins | ties | losses |
|---|---:|---:|---:|---:|---:|---:|---:|
| candidate | 1 | 0.0057 | 0.0417 | 0.0109 | 5 | 1 | 3 |
| minimum | 1 | 0.0200 | -0.0500 | 0.0237 | 3 | 1 | 5 |
| product | 1 | 0.0623 | 0.0000 | 0.0812 | 4 | 1 | 4 |
| requester | 1 | 0.0489 | 0.0500 | 0.0615 | 5 | 2 | 2 |
| candidate | 3 | 0.0031 | 0.0000 | 0.0056 | 4 | 1 | 4 |
| minimum | 3 | -0.0196 | -0.0278 | -0.0038 | 2 | 0 | 7 |
| product | 3 | -0.0113 | 0.0000 | 0.0144 | 3 | 2 | 4 |
| requester | 3 | -0.0167 | -0.0333 | 0.0179 | 4 | 0 | 5 |
| candidate | 5 | -0.0258 | -0.0143 | -0.0274 | 2 | 1 | 6 |
| minimum | 5 | -0.0053 | -0.0100 | 0.0099 | 3 | 1 | 5 |
| product | 5 | -0.0114 | -0.0083 | 0.0154 | 3 | 1 | 5 |
| requester | 5 | 0.0089 | -0.0100 | 0.0530 | 4 | 0 | 5 |

Positive Δ means direct-joint outperformed the comparison policy.

These are paired wave-level comparisons; they do not establish conditional dependence.
