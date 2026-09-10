# Production System Design

This document maps ReciprocalMatchLab's offline experiments to a production reciprocal matching system. Components such as ANN retrieval, feature stores, streaming, latency budgets, safety services, and online experiments are **production design extrapolations**; they are not empirically benchmarked by the speed-dating dataset.

## 1. Objective

A one-sided recommender ranks candidates by utility to the requester. A reciprocal marketplace must reason about at least three distinct objectives:

1. **requester preference** — is A likely to want B?
2. **candidate preference** — is B likely to want A?
3. **marketplace value** — how should scarce interaction opportunities be allocated across the graph?

These should not be collapsed into one score prematurely.

## 2. Online serving path

```text
Request from user A
      │
      ▼
Eligibility + safety filters
      │
      ▼
Candidate retrieval
      │
      ▼
Feature assembly
 user A + candidate B + pair/context
      │
      ├───────────────┐
      ▼               ▼
Directional model   Reverse-direction model
 P(A likes B)         P(B likes A)
      │               │
      └───────┬───────┘
              ▼
 Reciprocal / joint ranker
              │
              ▼
 Marketplace policy layer
 capacity, diversity, exposure controls
              │
              ▼
 Final slate + explanations / UI metadata
```

A production implementation may use one shared directional model evaluated in both orientations rather than two separately trained models.

## 3. Candidate generation

The speed-dating dataset has only a small candidate pool per participant, so ReciprocalMatchLab deliberately does not benchmark ANN retrieval.

At production scale, retrieval would typically combine:

- hard eligibility constraints
- geography / distance
- activity / availability
- age or preference constraints where policy permits
- safety / trust filters
- lightweight embedding or collaborative retrieval
- exploration candidates

The goal is high recall under a strict latency budget. Reciprocal scoring belongs after this stage because computing both directions for the entire marketplace may be too expensive.

## 4. Feature architecture

A practical feature layer separates:

### User features

Long-lived or slowly changing attributes, preferences, activity state, and learned embeddings.

### Pair features

Compatibility features computed from A and B, such as distance, shared interests, preference compatibility, or interaction history.

### Context features

Time, session state, device, geography, inventory pressure, and current marketplace conditions.

The offline project enforces an important production rule: **feature availability must match serving-time availability**. Post-interaction variables are forbidden from the primary ranker.

## 5. Directional and joint models

### Directional model

```text
pAB = P(A likes B | A, B, pair, context)
pBA = P(B likes A | B, A, pair, context)
```

A shared model can score both orientations to reduce maintenance complexity and encourage consistent feature semantics.

### Reciprocal constructions

Simple serving-time policies can include:

```text
requester = pAB
candidate = pBA
product   = pAB * pBA
minimum   = min(pAB, pBA)
```

ReciprocalMatchLab shows why these must be evaluated rather than assumed correct: product scoring underperformed one-sided alternatives in the offline study.

### Direct joint model

A second model can estimate:

```text
P(mutual positive outcome | symmetric pair representation)
```

This can be useful when the joint target contains structure that is not captured well by multiplying marginal models. In the project, direct joint improves held-out probability quality and K=1 ranking, but not every ranking depth.

## 6. Marketplace policy layer

Ranking candidates independently for every requester can over-concentrate attention on the same users. A two-sided marketplace therefore benefits from a policy layer after scoring.

Conceptually:

```text
maximize    Σ x_AB * value_AB
            - λ * exposure_concentration

subject to  per-user opportunity limits
            eligibility constraints
            pair budget / inventory constraints
            x_AB ∈ {0,1}
```

ReciprocalMatchLab uses a batch binary optimization over undirected pairs and penalizes squared relative exposure. Production systems may use faster approximations, rolling-window quotas, dual variables, or greedy/online solvers depending on latency and scale.

## 7. Relative exposure

Raw exposure alone can be misleading when users have different numbers of eligible opportunities.

The offline study normalizes exposure by the expected exposure under uniform random selection among eligible edges:

```text
relative_exposure_u = observed_exposure_u / expected_random_exposure_u
```

Interpretation:

```text
1.0  ≈ random-opportunity expectation
2.0  ≈ twice random-opportunity expectation
0.5  ≈ half random-opportunity expectation
```

The project reports both raw and relative exposure concentration. It does not call these metrics fairness without a separate fairness definition and protected-group estimand.

## 8. Logging and offline evaluation

Every served recommendation should log enough context to reconstruct the decision:

```text
request_id
user_id
candidate_ids
retrieval source
model / feature versions
raw directional scores
joint / reciprocal score
policy-adjusted score
rank / position
eligibility decisions
exposure-control state
outcomes and timestamps
```

The offline project uses Leave-One-Wave-Out evaluation because speed-dating events are natural grouped environments. In production, analogous grouped or temporal evaluation should avoid leakage from future interactions or policy-dependent signals.

## 9. Feedback loops and online experimentation

Once deployed, ranking changes what gets exposed, so future labels are policy-dependent. ReciprocalMatchLab's static replay does not solve that causal problem.

A production roadmap would add:

- randomized exploration where acceptable
- propensity logging
- counterfactual / off-policy evaluation
- A/B tests with marketplace-level guardrails
- monitoring for exposure concentration, response rates, blocks/reports, and retention

Those topics are intentionally not the empirical center of this repo because they are covered more directly by dedicated policy-evaluation work.

## 10. Safety and integrity

A dating or matching system requires hard safety constraints outside the ranking objective. Examples include block lists, age/eligibility rules, abuse detection, spam controls, location privacy, and sensitive-feature governance.

These are **filters and policy constraints**, not soft relevance features that a ranker is allowed to override.

## 11. Latency and deployment decomposition

A practical low-latency stack could look like:

```text
Offline / nearline
───────────────
feature computation
embedding training
model training + calibration
policy parameter selection

Online
──────
eligibility filtering
ANN / heuristic retrieval
feature fetch
batched reciprocal scoring
policy adjustment
slate assembly

Streaming
─────────
recent interactions
activity state
exposure counters
safety signals
```

The marketplace allocator used in this repo is a batch research solver. A production replacement would depend on traffic shape: periodic batch allocation, rolling horizon optimization, or an online approximation with per-user shadow prices are all plausible designs.

## 12. Monitoring

Model metrics alone are insufficient. A production dashboard should separate:

### Prediction / ranking

- directional calibration
- joint-match calibration
- ranking yield
- candidate recall

### Marketplace

- exposure Gini / relative-exposure Gini
- coverage
- successful-user rate
- opportunity distribution
- inventory utilization

### System

- p50 / p95 / p99 latency
- feature freshness
- retrieval failures
- model fallbacks

### Safety / quality

- blocks / reports
- spam or abuse rates
- policy-filter volume
- subgroup audits where legally and scientifically appropriate

## 13. Failure modes

| Failure mode | Mitigation |
|---|---|
| Strong requester model, weak reverse model | calibrate/evaluate directions separately; avoid assuming product helps |
| Popularity concentration | marketplace policy layer; exposure budgets; relative-exposure monitoring |
| Cold start | profile/content features, exploration, prior models |
| Sparse mutual-match labels | directional auxiliary objectives; shared representations; careful calibration |
| Feature leakage | serving-time feature contracts and offline assertions |
| Policy feedback loop | exploration + propensity logging + online experiments |
| Solver too slow | approximate online allocation or periodic batch optimization |

## 14. Design takeaway

The key architectural decision is to keep **prediction, reciprocal ranking, and marketplace allocation as separate layers**.

A model can be well calibrated and still induce a poor allocation. A reciprocal heuristic can sound principled and still rank worse than a one-sided score. And an allocation policy can reduce concentration while sacrificing some immediate predicted yield.

Treating those as distinct engineering problems makes the system easier to evaluate, debug, and govern.
