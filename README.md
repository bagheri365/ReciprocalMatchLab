# ReciprocalMatchLab

ReciprocalMatchLab studies reciprocal recommendation in two-sided matching systems using the Columbia Speed Dating dataset.

The empirical study focuses on:

- leakage-safe directional preference prediction
- calibrated reciprocal scoring
- direct symmetric joint-match prediction
- leave-one-wave-out evaluation
- pair-level marketplace allocation
- exposure concentration under explicit static-replay assumptions

## Milestone 1

Milestone 1 establishes the scientific contract and validates the raw dataset before any model is trained.

It includes:

- repository/package scaffold
- experiment contract
- leakage policy
- protocol-group configuration
- sensitive-feature policy
- raw-data audit CLI
- pair-level consistency checks
- tests for the dataset invariants

## Data

Place the Columbia Speed Dating CSV at:

```text
data/raw/Speed Dating Data.csv
```

The raw dataset is intentionally not committed.

## Run the audit

```bash
python scripts/audit_data.py --input "data/raw/Speed Dating Data.csv"
```

The command writes:

```text
reports/data_audit.md
reports/tables/wave_summary.csv
```

## Tests

```bash
pytest
```

## Scientific rule

No predictive model should be added until the Phase 0 audit passes and the choices in `configs/experiment_contract.yaml` are reviewed.


## Milestone 2

Milestone 2 freezes the protocol strata and feature-safety taxonomy before any predictive modeling.

Run:

```bash
python scripts/audit_features.py --input "data/raw/Speed Dating Data.csv"
```

This writes:

```text
reports/feature_audit.md
reports/tables/primary_feature_missingness.csv
reports/tables/primary_feature_missingness_by_wave.csv
```

The primary protocol group is fixed from the source data dictionary, not from model performance. Waves with different preference scales or experimental variations remain sensitivity/OOD analyses.


## Milestone 3

Milestone 3 freezes missingness decisions and introduces the leakage-safe leave-one-wave-out preprocessing pipeline.

Primary-model exclusions driven by the pre-modeling audit:

```text
expnum
attr4_1, sinc4_1, intel4_1, fun4_1, amb4_1, shar4_1
attr5_1, sinc5_1, intel5_1, fun5_1, amb5_1
```

These blocks are absent in entire primary waves and are retained for sensitivity analysis only.

`int_corr` remains in the primary feature set with training-fold median imputation and a missingness indicator.


## Milestone 4

Milestone 4 adds the first empirical model: a fixed-hyperparameter directional logistic-regression baseline evaluated with leave-one-wave-out folds.

Run:

```bash
python scripts/run_directional_baseline.py --input "data/raw/Speed Dating Data.csv"
```

Outputs:

```text
reports/tables/directional_logistic_per_wave.csv
reports/tables/directional_logistic_predictions.csv
reports/directional_logistic_baseline.md
```

The held-out `p_like` predictions are intentionally preserved for later calibration and reciprocal-scoring milestones.

## Milestone 5

Milestone 5 adds training-only Platt calibration plus requester-only, candidate-only, product, and minimum reciprocal scores. Calibration never uses the outer held-out wave; because it is fit on the base model's training scores, it is used for score construction rather than definitive probability-calibration claims.

```bash
python scripts/run_reciprocal_scoring.py --input "data/raw/Speed Dating Data.csv"
```

## Milestone 6

Milestone 6 adds a direct symmetric joint-match model trained on unique undirected pairs and compares it against requester-only, candidate-only, product, and minimum reciprocal scoring.

Run after Milestone 5 outputs exist:

```bash
python scripts/run_direct_joint_match.py \
  --input "data/raw/Speed Dating Data.csv" \
  --reciprocal-predictions "reports/tables/reciprocal_scores_predictions.csv"
```

The report includes paired per-wave differences and wins/ties/losses.

## Milestone 7

Milestone 7 closes the probabilistic part of RQ2 without adding a new model. It compares the independence-based calibrated product with the direct joint match probability on unique undirected pairs using held-out log loss, Brier score, fixed-width reliability tables, and paired wave-level differences.

Run after Milestone 6 outputs exist:

```bash
python scripts/evaluate_match_probabilities.py \
  --predictions "reports/tables/direct_joint_match_predictions.csv"
```

Outputs:

```text
reports/tables/match_probability_per_wave.csv
reports/tables/match_probability_reliability.csv
reports/tables/match_probability_paired_differences.csv
reports/match_probability_calibration.md
```


## Milestone 8

Milestone 8 adds pair-level marketplace allocation with batch MILP optimization and an exposure-concentration penalty.

```bash
python scripts/run_marketplace_allocation.py \
  --predictions "reports/tables/direct_joint_match_predictions.csv"
```

The experiment reports PairMatchRate, SuccessfulUserRate, coverage, raw exposure Gini, relative-exposure Gini, and paired trade-offs versus the unpenalized allocator. Relative exposure is normalized to the analytic uniform-random eligible-edge expectation and is not labeled a fairness metric.
