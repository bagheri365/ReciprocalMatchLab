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
