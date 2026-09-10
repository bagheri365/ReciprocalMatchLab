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
