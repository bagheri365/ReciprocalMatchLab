# Feature and Protocol Audit

## Taxonomy gate

- Dataset columns: **195**
- Primary modeling features: **55**
- Unclassified dataset columns: **0**
- Duplicate taxonomy assignments: **0**
- Taxonomy columns absent from dataset: **0**

## Missingness review

Primary features with max per-wave missingness > 20%: **13**

| feature | overall_missing | max_wave_missing | median_wave_missing | waves_with_any_missing |
|---|---:|---:|---:|---:|
| expnum | 0.785 | 1.000 | 1.000 | 19 |
| amb5_1 | 0.414 | 1.000 | 0.083 | 13 |
| attr5_1 | 0.414 | 1.000 | 0.083 | 13 |
| fun5_1 | 0.414 | 1.000 | 0.083 | 13 |
| intel5_1 | 0.414 | 1.000 | 0.083 | 13 |
| sinc5_1 | 0.414 | 1.000 | 0.083 | 13 |
| shar4_1 | 0.228 | 1.000 | 0.000 | 9 |
| amb4_1 | 0.225 | 1.000 | 0.000 | 8 |
| attr4_1 | 0.225 | 1.000 | 0.000 | 8 |
| fun4_1 | 0.225 | 1.000 | 0.000 | 8 |
| intel4_1 | 0.225 | 1.000 | 0.000 | 8 |
| sinc4_1 | 0.225 | 1.000 | 0.000 | 8 |
| int_corr | 0.019 | 0.222 | 0.000 | 5 |

## Gate

Modeling may begin only if taxonomy coverage is exact and every flagged
primary feature has an explicit keep/drop/imputation decision in the experiment contract.
