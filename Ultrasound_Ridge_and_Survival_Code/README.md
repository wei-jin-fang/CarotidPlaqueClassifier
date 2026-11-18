# Ultrasound ridge logistic regression and outpatient survival analysis

R analysis scripts for the 2,618-patient outpatient cohort. The frozen CTA
probability is supplied as a baseline input; these scripts do not train the CT network.

## Environment

Original analysis: R 4.3.3, survival 3.7-0, glmnet 4.1-8, Matrix 1.6-5,
ggplot2 3.5.1, and ragg 1.3.2.

## Run order

Use a private working directory outside your Git checkout. Place the approved,
analysis-ready input there as `analysis_data_private.csv`. Use an absolute path:

```sh
Rscript reanalyze_outpatients_current.R /path/to/private_run
Rscript outpatient_sensitivity_current.R /path/to/private_run
Rscript outpatient_km_ci_current.R /path/to/private_run
```

- `reanalyze_outpatients_current.R`: fits IPCW-weighted ridge logistic models
  at 12, 24 and 36 months, using grouped five-fold outer evaluation and
  three-fold inner tuning. Also fits cross-fitted combinations and produces
  time-dependent discrimination, calibration, decision curves, Kaplan-Meier
  and Cox results, including time-interval effects and PH diagnostics.
- `outpatient_sensitivity_current.R`: supplements the primary analysis with
  alternative Cox specifications, landmark and matching analyses, and plots.
  Some of these exploratory outputs were not retained in the manuscript.
- `outpatient_km_ci_current.R`: renders KM plots with log-log confidence
  intervals, censor marks and numbers at risk. Run it after the sensitivity
  script, as its optional matching figures read files generated there.

## Input fields

One row per eligible, de-identified outpatient; the input represents the
final analysis cohort after eligibility and development-overlap exclusions.
Do not substitute a table in which early losses to follow-up are classified
as known event-free at 36 months.

| Group | Fields and interpretation |
| --- | --- |
| Record bookkeeping | `patient_id` (pseudonymous and unique), `source` (source-cohort label) |
| Follow-up | `time_months` (first endpoint or last confirmed event-free observation, through 36 months), `event` (1 = endpoint, 0 = right censored) |
| Frozen CT output | `ai_probability` (0 to 1), `ai_binary` (1 if probability >= 0.50), `ai10` (probability divided by 0.10) |
| Baseline clinical variables | `age`, `ldl_c`, and binary `male`, `smoking`, `hypertension`, `diabetes`, `coronary_disease` |
| Treatment and strata | Binary `antiplatelet`, `lipid_lowering`; `stenosis3` (lt50, 50to69, 70to99); `stenosis4` (lt30, 30to49, 50to69, 70to99) |
| Numeric ultrasound inputs | `cca_psv`, `ica_psv`, `thickness`, `length`, `gsm`, `jba` |
| Categorical ultrasound inputs | `stenosis`, `echo`, `homogeneity`, `surface`, `ulcer`, `calcification`; use the study coding in Supplementary Table 5 |

For exact replication, preserve the finalized input row order and category
coding. The primary script writes aggregate tables and private patient-level
predictions. RDS intermediates also contain row-level data. Do not publish
private working files, predictions, matching outputs, or spreadsheets.
No patient data, predictions, or intermediate analysis objects are included.
