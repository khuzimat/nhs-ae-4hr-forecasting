# NHS A&E 4-Hour Performance Forecasting

A reproducible portfolio project exploring whether monthly provider-level data can predict the next month's Type 1 A&E four-hour performance. The project prioritises a fair baseline comparison, transparent data-quality checks and time-aware evaluation over using complex models for their own sake.

> **Research prototype only.** This analysis is not an NHS-approved operational or clinical tool and should not be used for staffing, capacity or patient-care decisions.

## Research question

Can this month's data predict the same provider's Type 1 four-hour performance in the next calendar month, and do regression models improve on a simple persistence forecast?

## What is included

- `src/load_data.py` — load and validate provider-month CSVs; create a diagnostic quality report without automatic imputation or duplicate deletion.
- `src/features.py` — calculate the project-defined Type 1 performance measure and build only consecutive-month pairs.
- `src/train.py` — expanding-window walk-forward evaluation, shared test rows, baselines, ML models, metrics and plots.
- `tests/` — checks for data integrity, formula consistency, invalid counts, consecutive-month pairing and temporal evaluation.
- `outputs/` — generated metrics, predictions, quality checks, charts and a portfolio report.
- `data/raw/` — input CSVs included in the supplied archive. Check the original source's licence/reuse terms before redistributing them.

## Data and unit of analysis

The included files cover ten months: May, June, July, September, October, November and December 2025, then January, February and March 2026. April and August 2025 are absent. The unit is a provider/trust-month; the national `TOTAL` row is excluded. The Type 1 analysis is restricted to rows with positive Type 1 attendances.

**Source:** [NHS England Digital — A&E Attendances and Emergency Admissions](https://digital.nhs.uk/data-and-information/publications/statistical/nhse-ae-attendances-and-emergency-admissions), monthly provider-level official statistics. Before publishing or redistributing the raw files, confirm the current metadata, definitions and reuse terms; this repository does not claim that its derived measure exactly matches every NHS England headline statistic.

### Project-defined outcome

`performance = 1 - (over4_t1 + over4_booked_t1) / (att_t1 + att_booked_t1)`

This is a project calculation using the named CSV fields. The denominator includes Type 1 attendances and booked Type 1 appointments; the numerator includes the corresponding over-four-hour counts. Validate this definition against the relevant NHS England metadata before calling it an official performance measure. The pipeline raises an error if over-four-hour counts exceed the denominator rather than silently clipping the result.

## Data quality principles

The pipeline checks expected columns, period parsing, organisation-month uniqueness, missing cells, exact duplicates, negative numeric values, non-positive denominators and over-four-hour counts that exceed their denominator. It does **not** automatically replace missing values with zero, drop outliers, or delete duplicate rows. The quality report is diagnostic; any proposed correction should be justified against source metadata and documented.

## Forecast design

Features at month *t* are used to predict the same provider's performance at *t + 1*. Only genuinely consecutive calendar months are paired; a July record is not paired directly with September because August is missing.

Models predict change from the current month's performance:

1. **Persistence baseline:** predicts no change.
2. **Mean-change baseline:** adds the average change seen in earlier training steps.
3. **Ridge regression**.
4. **Random Forest regression**.
5. **Histogram-based Gradient Boosting regression**.

The expanding-window evaluation trains on origin months strictly earlier than the current test origin. Preprocessing is fitted inside each training fold. All models use the same test records, and results are reported both overall and by forecast month. Raw predictions are evaluated without clipping; outputs flag predictions outside the valid [0, 1] range.

## Reproduce the analysis

Use Python 3.11 or a compatible Python version supported by the pinned dependencies. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.train
python -m pytest -q
```

`python -m src.train` regenerates `outputs/data_quality_report.csv`, `model_comparison.csv`, `metrics_by_month.csv`, `predictions.csv`, three PNG charts, `monthly_change_diagnostic.csv` and `portfolio_report.md`. It does not alter the raw input CSVs.

## How to read the results

MAE and RMSE are calculated on the same forecast rows for every model. They are expressed as proportions in the CSV and as percentage points in the report. A negative `MAE_change_vs_persistence_pct` means the model's MAE is lower than persistence; a positive value means it is worse. Always inspect month-level metrics as well as the pooled average. `monthly_change_diagnostic.csv` includes a hindsight-only diagnostic using the realised average change across providers in each target month; it is not a deployable forecast, because the realised month-level change is not known in advance.

Trusts observed in the same target month share national conditions. Therefore, hundreds of trust-level predictions across a handful of forecast months do not amount to hundreds of independent tests of temporal generalisation. This short, gappy dataset cannot establish stable performance across years or seasons.

## Known limitations and risks

- Only ten observed months are available, with calendar gaps and a small number of evaluated forecast months.
- Seasonal patterns, long-term changes and model stability cannot be assessed reliably from this short period.
- The current month's figures must actually be available at the intended forecast origin. Check NHS England's publication dates; otherwise the operational lead time may be shorter than described.
- `perf_vs_region` uses the contemporaneous region average. It is only available in a real forecast if the full set of current-month records is available at that time.
- The derived outcome should be reconciled with official metadata and headline values before use in external reporting.
- The analysis is observational and retrospective. It does not establish causality, operational utility or improved patient outcomes.

## Next steps

1. Add several years of monthly data and confirm stable definitions across reporting changes.
2. Define a realistic forecast issue date based on publication availability.
3. Retain a final untouched period and report errors by month, region and provider type where appropriate.
4. Compare with additional simple baselines before adding model complexity.
5. If predicting threshold breaches, define the threshold and evaluate calibration, sensitivity, specificity and the cost of false alarms.

## Project summary

“I built a reproducible Python pipeline using published NHS A&E data. I checked data quality, engineered provider-level features, compared three regression models with simple baselines, and used expanding-window validation to avoid training on future months. The key result is whether added model complexity beats persistence—not whether an ML model can be made to run. The main limitation is the short, gappy time series, so the next step is broader historical validation and a realistic publication-time forecast design.”
