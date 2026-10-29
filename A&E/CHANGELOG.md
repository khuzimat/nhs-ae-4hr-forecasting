# Changelog

## Improved portfolio version

- Preserved the original raw CSV inputs and project structure.
- Added a diagnostic data-quality report for missing cells, exact duplicates, trust-month duplicates, negative numeric values, non-positive Type 1 denominators and over-four-hour counts exceeding the denominator.
- Added explicit input validation for feature construction; invalid performance counts raise an error rather than being clipped or silently corrected.
- Expanded the test suite from 5 to 10 tests, including invalid-count handling, formula validation, shared test-row fairness and the hindsight diagnostic.
- Made walk-forward model comparison outputs more transparent with forecast-month metrics, out-of-range prediction flags and MAE change relative to persistence.
- Added a hindsight-only national month-shift diagnostic, explicitly documented as unavailable at forecast time and not a deployable model.
- Regenerated all result tables and plots from the project code.
- Rewrote the README with source link, setup instructions, assumptions, limitations, interpretation guidance and an interview summary.
- Added a generated portfolio report and pinned package versions for reproducibility.
