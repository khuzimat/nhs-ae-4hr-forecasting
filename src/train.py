"""Reproducible walk-forward evaluation for next-month A&E performance.

All models share the same forecast origins and rows. Metrics are reported overall
and by forecast month because multiple trusts in one month are not independent
replications of temporal generalisation. Predictions remain unclipped in the
primary evaluation; out-of-range predictions are reported as a diagnostic.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import FEATURES_CAT, FEATURES_NUM, build_pairs
from src.load_data import load_all, quality_report

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
MIN_TRAIN_STEPS = 3
MODEL_NAMES = ["persistence", "mean_change", "ridge", "random_forest", "grad_boosting"]

def make_model(kind: str):
    """Create a fresh sklearn pipeline; preprocessing is fitted on training data only."""
    pre = ColumnTransformer([
        ("num", StandardScaler(), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ])
    models = {
        "ridge": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                random_state=42, n_jobs=-1),
        "grad_boosting": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                        max_iter=150, random_state=42),
    }
    if kind not in models:
        raise ValueError(f"Unknown model: {kind}")
    return Pipeline([("pre", pre), ("reg", models[kind])])

def walk_forward(pairs: pd.DataFrame) -> pd.DataFrame:
    """Fit on earlier origin months and evaluate each later origin month."""
    steps = sorted(pd.to_datetime(pairs["month"]).unique())
    rows = []
    for origin in steps:
        train = pairs[pairs["month"] < origin]
        if train["month"].nunique() < MIN_TRAIN_STEPS:
            continue
        test = pairs[pairs["month"] == origin]
        if test.empty:
            continue
        x_train = train[FEATURES_NUM + FEATURES_CAT]
        x_test = test[FEATURES_NUM + FEATURES_CAT]
        base = test["perf"].to_numpy()
        predictions = {
            "persistence": base,
            "mean_change": base + train["delta"].mean(),
        }
        for kind in ["ridge", "random_forest", "grad_boosting"]:
            model = make_model(kind)
            model.fit(x_train, train["delta"])
            predictions[kind] = base + model.predict(x_test)
        for model_name, pred in predictions.items():
            rows.append(pd.DataFrame({
                "org_code": test["org_code"].to_numpy(),
                "org_name": test["org_name"].to_numpy(),
                "origin_month": test["month"].to_numpy(),
                "target_month": test["target_month"].to_numpy(),
                "model": model_name,
                "actual": test["target"].to_numpy(),
                "predicted": pred,
                "prediction_outside_0_1": (pred < 0) | (pred > 1),
            }))
    if not rows:
        raise ValueError("No walk-forward forecasts created; check data coverage and MIN_TRAIN_STEPS")
    return pd.concat(rows, ignore_index=True)

def _metrics(g: pd.DataFrame) -> pd.Series:
    error = g["predicted"] - g["actual"]
    return pd.Series({
        "MAE": mean_absolute_error(g["actual"], g["predicted"]),
        "RMSE": mean_squared_error(g["actual"], g["predicted"]) ** 0.5,
        "n_predictions": len(g),
        "n_forecast_months": g["target_month"].nunique(),
        "out_of_range_predictions": int(g["prediction_outside_0_1"].sum()),
    })

def summarise(res: pd.DataFrame) -> pd.DataFrame:
    """Overall metrics; same forecast rows are used for every model."""
    summary = res.groupby("model", sort=False).apply(_metrics, include_groups=False)
    summary["MAE_pct_points"] = summary["MAE"] * 100
    summary["RMSE_pct_points"] = summary["RMSE"] * 100
    baseline_mae = summary.loc["persistence", "MAE"]
    summary["MAE_change_vs_persistence_pct"] = (summary["MAE"] / baseline_mae - 1) * 100
    return summary.sort_values("MAE")

def summarise_by_month(res: pd.DataFrame) -> pd.DataFrame:
    """Month-specific errors, making temporal instability visible."""
    return (res.groupby(["target_month", "model"], as_index=False)
            .apply(lambda g: pd.Series({
                "MAE": mean_absolute_error(g["actual"], g["predicted"]),
                "RMSE": mean_squared_error(g["actual"], g["predicted"]) ** 0.5,
                "n_predictions": len(g),
            }), include_groups=False))

def national_shift_diagnostic(pairs: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """Hindsight diagnostic for shared month-to-month shifts (not deployable).

    Uses the realised mean change across providers in each target month. Since this
    value is only known after the target month, it must not be treated as a model.
    """
    month_shift = (pairs.groupby("target_month", as_index=False)["delta"]
                   .agg(n_providers="size", realised_mean_change="mean"))
    persistence = res[res["model"] == "persistence"].copy()
    persistence = persistence.merge(month_shift, on="target_month", how="left", validate="many_to_one")
    persistence["abs_error_persistence"] = (persistence["predicted"] - persistence["actual"]).abs()
    persistence["abs_error_after_realised_month_shift"] = (
        persistence["predicted"] + persistence["realised_mean_change"] - persistence["actual"]
    ).abs()
    return (persistence.groupby("target_month", as_index=False)
            .agg(n_predictions=("org_code", "size"),
                 realised_mean_change=("realised_mean_change", "first"),
                 persistence_MAE=("abs_error_persistence", "mean"),
                 hindsight_shift_adjusted_MAE=("abs_error_after_realised_month_shift", "mean")))

def make_charts(res: pd.DataFrame, summary: pd.DataFrame, by_month: pd.DataFrame) -> None:
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    s = summary["MAE_pct_points"].sort_values()
    ax.barh(s.index[::-1], s.values[::-1])
    for y, value in enumerate(s.values[::-1]):
        ax.text(value + 0.025, y, f"{value:.2f}", va="center", fontsize=9)
    ax.set_xlabel("Mean absolute error (percentage points)")
    ax.set_title("Next-month Type 1 four-hour performance: model comparison")
    fig.tight_layout(); fig.savefig(OUT / "model_comparison.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for model in MODEL_NAMES:
        subset = by_month[by_month["model"] == model].sort_values("target_month")
        ax.plot(subset["target_month"], subset["MAE"] * 100, marker="o", label=model)
    ax.set_ylabel("MAE (percentage points)"); ax.set_title("Forecast error by target month")
    ax.legend(ncol=2, fontsize=8); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(OUT / "error_by_month.png", dpi=160); plt.close(fig)

    model = summary.drop(index=["persistence", "mean_change"], errors="ignore")["MAE"].idxmin()
    subset = res[res["model"] == model]
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.scatter(subset["actual"] * 100, subset["predicted"] * 100, s=10, alpha=.35)
    ax.plot([0, 100], [0, 100], color="grey", linewidth=1)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("Actual next-month performance (%)"); ax.set_ylabel("Predicted (%)")
    ax.set_title(f"Predicted vs actual: {model}")
    fig.tight_layout(); fig.savefig(OUT / "predicted_vs_actual.png", dpi=160); plt.close(fig)

def write_report(data: pd.DataFrame, pairs: pd.DataFrame, res: pd.DataFrame, summary: pd.DataFrame, by_month: pd.DataFrame, shift_diagnostic: pd.DataFrame) -> None:
    first = summary.iloc[0]
    baseline = summary.loc["persistence"]
    best_ml = summary.drop(index=["persistence", "mean_change"], errors="ignore").iloc[0]
    change = best_ml["MAE_change_vs_persistence_pct"]
    months = sorted(data["month"].dt.strftime("%Y-%m").unique())
    persistence_month_mae = shift_diagnostic["persistence_MAE"].mean() * 100
    hindsight_month_mae = shift_diagnostic["hindsight_shift_adjusted_MAE"].mean() * 100
    month_list = ", ".join(months)
    report = f"""# NHS A&E forecasting: reproducible results report

## Question
Can this month's provider-level data predict the same provider's Type 1 four-hour performance in the next calendar month?

## Dataset and quality checks
- Provider-month rows after national TOTAL rows were excluded: **{len(data):,}**.
- Months present: **{data['month'].nunique()}** ({month_list}).
- Trust-month modelling pairs: **{len(pairs):,}**. Only consecutive calendar months are paired.
- Missing cells in the loaded table: **{int(data.isna().sum().sum())}**. No missing values were imputed.
- Exact duplicate rows in the loaded table: **{int(data.duplicated().sum())}**. No duplicates were silently removed.
- Data quality details: see `data_quality_report.csv`.
- Official source: [NHS England Digital — A&E Attendances and Emergency Admissions](https://digital.nhs.uk/data-and-information/publications/statistical/nhse-ae-attendances-and-emergency-admissions).

## Evaluation design
An expanding-window walk-forward evaluation trains only on origin months earlier than the month being forecast. All models use identical test rows. The models predict the change from current performance; persistence predicts no change. Metrics are shown overall and by target month.

## Results
| Model | MAE (percentage points) | RMSE (percentage points) | Predictions | MAE change vs persistence |
|---|---:|---:|---:|---:|
"""
    for name, row in summary.iterrows():
        report += f"| {name} | {row['MAE_pct_points']:.3f} | {row['RMSE_pct_points']:.3f} | {int(row['n_predictions'])} | {row['MAE_change_vs_persistence_pct']:+.2f}% |\n"
    report += f"""

The lowest overall MAE in this run is **{summary.index[0]}** ({first['MAE_pct_points']:.3f} percentage points). The best machine-learning model is **{best_ml.name}**, with an MAE change of **{change:+.2f}%** relative to persistence (negative means better; positive means worse). The persistence baseline MAE is {baseline['MAE_pct_points']:.3f} percentage points.

![Model comparison](model_comparison.png)

![Error by target month](error_by_month.png)

![Predicted versus actual](predicted_vs_actual.png)

## Interpretation
The baseline comparison is essential: a more complex model is useful only if it improves on a credible simple forecast. If persistence wins, current performance is a strong short-term predictor and the extra features/models did not add enough signal in this dataset. Check `metrics_by_month.csv` before drawing conclusions; a pooled metric can hide month-to-month variation.

### Why might machine learning underperform?
Persistence already captures continuity in trust-level performance. This short, gappy dataset gives models few time steps from which to learn changing patterns, and the engineered features may add little signal beyond current performance. National conditions can also shift many trusts together. `monthly_change_diagnostic.csv` compares persistence with a hindsight-only adjustment using the realised average change across providers in each target month. The average month-specific persistence MAE was {persistence_month_mae:.3f} percentage points; after this adjustment it was {hindsight_month_mae:.3f} points. This is **not a valid forecast** because the realised month-level change is unknown at forecast time; it only diagnoses how much common movement contributed to error.

## Limitations and assumptions
- The available dataset covers only {data['month'].nunique()} months and has calendar gaps; only {by_month['target_month'].nunique()} forecast months are evaluated.
- Trusts within the same target month share national conditions, so 605 trust-level predictions (if applicable) are not equivalent to hundreds of independent temporal tests.
- The performance measure is calculated from the selected Type 1 attendance and booked-appointment columns. It should be reconciled with the relevant NHS England definition and headline figures before being described as an official measure.
- The feature `perf_vs_region` uses contemporaneous region-level data. This is available only if the corresponding month's provider data would have been published at the forecast origin; publication timing must be checked for any operational use.
- Forecasts are evaluated as raw model outputs. Predictions outside [0, 1] are counted in the output diagnostics rather than silently clipped.
- This is retrospective research, not a validated operational, clinical or capacity-planning tool. It does not establish causality or prove that acting on a forecast improves patient outcomes.

## Reproducibility
Run `python -m src.train` from the project root. The command regenerates the quality report, predictions, overall and month-level metrics, the hindsight diagnostic, charts and this report. Run `python -m pytest -q` to execute the test suite.
"""
    (OUT / "portfolio_report.md").write_text(report, encoding="utf-8")

def main() -> dict:
    data = load_all()
    pairs = build_pairs(data)
    res = walk_forward(pairs)
    summary = summarise(res)
    by_month = summarise_by_month(res)
    shift_diagnostic = national_shift_diagnostic(pairs, res)
    OUT.mkdir(exist_ok=True)
    quality_report(data).to_csv(OUT / "data_quality_report.csv", index=False)
    summary.to_csv(OUT / "model_comparison.csv")
    by_month.to_csv(OUT / "metrics_by_month.csv", index=False)
    shift_diagnostic.to_csv(OUT / "monthly_change_diagnostic.csv", index=False)
    res.to_csv(OUT / "predictions.csv", index=False)
    make_charts(res, summary, by_month)
    write_report(data, pairs, res, summary, by_month, shift_diagnostic)
    print(summary[["MAE_pct_points", "RMSE_pct_points", "n_predictions", "n_forecast_months", "MAE_change_vs_persistence_pct"]].round(3))
    print(f"\nWrote reproducible outputs to: {OUT}")
    return {"summary": summary, "results": res, "pairs": pairs, "metrics_by_month": by_month, "shift_diagnostic": shift_diagnostic}

if __name__ == "__main__":
    main()
