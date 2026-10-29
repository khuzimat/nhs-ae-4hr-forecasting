"""Tests for data integrity, feature definitions, and temporal evaluation."""
import numpy as np
import pandas as pd
import pytest

from src.features import add_performance, build_pairs
from src.load_data import load_all, parse_period, quality_report
from src.train import national_shift_diagnostic, summarise, summarise_by_month, walk_forward


@pytest.fixture(scope="module")
def data():
    return load_all()


def test_parse_period():
    assert parse_period("MSitAE-SEPTEMBER-2025") == pd.Timestamp("2025-09-01")
    with pytest.raises((ValueError, KeyError)):
        parse_period("not-a-month")


def test_no_total_rows_or_duplicate_trust_months(data):
    assert not (data["org_code"].str.upper() == "TOTAL").any()
    assert not (data["period"].str.upper() == "TOTAL").any() if "period" in data else True
    assert not data.duplicated(["org_code", "month"]).any()
    assert not data.duplicated().any()


def test_loaded_numeric_columns_have_no_missing_or_negative_values(data):
    numeric = data.select_dtypes(include="number")
    assert not numeric.isna().any().any()
    assert not (numeric < 0).any().any()


def test_quality_report_is_diagnostic_and_does_not_mutate_data(data):
    before = data.copy(deep=True)
    report = quality_report(data)
    assert {"check", "count", "detail"}.issubset(report.columns)
    assert report.set_index("check").loc["missing_cells", "count"] == 0
    pd.testing.assert_frame_equal(data, before)


def test_performance_is_proportion_and_matches_documented_formula(data):
    d = add_performance(data)
    expected = 1 - (d["over4_t1"] + d["over4_booked_t1"]) / (d["att_t1"] + d["att_booked_t1"])
    assert np.allclose(d["perf"], expected)
    assert d["perf"].between(0, 1).all()


def test_performance_rejects_invalid_over_four_hour_count(data):
    bad = data.loc[data["att_t1"] > 0].head(1).copy()
    bad["over4_t1"] = bad["att_t1"] + bad["att_booked_t1"] + 1
    with pytest.raises(ValueError, match="Over-four-hour"):
        add_performance(bad)


def test_pairs_only_link_consecutive_calendar_months(data):
    pairs = build_pairs(data)
    gap = (pairs["target_month"].dt.year * 12 + pairs["target_month"].dt.month) - \
          (pairs["month"].dt.year * 12 + pairs["month"].dt.month)
    assert (gap == 1).all()
    assert pd.Timestamp("2025-07-01") not in set(pairs["month"])


def test_walk_forward_is_temporal_and_models_share_test_rows(data):
    pairs = build_pairs(data)
    result = walk_forward(pairs)
    assert (result["target_month"] > result["origin_month"]).all()
    assert (result["origin_month"] < result["target_month"]).all()
    assert set(result["model"]) == {"persistence", "mean_change", "ridge", "random_forest", "grad_boosting"}
    counts = result.groupby("model").size()
    assert counts.nunique() == 1
    assert result.groupby("model")["target_month"].apply(lambda s: tuple(sorted(s.unique()))).nunique() == 1


def test_metrics_include_month_level_and_out_of_range_diagnostics(data):
    result = walk_forward(build_pairs(data))
    summary = summarise(result)
    by_month = summarise_by_month(result)
    assert (summary["MAE"] >= 0).all()
    assert (summary["RMSE"] >= summary["MAE"] - 1e-12).all()
    assert "MAE_change_vs_persistence_pct" in summary
    assert by_month["target_month"].nunique() >= 1
    assert "n_predictions" in by_month


def test_national_shift_diagnostic_is_explicitly_separate_from_forecasts(data):
    pairs = build_pairs(data)
    result = walk_forward(pairs)
    diagnostic = national_shift_diagnostic(pairs, result)
    assert {"persistence_MAE", "hindsight_shift_adjusted_MAE", "realised_mean_change"}.issubset(diagnostic.columns)
    assert diagnostic["target_month"].nunique() == result.loc[result["model"] == "persistence", "target_month"].nunique()
