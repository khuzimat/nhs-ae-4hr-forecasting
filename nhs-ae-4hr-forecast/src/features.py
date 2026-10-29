"""Build the modelling table: features at month t, target at month t+1."""
import numpy as np
import pandas as pd

MIN_T1_ATTENDANCES = 1  # Type 1 providers only (major A&E departments)

FEATURES_NUM = ["perf", "log_att", "booked_share", "t1_share", "dta12_rate",
                "dta4_12_rate", "adm_rate", "perf_vs_region"]
FEATURES_CAT = ["region"]


def add_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Type 1 four-hour performance (incl. booked appointments) and helpers."""
    required = ["att_t1", "att_booked_t1", "over4_t1", "over4_booked_t1",
                "att_t2", "att_other", "dta_12h_plus", "dta_4_12h", "adm_t1",
                "month", "org_code", "region", "org_name"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns required for feature engineering: {sorted(missing)}")
    d = df[df["att_t1"] > 0].copy()
    denom = d["att_t1"] + d["att_booked_t1"]
    if (denom <= 0).any():
        raise ValueError("Type 1 performance denominator must be positive")
    over4 = d["over4_t1"] + d["over4_booked_t1"]
    if (over4 < 0).any() or (over4 > denom).any():
        raise ValueError("Over-four-hour Type 1 counts must be between zero and denominator; review source definition/data")
    d["perf"] = 1 - over4 / denom
    d["log_att"] = np.log(denom)
    d["booked_share"] = d["att_booked_t1"] / denom
    total_att = d["att_t1"] + d["att_t2"] + d["att_other"]
    d["t1_share"] = d["att_t1"] / total_att
    d["dta12_rate"] = d["dta_12h_plus"] / denom
    d["dta4_12_rate"] = d["dta_4_12h"] / denom
    d["adm_rate"] = d["adm_t1"] / denom
    d["perf_vs_region"] = d["perf"] - d.groupby(["month", "region"])["perf"].transform("mean")
    return d


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Join each trust-month to the SAME trust's following calendar month.

    Only genuinely consecutive months are paired (April and August 2025 are
    missing from the data, so July->Sept is not a one-month step and is skipped).
    """
    d = add_performance(df)
    nxt = d[["org_code", "month", "perf"]].copy()
    nxt["month"] = nxt["month"] - pd.DateOffset(months=1)
    nxt = nxt.rename(columns={"perf": "target"})
    out = d.merge(nxt, on=["org_code", "month"], how="inner")
    out["target_month"] = out["month"] + pd.DateOffset(months=1)
    out["delta"] = out["target"] - out["perf"]
    return out.sort_values(["month", "org_code"]).reset_index(drop=True)
