"""Load NHS England 'Monthly A&E Attendances and Emergency Admissions' CSVs.

Each monthly file has one row per provider (NHS trust or other A&E provider)
plus a national TOTAL row. This module reads every file in data/raw/, gives
the columns short readable names, drops the TOTAL rows and returns one tidy
table with a `month` column.
"""

from pathlib import Path

import pandas as pd

QUALITY_DIR = Path(__file__).resolve().parent.parent / "outputs"

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

COLUMN_MAP = {
    "Period": "period",
    "Org Code": "org_code",
    "Parent Org": "region",
    "Org name": "org_name",
    "A&E attendances Type 1": "att_t1",
    "A&E attendances Type 2": "att_t2",
    "A&E attendances Other A&E Department": "att_other",
    "A&E attendances Booked Appointments Type 1": "att_booked_t1",
    "A&E attendances Booked Appointments Type 2": "att_booked_t2",
    "A&E attendances Booked Appointments Other Department": "att_booked_other",
    "Attendances over 4hrs Type 1": "over4_t1",
    "Attendances over 4hrs Type 2": "over4_t2",
    "Attendances over 4hrs Other Department": "over4_other",
    "Attendances over 4hrs Booked Appointments Type 1": "over4_booked_t1",
    "Attendances over 4hrs Booked Appointments Type 2": "over4_booked_t2",
    "Attendances over 4hrs Booked Appointments Other Department": "over4_booked_other",
    "Patients who have waited 4-12 hs from DTA to admission": "dta_4_12h",
    "Patients who have waited 12+ hrs from DTA to admission": "dta_12h_plus",
    "Emergency admissions via A&E - Type 1": "adm_t1",
    "Emergency admissions via A&E - Type 2": "adm_t2",
    "Emergency admissions via A&E - Other A&E department": "adm_other_ae",
    "Other emergency admissions": "adm_other_emerg",
}
NUMERIC_COLS = [c for c in COLUMN_MAP.values() if c not in
                ("period", "org_code", "region", "org_name")]

MONTHS = {m: i for i, m in enumerate(
    ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST",
     "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"], start=1)}


def parse_period(period: str) -> pd.Timestamp:
    """'MSitAE-SEPTEMBER-2025' -> Timestamp('2025-09-01')."""
    _, month_name, year = period.strip().split("-")
    return pd.Timestamp(year=int(year), month=MONTHS[month_name.upper()], day=1)


def load_month(path: Path) -> pd.DataFrame:
    """Read one monthly file and return provider rows only (no TOTAL row)."""
    df = pd.read_csv(path)
    missing = set(COLUMN_MAP) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing expected columns {sorted(missing)}")
    df = df.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())]

    # The national TOTAL row is labelled in the Period column in most files but
    # in the Org Code column in others (e.g. Feb 2026), so check both.
    is_total = (df["period"].str.upper() == "TOTAL") | (df["org_code"].str.upper() == "TOTAL")
    df = df.loc[~is_total].copy()

    periods = df["period"].unique()
    if len(periods) != 1:
        raise ValueError(f"{path.name}: expected one period, found {list(periods)}")
    df["month"] = parse_period(periods[0])

    df["region"] = df["region"].str.strip().str.title()
    df["org_name"] = df["org_name"].str.strip()
    df["org_code"] = df["org_code"].str.strip()
    # Coercion is deliberately strict: malformed numbers should fail visibly,
    # not silently become missing values or be replaced with zero.
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df.drop(columns="period")


def load_all(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load every CSV in raw_dir into one table, sorted by month then org."""
    files = sorted(Path(raw_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    df = pd.concat([load_month(f) for f in files], ignore_index=True)
    if df["org_code"].isna().any() or df["org_code"].astype(str).str.strip().eq("").any():
        raise ValueError("Missing or blank organisation codes found")
    if df["month"].isna().any():
        raise ValueError("Missing month values found")
    duplicates = df.duplicated(["org_code", "month"], keep=False)
    if duplicates.any():
        examples = df.loc[duplicates, ["org_code", "month"]].head(10).to_dict("records")
        raise ValueError(f"Duplicate (org_code, month) rows found; examples: {examples}")
    return df.sort_values(["month", "org_code"]).reset_index(drop=True)


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return transparent data-quality checks without modifying source data.

    Missingness is reported rather than imputed. Negative counts and over-four-hour
    counts greater than their corresponding denominator are flagged for review.
    """
    checks = []
    def add(check, count, detail):
        checks.append({"check": check, "count": int(count), "detail": detail})

    add("rows", len(df), "Provider-month rows after TOTAL rows are excluded")
    add("columns", len(df.columns), "Columns in the tidy table")
    add("missing_cells", df.isna().sum().sum(), "Missing cells; not imputed")
    add("exact_duplicate_rows", df.duplicated().sum(), "Exact duplicate rows; not removed automatically")
    add("duplicate_org_month", df.duplicated(["org_code", "month"]).sum(), "Must be zero; loader raises if non-zero")
    numeric = df.select_dtypes(include="number")
    negative_count = int((numeric < 0).sum().sum())
    add("negative_numeric_cells", negative_count, "Count-like numeric fields should be reviewed if negative")
    denominator = df["att_t1"] + df["att_booked_t1"]
    over4 = df["over4_t1"] + df["over4_booked_t1"]
    invalid_denom = int((denominator <= 0).sum())
    add("nonpositive_type1_denominator", invalid_denom, "Rows excluded from Type 1 performance calculation")
    add("over4_exceeds_denominator", int((over4 > denominator).sum()), "Potential definition/data issue; review against source")
    add("months_present", df["month"].nunique(), ", ".join(df["month"].drop_duplicates().sort_values().dt.strftime("%Y-%m")))
    return pd.DataFrame(checks)


if __name__ == "__main__":
    data = load_all()
    print(data.groupby("month").size().rename("providers"))
    QUALITY_DIR.mkdir(exist_ok=True)
    quality_report(data).to_csv(QUALITY_DIR / "data_quality_report.csv", index=False)
