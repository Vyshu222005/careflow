"""Rework and loop analysis: which steps are repeated, and which patients come back?

Definitions (on the clinical-steps log, repeated lab tests left out):

    repeat         a step occurs more than once in the same case (extra events beyond the first)
    immediate      the same step directly after itself (A, A)
    loop back      a step reappears after a different step in between (A, B, A)
    return         a Return ER event, measured from the first discharge

Lab tests are looked at separately, because repeating blood tests is routine monitoring and not
rework.

Usage:
    python src/rework.py

Outputs:
    data/processed/sepsis_rework.csv   one row per case
    reports/rework_steps.csv           how often each step is repeated
    reports/rework_summary.csv         the key numbers
"""
from pathlib import Path

import pandas as pd

from conformance import load_log
from discovery import milestone_log
from load_data import ACTIVITY, CASE_ID, TIMESTAMP
from variants import LAB_ACTIVITIES

ROOT = Path(__file__).resolve().parents[1]
REWORK_PATH = ROOT / "data" / "processed" / "sepsis_rework.csv"
STEPS_PATH = ROOT / "reports" / "rework_steps.csv"
SUMMARY_PATH = ROOT / "reports" / "rework_summary.csv"

WARD = "Admission NC"
RETURN = "Return ER"
RETURN_WINDOWS_DAYS = (7, 30, 90)


def clinical_steps(log: pd.DataFrame) -> pd.DataFrame:
    return milestone_log(log)


def rework_per_case(steps: pd.DataFrame) -> pd.DataFrame:
    """Per case: number of repeats, which steps repeat, immediate repeats and loop backs.

    `steps` must be in time order within each case.
    """
    cases = pd.Index(steps[CASE_ID].unique(), name=CASE_ID)
    by_step = steps.groupby([CASE_ID, ACTIVITY])
    occurrence = by_step.cumcount()  # 0 for the first time a step occurs in a case
    immediate = steps[ACTIVITY] == steps.groupby(CASE_ID)[ACTIVITY].shift(1)

    counts = by_step.size()
    repeated = counts[counts > 1].reset_index(name="n")
    table = pd.DataFrame(index=cases)
    table["n_repeats"] = (occurrence > 0).groupby(steps[CASE_ID]).sum().reindex(cases).astype(int)
    table["repeated_steps"] = repeated.groupby(CASE_ID)[ACTIVITY].agg("; ".join).reindex(cases).fillna("")
    table["immediate_repeats"] = immediate.groupby(steps[CASE_ID]).sum().reindex(cases).astype(int)
    # a loop back is a step that has occurred before and does not directly follow itself
    table["loop_backs"] = ((occurrence > 0) & ~immediate).groupby(steps[CASE_ID]).sum().reindex(cases).astype(int)
    table["has_rework"] = table["n_repeats"] > 0
    return table.reset_index()


def step_rework_table(steps: pd.DataFrame) -> pd.DataFrame:
    """For each step: patients who repeat it and the extra events, as share of all patients."""
    counts = steps.groupby([CASE_ID, ACTIVITY]).size().rename("n").reset_index()
    repeated = counts[counts["n"] > 1]
    table = repeated.groupby(ACTIVITY).agg(patients=(CASE_ID, "nunique"), extra_events=("n", lambda s: int((s - 1).sum())))
    table["share_of_patients_pct"] = table["patients"] / steps[CASE_ID].nunique() * 100
    return table.sort_values("patients", ascending=False).reset_index().rename(columns={ACTIVITY: "step"})


def ward_record_gaps(steps: pd.DataFrame) -> pd.Series:
    """Hours between consecutive ward admission records of the same case."""
    ward = steps[steps[ACTIVITY] == WARD]
    return (ward.groupby(CASE_ID)[TIMESTAMP].diff().dt.total_seconds() / 3600).dropna()


def ward_stay_by_records(log: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    """Per patient with a ward record and a discharge: number of ward records and days from first ward record to discharge."""
    ward = steps[steps[ACTIVITY] == WARD].groupby(CASE_ID)[TIMESTAMP].agg(first="min", last="max", records="size")
    release = log[log[ACTIVITY].str.startswith("Release")].groupby(CASE_ID)[TIMESTAMP].min()
    table = ward.join(release.rename("release"), how="inner")
    table["stay_days"] = (table["release"] - table["first"]).dt.total_seconds() / 86400
    table["record_span_days"] = (table["last"] - table["first"]).dt.total_seconds() / 86400
    return table.reset_index()


def return_intervals(log: pd.DataFrame) -> pd.DataFrame:
    """Days from the first discharge to the first return to the ER, for patients who returned."""
    release = log[log[ACTIVITY].str.startswith("Release")].groupby(CASE_ID)[TIMESTAMP].min()
    back = log[log[ACTIVITY] == RETURN].groupby(CASE_ID)[TIMESTAMP].min()
    days = ((back - release.reindex(back.index)).dt.total_seconds() / 86400).dropna()
    return days.rename("days_to_return").reset_index()


def lab_tests_per_patient(log: pd.DataFrame) -> pd.Series:
    """Number of lab-test events per patient (0 for patients with none)."""
    labs = log[log[ACTIVITY].isin(LAB_ACTIVITIES)].groupby(CASE_ID).size()
    return labs.reindex(log[CASE_ID].unique()).fillna(0).astype(int)


def summarize(log: pd.DataFrame) -> pd.DataFrame:
    """The key numbers as a small table: metric, value, explanation."""
    steps = clinical_steps(log)
    per_case = rework_per_case(steps)
    n = len(per_case)
    gaps = ward_record_gaps(steps)
    stays = ward_stay_by_records(log, steps)
    multi = stays[stays["records"] > 1]
    returns = return_intervals(log)
    released = log[log[ACTIVITY].str.startswith("Release")][CASE_ID].nunique()
    labs = lab_tests_per_patient(log)

    rows = [
        ("patients", n, "patients analysed"),
        ("patients_with_rework", int(per_case["has_rework"].sum()), "patients who repeat at least one clinical step"),
        ("share_with_rework_pct", per_case["has_rework"].mean() * 100, "share of all patients"),
        ("patients_repeating_ward_record", int(per_case["repeated_steps"].str.contains(WARD).sum()), "patients with more than one Admission NC record"),
        ("extra_ward_records", int(len(gaps)), "Admission NC records beyond the first"),
        ("immediate_repeats", int(per_case["immediate_repeats"].sum()), "a step directly after itself"),
        ("patients_with_loop_back", int((per_case["loop_backs"] > 0).sum()), "a step reappears after a different step"),
        ("ward_gaps_under_24h_pct", (gaps < 24).mean() * 100, "share of gaps between repeated ward records under one day"),
        ("ward_gaps_median_hours", gaps.median(), "median hours between repeated ward records"),
        ("ward_record_span_median_days", multi["record_span_days"].median(), "median days from first to last ward record (patients with several)"),
        ("ward_stay_median_days_one_record", stays.loc[stays["records"] == 1, "stay_days"].median(), "median ward stay, patients with one ward record"),
        ("ward_stay_median_days_several_records", multi["stay_days"].median(), "median ward stay, patients with several ward records"),
        ("patients_returning", len(returns), "patients with a Return ER event"),
        ("released_patients", released, "patients with a discharge event"),
        ("return_rate_of_released_pct", len(returns) / released * 100, "share of discharged patients who return"),
        ("return_median_days", returns["days_to_return"].median(), "median days from discharge to return"),
    ]
    for days in RETURN_WINDOWS_DAYS:
        within = int((returns["days_to_return"] <= days).sum())
        rows.append((f"returns_within_{days}d", within, f"returns within {days} days of discharge"))
        rows.append((f"returns_within_{days}d_share_of_returns_pct", within / len(returns) * 100, f"share of returns within {days} days"))
        rows.append((f"returns_within_{days}d_share_of_released_pct", within / released * 100, f"share of all discharged patients who return within {days} days"))
    rows += [
        ("lab_tests_median_per_patient", float(labs.median()), "repeated blood tests are routine monitoring, not rework"),
        ("lab_events_total", int(labs.sum()), "lab-test events"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "explanation"])


def main():
    log = load_log()
    steps = clinical_steps(log)
    per_case = rework_per_case(steps)
    steps_table = step_rework_table(steps)
    summary = summarize(log)

    REWORK_PATH.parent.mkdir(parents=True, exist_ok=True)
    per_case.to_csv(REWORK_PATH, index=False)
    STEPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    steps_table.round(2).to_csv(STEPS_PATH, index=False)
    summary.round(2).to_csv(SUMMARY_PATH, index=False)

    get = summary.set_index("metric")["value"]
    print(f"Patients with rework     : {int(get['patients_with_rework'])} ({get['share_with_rework_pct']:.1f}%)")
    print(f"Repeated ward records    : {int(get['patients_repeating_ward_record'])} patients, {int(get['extra_ward_records'])} extra records")
    print(f"  under one day apart    : {get['ward_gaps_under_24h_pct']:.0f}% of gaps")
    print(f"Loops back (A, B, A)     : {int(get['patients_with_loop_back'])} patients")
    print(f"Returns to the ER        : {int(get['patients_returning'])} ({get['return_rate_of_released_pct']:.1f}% of discharged); "
          f"{get['returns_within_30d_share_of_returns_pct']:.0f}% of returns within 30 days")
    print(f"Saved: {REWORK_PATH}\n       {STEPS_PATH}\n       {SUMMARY_PATH}")
    return per_case


if __name__ == "__main__":
    main()
