"""Prepare the cleaned log for process mining.

Steps:
    1. Order events consistently: by case, time, then a fixed process order for
       events that share a timestamp (batched lab results, for example).
    2. Filter noise: drop activities that occur in very few cases.
    3. Add derived features: position in case, time since case start, waiting time
       since the previous event, previous/next activity.
    4. Build a one-row-per-case table: duration, longest wait, variant, outcome, flags.
    5. Check that pm4py can read the result.

The event log keeps the standard XES column names (case:concept:name, concept:name,
time:timestamp), which is the format pm4py expects.

Usage:
    python src/prepare_log.py

Outputs:
    data/processed/sepsis_events.csv   one row per event, with derived features
    data/processed/sepsis_cases.csv    one row per case
    reports/event_log_preparation.md
"""
from pathlib import Path

import pandas as pd
import pm4py

from clean_data import load_clean_log
from load_data import ACTIVITY, CASE_ID, TIMESTAMP

ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = ROOT / "data" / "processed" / "sepsis_events.csv"
CASES_PATH = ROOT / "data" / "processed" / "sepsis_cases.csv"
REPORT_PATH = ROOT / "reports" / "event_log_preparation.md"

# Activities seen in fewer than this share of cases are treated as noise.
MIN_CASE_SHARE = 0.01

# Used only to order events that have the same timestamp.
ACTIVITY_ORDER = [
    "ER Registration", "ER Triage", "ER Sepsis Triage",
    "Leucocytes", "CRP", "LacticAcid",
    "IV Liquid", "IV Antibiotics",
    "Admission NC", "Admission IC",
    "Release A", "Release B", "Release C", "Release D", "Release E",
    "Return ER",
]
NO_RELEASE = "No release recorded"
AGE_BINS = [0, 44, 64, 79, 120]
AGE_LABELS = ["under 45", "45-64", "65-79", "80+"]


def order_events(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by case and time; ties follow ACTIVITY_ORDER so the order is the same in every case."""
    rank = {name: i for i, name in enumerate(ACTIVITY_ORDER)}
    df = df.assign(_rank=df[ACTIVITY].map(rank).fillna(len(rank)))
    df = df.sort_values([CASE_ID, TIMESTAMP, "_rank"], kind="stable")
    return df.drop(columns="_rank").reset_index(drop=True)


def filter_rare_activities(df: pd.DataFrame, min_case_share: float = MIN_CASE_SHARE):
    """Drop activities that appear in under `min_case_share` of cases. Returns (log, table of dropped)."""
    case_share = df.groupby(ACTIVITY)[CASE_ID].nunique() / df[CASE_ID].nunique()
    rare = case_share[case_share < min_case_share].index
    dropped = pd.DataFrame(
        {
            "events": df[df[ACTIVITY].isin(rare)].groupby(ACTIVITY).size(),
            "case_share_%": (case_share[rare] * 100).round(1),
        }
    )
    return df[~df[ACTIVITY].isin(rare)].reset_index(drop=True), dropped


def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-event features. Times are in hours."""
    df = df.copy()
    by_case = df.groupby(CASE_ID, sort=False)
    df["event_order"] = by_case.cumcount() + 1
    df["hours_since_start"] = (df[TIMESTAMP] - by_case[TIMESTAMP].transform("min")).dt.total_seconds() / 3600
    df["hours_since_prev"] = by_case[TIMESTAMP].diff().dt.total_seconds() / 3600
    df["prev_activity"] = by_case[ACTIVITY].shift(1)
    df["next_activity"] = by_case[ACTIVITY].shift(-1)
    return df


def case_outcomes(df: pd.DataFrame) -> pd.Series:
    """First Release code of each case, or NO_RELEASE."""
    release = df[df[ACTIVITY].str.startswith("Release")].groupby(CASE_ID)[ACTIVITY].first()
    return release.reindex(df[CASE_ID].unique()).fillna(NO_RELEASE)


def build_case_table(df: pd.DataFrame, outcomes: pd.Series) -> pd.DataFrame:
    """One row per case (df must already have the event features).

    `outcomes` comes from the unfiltered log, so dropping a rare Release code
    does not turn its cases into "No release recorded".
    """
    by_case = df.groupby(CASE_ID, sort=False)
    cases = by_case.agg(
        start=(TIMESTAMP, "min"),
        end=(TIMESTAMP, "max"),
        n_events=(ACTIVITY, "size"),
        n_activities=(ACTIVITY, "nunique"),
        longest_wait_hours=("hours_since_prev", "max"),
        age=("Age", "first"),
        diagnose=("Diagnose", "first"),
    )
    cases["duration_hours"] = (cases["end"] - cases["start"]).dt.total_seconds() / 3600
    cases["duration_days"] = cases["duration_hours"] / 24
    cases["variant"] = by_case[ACTIVITY].agg(" > ".join)

    cases["outcome"] = outcomes.reindex(cases.index)

    has = lambda name: (df[ACTIVITY] == name).groupby(df[CASE_ID]).any().reindex(cases.index)
    cases["reached_icu"] = has("Admission IC")
    cases["admitted_normal_care"] = has("Admission NC")
    cases["returned_to_er"] = has("Return ER")
    cases["got_antibiotics"] = has("IV Antibiotics")

    cases["age_group"] = pd.cut(cases["age"], AGE_BINS, labels=AGE_LABELS).astype(str)
    cases["start_month"] = cases["start"].dt.tz_convert(None).dt.to_period("M").astype(str)
    return cases.reset_index()


def check_pm4py(df: pd.DataFrame) -> dict:
    """Confirm pm4py reads the log and finds the expected structure."""
    return {
        "cases": df[CASE_ID].nunique(),
        "start_activities": pm4py.get_start_activities(df),
        "end_activities": pm4py.get_end_activities(df),
        "dfg_edges": len(pm4py.discover_dfg(df)[0]),
    }


def prepare(clean: pd.DataFrame):
    """Return (events, cases, info) ready for process mining."""
    ordered = order_events(clean)
    filtered, dropped = filter_rare_activities(ordered)
    events = add_event_features(filtered)
    cases = build_case_table(events, case_outcomes(ordered))
    info = {
        "variants_file_order": clean.groupby(CASE_ID, sort=False)[ACTIVITY].agg(" > ".join).nunique(),
        "before": {"events": len(ordered), "cases": ordered[CASE_ID].nunique(),
                   "activities": ordered[ACTIVITY].nunique(),
                   "variants": ordered.groupby(CASE_ID)[ACTIVITY].agg(" > ".join).nunique()},
        "after": {"events": len(events), "cases": events[CASE_ID].nunique(),
                  "activities": events[ACTIVITY].nunique(), "variants": cases["variant"].nunique()},
        "dropped": dropped,
        "pm4py": check_pm4py(events),
    }
    return events, cases, info


def write_report(events: pd.DataFrame, cases: pd.DataFrame, info: dict, path: Path = REPORT_PATH) -> None:
    b, a = info["before"], info["after"]
    wait = events["hours_since_prev"].dropna()
    pm = info["pm4py"]
    lines = [
        "# Event Log Preparation",
        "",
        "Generated by `src/prepare_log.py` from `data/processed/sepsis_clean.csv`.",
        "",
        "## Before vs after noise filtering",
        "",
        "| | Before | After |",
        "|---|---|---|",
        f"| Cases | {b['cases']:,} | {a['cases']:,} |",
        f"| Events | {b['events']:,} | {a['events']:,} |",
        f"| Activities | {b['activities']} | {a['activities']} |",
        f"| Distinct pathways (variants) | {b['variants']:,} | {a['variants']:,} |",
        "",
        "## Noise filter",
        "",
        f"Activities that occur in fewer than {MIN_CASE_SHARE:.0%} of cases are dropped. "
        "Change `MIN_CASE_SHARE` in `src/prepare_log.py` to make this stricter or looser; "
        "the unfiltered log stays in `data/processed/sepsis_clean.csv`.",
        "",
        "| Dropped activity | Events | Share of cases |",
        "|---|---|---|",
    ]
    for name, row in info["dropped"].iterrows():
        lines.append(f"| {name} | {int(row['events'])} | {row['case_share_%']}% |")
    lines += [
        "",
        "## Event order",
        "",
        "Many events share one timestamp (for example Leucocytes, CRP and LacticAcid are recorded together). "
        "Left in file order, the same pathway shows up in several different orders and looks like different variants. "
        "Events with the same timestamp are therefore sorted in one fixed process order (`ACTIVITY_ORDER`), "
        f"so identical pathways are counted once. This reduced the number of distinct pathways from "
        f"{info['variants_file_order']:,} (file order) to {b['variants']:,}.",
        "",
        "## Derived features",
        "",
        "Event table (`data/processed/sepsis_events.csv`), one row per event, times in hours:",
        "",
        "| Column | Meaning |",
        "|---|---|",
        "| `event_order` | Position of the event within its case (1, 2, 3, ...) |",
        "| `hours_since_start` | Time from the first event of the case |",
        "| `hours_since_prev` | Waiting time since the previous event (empty for the first event) |",
        "| `prev_activity`, `next_activity` | Neighbouring activities in the case |",
        "",
        "Case table (`data/processed/sepsis_cases.csv`), one row per case:",
        "",
        "| Column | Meaning |",
        "|---|---|",
        "| `start`, `end` | First and last event time |",
        "| `duration_hours`, `duration_days` | Case duration |",
        "| `n_events`, `n_activities` | Number of events and distinct activities |",
        "| `longest_wait_hours` | Largest gap between two consecutive events |",
        "| `variant` | The activity sequence, joined with ` > ` |",
        "| `outcome` | First `Release` code, or `No release recorded` (taken from the unfiltered log) |",
        "| `reached_icu`, `admitted_normal_care`, `returned_to_er`, `got_antibiotics` | True/False flags |",
        "| `age`, `age_group`, `diagnose`, `start_month` | Patient and timing attributes |",
        "",
        "## Waiting time between consecutive events",
        "",
        f"- Median: {wait.median():.2f} hours; mean: {wait.mean():.1f} hours; 90th percentile: {wait.quantile(0.9):.1f} hours.",
        f"- {(wait == 0).mean():.0%} of consecutive events have the same timestamp (recorded together).",
        "",
        "## pm4py check",
        "",
        f"- pm4py reads the log: {pm['cases']:,} cases.",
        f"- Start activities: {pm['start_activities']}",
        f"- Directly-follows relations found: {pm['dfg_edges']}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def read_processed(path: Path) -> pd.DataFrame:
    """Read a processed CSV. Only empty cells are missing, because one case is literally named 'NA'."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def load_events(path: Path = EVENTS_PATH) -> pd.DataFrame:
    df = read_processed(path)
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], utc=True)
    return df


def load_cases(path: Path = CASES_PATH) -> pd.DataFrame:
    df = read_processed(path)
    for col in ("start", "end"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def main():
    events, cases, info = prepare(load_clean_log())
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(EVENTS_PATH, index=False)
    cases.to_csv(CASES_PATH, index=False)
    write_report(events, cases, info)

    b, a = info["before"], info["after"]
    print(f"Events  : {b['events']:,} -> {a['events']:,}")
    print(f"Variants: {info['variants_file_order']:,} (file order) -> {b['variants']:,} (consistent order) -> {a['variants']:,} (after filter)")
    print(f"Dropped : {', '.join(info['dropped'].index) or 'none'}")
    print(f"Saved   : {EVENTS_PATH}\n          {CASES_PATH}\nReport  : {REPORT_PATH}")
    return events, cases


if __name__ == "__main__":
    main()
