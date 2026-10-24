"""Bottleneck analysis: where do patients wait?

Works on the clinical-steps log (repeated lab tests removed), so a "wait" is the
time between two clinical steps, for example from sepsis triage to IV fluid.

Three views:
    transition_stats        time between each pair of consecutive steps
    time_from_registration  time from ER registration to each clinical step
    performance map         the process map with the median wait on every arrow

Long gaps that are not waiting in the ER are labelled and kept apart:
    return to ER            Release -> Return ER, the patient comes back weeks later
    time on ward / in ICU   any step that starts at an admission (stay, transfer, discharge)
    repeated ward record    Admission NC followed by another Admission NC

Usage:
    python src/bottlenecks.py

Outputs:
    reports/bottleneck_transitions.csv
    reports/figures/process_performance_dfg.png
"""
from pathlib import Path

import pandas as pd
import pm4py

from discovery import DFG_PATH_SHARE, ensure_graphviz, milestone_log
from load_data import ACTIVITY, CASE_ID, TIMESTAMP
from prepare_log import load_events

ROOT = Path(__file__).resolve().parents[1]
TRANSITIONS_PATH = ROOT / "reports" / "bottleneck_transitions.csv"
PERFORMANCE_MAP_PATH = ROOT / "reports" / "figures" / "process_performance_dfg.png"

REGISTRATION = "ER Registration"
RETURN = "Return ER"
STEPS_FROM_REGISTRATION = ["ER Triage", "ER Sepsis Triage", "IV Liquid", "IV Antibiotics", "Admission NC", "Admission IC"]
# Antibiotics within 1 hour is the commonly quoted sepsis target (Surviving Sepsis Campaign hour-1 bundle);
# 3 hours is shown as a looser reference.
TARGET_HOURS = (1, 3)
MIN_CASES = 30


def format_hours(hours: float) -> str:
    """Human-readable duration: minutes, hours or days."""
    if hours < 1:
        return f"{hours * 60:.0f} min"
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} days"


def clinical_steps(events: pd.DataFrame) -> pd.DataFrame:
    """Clinical-step events with the next step and the wait until it (hours)."""
    steps = milestone_log(events)
    by_case = steps.groupby(CASE_ID, sort=False)
    steps["next_activity"] = by_case[ACTIVITY].shift(-1)
    steps["wait_hours"] = (by_case[TIMESTAMP].shift(-1) - steps[TIMESTAMP]).dt.total_seconds() / 3600
    return steps


def transition_stats(steps: pd.DataFrame) -> pd.DataFrame:
    """One row per pair of consecutive steps, most frequent first."""
    pairs = steps.dropna(subset=["next_activity"])
    table = (
        pairs.groupby([ACTIVITY, "next_activity"])["wait_hours"]
        .agg(cases="size", mean_hours="mean", median_hours="median",
             p90_hours=lambda s: s.quantile(0.9), total_hours="sum")
        .reset_index()
        .rename(columns={ACTIVITY: "from_step", "next_activity": "to_step"})
        .sort_values("cases", ascending=False)
        .reset_index(drop=True)
    )
    table["kind"] = "wait between steps"
    table.loc[table["to_step"] == RETURN, "kind"] = "return to ER"
    # Share of time spent in hospital; the weeks before a patient returns are not counted.
    in_hospital = table["kind"] != "return to ER"
    table["share_of_hospital_time_pct"] = table["total_hours"] / table.loc[in_hospital, "total_hours"].sum() * 100
    table.loc[~in_hospital, "share_of_hospital_time_pct"] = float("nan")
    table.loc[table["from_step"] == "Admission NC", "kind"] = "time on ward"
    table.loc[table["from_step"] == "Admission IC", "kind"] = "time in ICU"
    table.loc[(table["from_step"] == "Admission NC") & (table["to_step"] == "Admission NC"), "kind"] = "repeated ward record"
    return table


def er_waits(table: pd.DataFrame, min_cases: int = MIN_CASES) -> pd.DataFrame:
    """Waits between steps in the ER phase, longest typical wait first (frequent transitions only)."""
    waits = table[(table["kind"] == "wait between steps") & (table["cases"] >= min_cases)]
    return waits.sort_values("median_hours", ascending=False).reset_index(drop=True)


def time_from_registration(events: pd.DataFrame) -> pd.DataFrame:
    """Hours from ER registration to the first time each step happens (one row per case)."""
    steps = milestone_log(events)
    first = steps.groupby([CASE_ID, ACTIVITY])[TIMESTAMP].min().unstack()
    start = first[REGISTRATION]
    hours = pd.DataFrame({name: (first[name] - start).dt.total_seconds() / 3600 for name in STEPS_FROM_REGISTRATION})
    return hours


def target_summary(hours: pd.Series) -> dict:
    hours = hours.dropna()
    return {
        "cases": len(hours),
        "median_hours": hours.median(),
        "p90_hours": hours.quantile(0.9),
        **{f"within_{t}h_pct": (hours <= t).mean() * 100 for t in TARGET_HOURS},
    }


def save_performance_map(steps: pd.DataFrame, path: Path = PERFORMANCE_MAP_PATH) -> None:
    """Process map with the median wait on each arrow (same arrows as the frequency map)."""
    ensure_graphviz()
    core = steps[[CASE_ID, ACTIVITY, TIMESTAMP]]  # pm4py shifts every column, so give it only the three it needs
    frequency, start, end = pm4py.discover_dfg(core)
    keep, start, end = pm4py.filter_dfg_paths_percentage(frequency, start, end, percentage=DFG_PATH_SHARE)
    performance, _, _ = pm4py.discover_performance_dfg(core, perf_aggregation_key="all")
    performance = {edge: stats for edge, stats in performance.items() if edge in keep}
    path.parent.mkdir(parents=True, exist_ok=True)
    pm4py.save_vis_performance_dfg(
        performance, start, end, str(path), aggregation_measure="median",
        graph_title="Median time between steps (most frequent paths)",
    )


def main():
    events = load_events()
    steps = clinical_steps(events)
    table = transition_stats(steps)
    TRANSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.round(2).to_csv(TRANSITIONS_PATH, index=False)
    save_performance_map(steps)

    print("Slowest waits between steps (at least %d cases):" % MIN_CASES)
    for _, row in er_waits(table).head(5).iterrows():
        print(f"  {row['from_step']} -> {row['to_step']}: median {format_hours(row['median_hours'])}, "
              f"90th percentile {format_hours(row['p90_hours'])} ({row['cases']} cases)")
    antibiotics = target_summary(time_from_registration(events)["IV Antibiotics"])
    print(f"Registration -> antibiotics: median {format_hours(antibiotics['median_hours'])}, "
          f"{antibiotics['within_1h_pct']:.0f}% within 1 h, {antibiotics['within_3h_pct']:.0f}% within 3 h")
    print(f"Saved: {TRANSITIONS_PATH}\n       {PERFORMANCE_MAP_PATH}")
    return table


if __name__ == "__main__":
    main()
