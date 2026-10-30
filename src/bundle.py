"""Sepsis care-bundle scorecard with a monthly control chart.

The scorecard is ADAPTED from the Surviving Sepsis Campaign hour-1 bundle, using only what the
event log records. It is not an official quality measure: the clock starts at ER registration (the
log has no "time of recognition"), and blood cultures and vasopressors are not in the data.

Elements
    lactate_1h        lactate measured within 1 hour of registration (all patients)
    antibiotics_1h    antibiotics within 1 hour of registration (all patients)
    antibiotics_3h    antibiotics within 3 hours of registration (a looser reference)
    fluid_3h          IV fluid within 3 hours, for patients with hypotension
    repeat_lactate    lactate measured again, for patients whose first lactate is above 2
    complete_bundle   every element that applies to the patient is met

The monthly trend is a p-chart (control chart for a proportion): limits are the overall rate plus or
minus 3 standard errors, so a month is flagged only when it differs by more than chance would explain.
A run of 8 months on the same side of the average is flagged as a lasting shift.

Usage:
    python src/bundle.py

Outputs:
    data/processed/sepsis_bundle.csv    one row per patient
    reports/bundle_scorecard.csv        compliance per element
    reports/bundle_trend.csv            monthly compliance with control limits
"""
from pathlib import Path

import numpy as np
import pandas as pd

from conformance import load_log
from load_data import ACTIVITY, CASE_ID, TIMESTAMP

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "processed" / "sepsis_bundle.csv"
SCORECARD_PATH = ROOT / "reports" / "bundle_scorecard.csv"
TREND_PATH = ROOT / "reports" / "bundle_trend.csv"

REGISTRATION, LACTATE, ANTIBIOTICS, FLUID = "ER Registration", "LacticAcid", "IV Antibiotics", "IV Liquid"
HOUR_TARGET, LOOSE_TARGET = 1, 3
HIGH_LACTATE = 2.0
MIN_MONTH_N = 10  # months with fewer patients than this are not judged against the control limits
RUN_LENGTH = 8

# column: (label, column that says whether the element applies, or None for "all patients")
ELEMENTS = {
    "lactate_1h": ("Lactate measured within 1 hour", None),
    "antibiotics_1h": ("Antibiotics within 1 hour", None),
    "antibiotics_3h": ("Antibiotics within 3 hours", None),
    "fluid_3h": ("IV fluid within 3 hours (patients with hypotension)", "hypotension"),
    "repeat_lactate": ("Lactate measured again (first lactate above 2)", "high_lactate"),
    "complete_bundle": ("All applicable elements met", None),
}
TREND_ELEMENTS = ["lactate_1h", "antibiotics_1h", "antibiotics_3h", "complete_bundle"]


def bundle_table(log: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: which bundle elements were met and which apply."""
    first = log.groupby([CASE_ID, ACTIVITY])[TIMESTAMP].min().unstack()
    registered = first[REGISTRATION]
    hours = lambda step: (first[step] - registered).dt.total_seconds() / 3600  # noqa: E731

    lactate_events = log[log[ACTIVITY] == LACTATE].sort_values([CASE_ID, TIMESTAMP], kind="stable")
    table = pd.DataFrame(index=first.index)
    table["registered"] = registered
    table["month"] = registered.dt.tz_convert(None).dt.to_period("M").astype(str)
    table["lactate_1h"] = (hours(LACTATE) <= HOUR_TARGET).fillna(False)
    table["antibiotics_1h"] = (hours(ANTIBIOTICS) <= HOUR_TARGET).fillna(False)
    table["antibiotics_3h"] = (hours(ANTIBIOTICS) <= LOOSE_TARGET).fillna(False)
    table["hypotension"] = log.groupby(CASE_ID)["Hypotensie"].first().astype(bool).reindex(table.index).fillna(False)
    table["fluid_3h"] = (hours(FLUID) <= LOOSE_TARGET).fillna(False)
    table["first_lactate"] = lactate_events.groupby(CASE_ID)["LacticAcid"].first().reindex(table.index)
    table["high_lactate"] = (table["first_lactate"] > HIGH_LACTATE).fillna(False)
    table["repeat_lactate"] = lactate_events.groupby(CASE_ID).size().reindex(table.index).fillna(0) >= 2
    table["complete_bundle"] = (
        table["lactate_1h"] & table["antibiotics_1h"]
        & (~table["hypotension"] | table["fluid_3h"])
        & (~table["high_lactate"] | table["repeat_lactate"])
    )
    return table.reset_index()


def scorecard(table: pd.DataFrame) -> pd.DataFrame:
    """Compliance per element: patients it applies to, patients who met it, and the share."""
    rows = []
    for column, (label, applies) in ELEMENTS.items():
        applicable = table[applies] if applies else pd.Series(True, index=table.index)
        met = int(table.loc[applicable, column].sum())
        n = int(applicable.sum())
        rows.append({"element": column, "label": label, "applicable": n, "met": met, "met_pct": met / n * 100 if n else np.nan})
    return pd.DataFrame(rows)


def control_chart(table: pd.DataFrame, element: str, min_n: int = MIN_MONTH_N) -> pd.DataFrame:
    """Monthly compliance for one element with p-chart limits and signals.

    The centre line is the overall rate. A month is a signal if it falls outside the limits, or if it
    is part of a run of RUN_LENGTH months on the same side of the centre line.
    """
    label, applies = ELEMENTS[element]
    data = table[table[applies]] if applies else table
    monthly = data.groupby("month")[element].agg(n="size", met="sum").reset_index()
    monthly = monthly[monthly["n"] >= min_n].reset_index(drop=True)
    if monthly.empty:
        return monthly.assign(pct=np.nan, center=np.nan, ucl=np.nan, lcl=np.nan, signal="")
    centre = monthly["met"].sum() / monthly["n"].sum()
    se = np.sqrt(centre * (1 - centre) / monthly["n"])
    monthly["pct"] = monthly["met"] / monthly["n"] * 100
    monthly["center"] = centre * 100
    monthly["ucl"] = np.minimum(centre + 3 * se, 1) * 100
    monthly["lcl"] = np.maximum(centre - 3 * se, 0) * 100

    signal = np.where(monthly["pct"] > monthly["ucl"], "above the limit", np.where(monthly["pct"] < monthly["lcl"], "below the limit", ""))
    side = np.sign(monthly["pct"] - monthly["center"]).to_numpy()
    start = 0
    for i in range(1, len(side) + 1):
        if i == len(side) or side[i] != side[start] or side[start] == 0:
            if side[start] != 0 and i - start >= RUN_LENGTH:
                for j in range(start, i):
                    if not signal[j]:
                        signal[j] = f"run of {RUN_LENGTH}+ {'above' if side[start] > 0 else 'below'} average"
            start = i
    monthly["signal"] = signal
    return monthly


def trend_summary(table: pd.DataFrame, elements=TREND_ELEMENTS) -> pd.DataFrame:
    return pd.concat([control_chart(table, e).assign(element=e) for e in elements], ignore_index=True)


def main():
    table = bundle_table(load_log())
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(BUNDLE_PATH, index=False)
    card = scorecard(table)
    trend = trend_summary(table)
    SCORECARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    card.round(2).to_csv(SCORECARD_PATH, index=False)
    trend.round(2).to_csv(TREND_PATH, index=False)

    print("Bundle scorecard (share of applicable patients):")
    for _, row in card.iterrows():
        print(f"  {row['label']:<55} {int(row['met']):>4} of {int(row['applicable']):>4}  ({row['met_pct']:.0f}%)")
    signals = trend[trend["signal"] != ""]
    print(f"Control-chart signals: {len(signals)}")
    for _, row in signals.iterrows():
        print(f"  {row['element']:<16} {row['month']}  {row['pct']:.0f}% ({row['signal']})")
    print(f"Saved: {BUNDLE_PATH}\n       {SCORECARD_PATH}\n       {TREND_PATH}")
    return table


if __name__ == "__main__":
    main()
