"""Input validation and count reconciliation.

Two jobs:

1. `validate_inputs` checks that the processed files the dashboard needs exist and have the expected
   columns and no missing identifiers, and raises a `DataError` with a clear message otherwise.
2. `stage_counts` / `reconcile` explain how many patients and events there are at each stage of the
   pipeline, so the numbers shown in the dashboard and the documentation can never silently disagree.

Why 14,214 and 14,208? The cleaned log has 14,214 events. `prepare_log.py` then drops activities that
occur in fewer than 1% of cases (only `Release E`: 6 events in 6 patients). Pathway, process-map and
waiting-time analysis use the 14,208 events that remain; the patient count (995) does not change.
"""
from dataclasses import dataclass

import pandas as pd

from load_data import ACTIVITY, CASE_ID, TIMESTAMP

REQUIRED_EVENT_COLUMNS = [CASE_ID, ACTIVITY, TIMESTAMP, "org:group", "Age"]
REQUIRED_CASE_COLUMNS = [CASE_ID, "start", "duration_days", "outcome", "age_group", "returned_to_er", "reached_icu", "variant"]


class DataError(Exception):
    """The processed data is missing or malformed. The message says how to fix it."""


def validate_frame(frame: pd.DataFrame, required: list, name: str) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise DataError(f"{name} is missing columns {missing}. Re-run the pipeline: python src/clean_data.py, then python src/prepare_log.py.")
    if frame.empty:
        raise DataError(f"{name} is empty. Re-run the pipeline.")
    if frame[CASE_ID].isna().any():
        raise DataError(f"{name} has events without a case ID. A plain pd.read_csv turns the case 'NA' into a missing value; use load_events()/load_cases().")


def validate_inputs(clean: pd.DataFrame, events: pd.DataFrame, cases: pd.DataFrame) -> None:
    validate_frame(clean, REQUIRED_EVENT_COLUMNS, "The cleaned log")
    validate_frame(events, REQUIRED_EVENT_COLUMNS, "The prepared event log")
    validate_frame(cases, REQUIRED_CASE_COLUMNS, "The case table")
    if set(cases[CASE_ID]) != set(events[CASE_ID]):
        raise DataError("The case table and the event log contain different patients. Re-run python src/prepare_log.py.")


@dataclass(frozen=True)
class StageCounts:
    cleaned_events: int
    analysis_events: int
    patients: int
    excluded_events: int
    excluded_activities: tuple
    excluded_patients: int

    @property
    def explanation(self) -> str:
        names = ", ".join(self.excluded_activities) or "none"
        return (f"{self.cleaned_events:,} cleaned events; {self.excluded_events} rare-step events ({names}, in {self.excluded_patients} patients) "
                f"are left out of pathway and waiting-time analysis, giving {self.analysis_events:,}. Patients: {self.patients:,} in both.")


def stage_counts(clean: pd.DataFrame, events: pd.DataFrame) -> StageCounts:
    """Reconcile the cleaned log with the prepared (analysis) log."""
    removed_activities = tuple(sorted(set(clean[ACTIVITY]) - set(events[ACTIVITY])))
    removed = clean[clean[ACTIVITY].isin(removed_activities)]
    counts = StageCounts(
        cleaned_events=len(clean), analysis_events=len(events), patients=clean[CASE_ID].nunique(),
        excluded_events=len(removed), excluded_activities=removed_activities, excluded_patients=removed[CASE_ID].nunique(),
    )
    if counts.cleaned_events - counts.analysis_events != counts.excluded_events or clean[CASE_ID].nunique() != events[CASE_ID].nunique():
        raise DataError("The cleaned and prepared logs cannot be reconciled: events were lost in a way the rare-activity filter does not explain.")
    return counts


def selected_counts(clean: pd.DataFrame, events: pd.DataFrame, case_ids) -> tuple:
    """(cleaned events, analysis events) for a set of patients."""
    ids = set(case_ids)
    return int(clean[CASE_ID].isin(ids).sum()), int(events[CASE_ID].isin(ids).sum())
