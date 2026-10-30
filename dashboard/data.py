"""Cached data loading and the selection context shared by every tab."""
from dataclasses import dataclass

import pandas as pd
import streamlit as st

import bundle
import conformance
import rework
from bottlenecks import target_summary, time_from_registration
from clean_data import CLEAN_PATH
from load_data import CASE_ID
from prepare_log import CASES_PATH, EVENTS_PATH, load_cases, load_events
from validation import StageCounts, selected_counts, stage_counts, validate_inputs
from variants import add_milestone_variant, variant_table


@st.cache_data(show_spinner="Loading data...")
def load_data():
    events = load_events()
    cases = add_milestone_variant(events, load_cases())
    cases["start_date"] = cases["start"].dt.date
    return events, cases


@st.cache_data(show_spinner="Checking conformance and rework...")
def load_conformance_rework():
    """Conformance and rework use the cleaned log with every discharge code kept (see src/conformance.py)."""
    log = conformance.load_log()
    conformance_table = conformance.check_conformance(log)
    rework_table = rework.rework_per_case(rework.clinical_steps(log))
    return log, conformance_table, rework_table


@st.cache_data(show_spinner="Scoring the care bundle...")
def load_bundle_table():
    return bundle.bundle_table(conformance.load_log())


def data_ready() -> bool:
    return EVENTS_PATH.exists() and CASES_PATH.exists() and CLEAN_PATH.exists()


@dataclass
class Context:
    """Everything a tab needs for the patients currently selected in the sidebar."""

    cases_all: pd.DataFrame
    events_all: pd.DataFrame
    clean_log: pd.DataFrame
    conformance_all: pd.DataFrame
    rework_all: pd.DataFrame
    bundle_all: pd.DataFrame
    cases: pd.DataFrame
    events: pd.DataFrame
    hours: pd.DataFrame
    antibiotics: dict
    milestone_table: pd.DataFrame
    has_abx: bool
    counts: StageCounts
    cleaned_events_selected: int
    filters_text: str


def load_all():
    """Load and validate every input. Raises DataError with a helpful message if something is wrong."""
    events_all, cases_all = load_data()
    clean_log, conformance_all, rework_all = load_conformance_rework()
    bundle_all = load_bundle_table()
    validate_inputs(clean_log, events_all, cases_all)
    return events_all, cases_all, clean_log, conformance_all, rework_all, bundle_all


def build_context(loaded, cases: pd.DataFrame, filters_text: str) -> Context:
    events_all, cases_all, clean_log, conformance_all, rework_all, bundle_all = loaded
    events = events_all[events_all[CASE_ID].isin(cases[CASE_ID])]
    hours = time_from_registration(events)
    antibiotics = target_summary(hours["IV Antibiotics"])
    cleaned_selected, _ = selected_counts(clean_log, events_all, cases[CASE_ID])
    return Context(
        cases_all=cases_all, events_all=events_all, clean_log=clean_log, conformance_all=conformance_all,
        rework_all=rework_all, bundle_all=bundle_all, cases=cases, events=events, hours=hours, antibiotics=antibiotics,
        milestone_table=variant_table(cases, "milestone_variant"), has_abx=antibiotics["cases"] > 0,
        counts=stage_counts(clean_log, events_all), cleaned_events_selected=cleaned_selected, filters_text=filters_text,
    )
