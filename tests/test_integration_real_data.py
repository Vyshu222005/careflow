"""Checks against the real processed data. Skipped automatically when the pipeline has not been run."""
import pandas as pd
import pytest

import bundle
import conformance
import rework
from clean_data import load_clean_log
from prepare_log import load_cases, load_events
from validation import stage_counts

pytestmark = pytest.mark.integration
C = "case:concept:name"


@pytest.fixture(scope="module")
def data(real_paths):
    return load_clean_log(), load_events(), load_cases()


def test_event_counts_reconcile_14214_and_14208(data):
    clean, events, cases = data
    counts = stage_counts(clean, events)
    assert (counts.cleaned_events, counts.analysis_events, counts.patients) == (14214, 14208, 995)
    assert counts.excluded_activities == ("Release E",) and counts.excluded_events == 6 and counts.excluded_patients == 6
    assert len(cases) == 995 and cases["n_events"].sum() == 14208


def test_every_table_describes_the_same_995_patients(data):
    clean, events, cases = data
    log = conformance.load_log()
    assert set(clean[C]) == set(events[C]) == set(cases[C]) == set(log[C])
    assert len(conformance.check_conformance(log)) == 995 and len(bundle.bundle_table(log)) == 995


def test_headline_results_are_unchanged(data):
    log = conformance.load_log()
    conf = conformance.check_conformance(log)
    assert int(conf["conforms"].sum()) == 329
    card = bundle.scorecard(bundle.bundle_table(log)).set_index("element")
    assert round(card.loc["complete_bundle", "met_pct"], 1) == 18.4 and int(card.loc["antibiotics_1h", "met"]) == 259
    rw = rework.rework_per_case(rework.clinical_steps(log))
    assert int(rw["has_rework"].sum()) == 300
    assert int(conf["no_discharge"].sum()) == 256  # the discharge count is the same in every part of the project


def test_antibiotic_percentages_use_the_documented_denominators(data):
    log = conformance.load_log()
    card = bundle.scorecard(bundle.bundle_table(log)).set_index("element")
    treated = 776
    within = int(card.loc["antibiotics_1h", "met"])
    assert card.loc["antibiotics_1h", "applicable"] == 995  # the bundle counts all patients
    assert round(within / 995 * 100) == 26 and round(within / treated * 100) == 33  # the waiting-time tab counts treated patients


def test_prediction_dataset_matches_the_treated_population(data):
    from delay_features import build_dataset

    ds = build_dataset(conformance.load_log())
    assert (ds.n_patients, ds.n_with_antibiotics, ds.n_without_antibiotics) == (995, 776, 219)
    assert round(ds.y.mean() * 100, 1) == 66.6
