import numpy as np
import pandas as pd
import pytest

from delay_features import (
    FEATURE_COLUMNS, FORBIDDEN, STAGE_FEATURES, LeakageError, assert_no_leakage, build_dataset, feature_row, registration_load,
)

C, A, T = "case:concept:name", "concept:name", "time:timestamp"


def test_no_forbidden_column_is_a_feature():
    for stage, columns in STAGE_FEATURES.items():
        assert not set(columns) & FORBIDDEN, stage


@pytest.mark.parametrize("bad", ["hours_to_antibiotics", "delayed", "Infusion", "DiagnosticBlood", "CRP", "LacticAcid", "Diagnose"])
def test_leakage_guard_rejects_forbidden_columns(bad):
    with pytest.raises(LeakageError):
        assert_no_leakage(FEATURE_COLUMNS + [bad])


def test_leakage_guard_rejects_undeclared_columns_and_triage_columns_at_registration():
    with pytest.raises(LeakageError):
        assert_no_leakage(["age", "mystery"])
    with pytest.raises(LeakageError):
        assert_no_leakage(["age", "minutes_to_sepsis_triage"], stage="registration")
    assert_no_leakage(["age", "minutes_to_sepsis_triage"], stage="triage")  # allowed once triage is done


def test_dataset_has_one_row_per_patient_and_only_treated_patients(synthetic_log):
    ds = build_dataset(synthetic_log)
    assert ds.table[C].is_unique
    treated = synthetic_log[synthetic_log[A] == "IV Antibiotics"][C].nunique()
    assert ds.n_with_antibiotics == treated == len(ds.table)
    assert ds.n_patients == synthetic_log[C].nunique()
    assert ds.n_without_antibiotics == ds.n_patients - treated
    assert list(ds.X.columns) == FEATURE_COLUMNS


def test_delayed_label_matches_the_one_hour_rule(synthetic_log):
    ds = build_dataset(synthetic_log).table
    assert ((ds["hours_to_antibiotics"] > 1).astype(int) == ds["delayed"]).all()
    assert ds["hours_to_antibiotics"].min() >= 0


def test_er_load_counts_only_earlier_registrations():
    times = pd.Series(pd.to_datetime(["2014-01-01 10:00", "2014-01-01 10:30", "2014-01-01 11:10", "2014-01-01 20:00"], utc=True))
    assert registration_load(times, 1).tolist() == [0, 1, 1, 0]
    assert registration_load(times, 6).tolist() == [0, 1, 2, 0]


def test_triage_stage_uses_only_events_before_sepsis_triage(synthetic_log):
    ds = build_dataset(synthetic_log, "triage")
    assert ds.stage == "triage" and set(ds.feature_columns) == set(STAGE_FEATURES["triage"])
    # the synthetic lactate is recorded 1 minute after registration, which is always before sepsis triage (>= 3 minutes)
    assert (ds.table["lactate_measured_before"] == 1).all()
    assert (ds.table["minutes_to_sepsis_triage"] > ds.table["minutes_to_triage"]).all()
    # patients treated before sepsis triage are not part of the triage-stage question
    assert len(ds.table) <= ds.n_with_antibiotics


def test_lab_measured_after_the_cut_is_ignored():
    log = pd.DataFrame({
        C: ["a"] * 5, A: ["ER Registration", "ER Triage", "ER Sepsis Triage", "LacticAcid", "IV Antibiotics"],
        T: pd.to_datetime(["2014-01-01 10:00", "2014-01-01 10:05", "2014-01-01 10:10", "2014-01-01 10:20", "2014-01-01 11:30"], utc=True),
        "org:group": "A", "Age": 70.0, "LacticAcid": [np.nan, np.nan, np.nan, 3.1, np.nan], "CRP": np.nan, "Leucocytes": np.nan,
        **{f: False for f in ["InfectionSuspected", "SIRSCritTachypnea", "SIRSCritHeartRate", "SIRSCritTemperature", "SIRSCritLeucos", "SIRSCriteria2OrMore",
                              "DisfuncOrg", "Hypotensie", "Hypoxie", "Oligurie"]},
    })
    row = build_dataset(log, "triage").table.iloc[0]
    assert row["lactate_measured_before"] == 0 and row["lactate_value"] == 0  # measured after sepsis triage: not known yet
    assert row["delayed"] == 1


def test_unknown_stage_is_rejected(synthetic_log):
    with pytest.raises(ValueError):
        build_dataset(synthetic_log, "discharge")


def test_feature_row_defaults_and_guard():
    row = feature_row({"age": 70, "hypotension": 1})
    assert list(row.columns) == FEATURE_COLUMNS and row.loc[0, "age"] == 70 and row.loc[0, "oliguria"] == 0
