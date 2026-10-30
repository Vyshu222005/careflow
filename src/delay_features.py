"""Features for the antibiotic-delay model, with a guard against data leakage.

Prediction question
    "At the moment a patient is registered in the ER, how likely is it that their first antibiotics will be
    given MORE than one hour after registration?"

Two prediction points are supported (`stage`): "registration" (the moment the patient is registered) and
"triage" (the moment sepsis triage is completed, for patients not yet treated). At each point only information
that already exists may be used. At registration:

    allowed     age; arrival hour, night and weekend arrival; how many other patients registered in the previous
                1 and 6 hours (ER load); the registering ER team; the clinical presentation flags that this dataset
                stores on the ER Registration event (suspected infection, SIRS criteria, organ dysfunction,
                hypotension, hypoxia, oliguria)
    forbidden   anything recorded later: lab tests and their values, IV fluid, the diagnosis code, discharge, plus the
                test-order flags (Diagnostic*) and the infusion flag, which describe decisions or treatment rather
                than presentation
At sepsis triage the same features plus what happened strictly BEFORE the triage was completed: minutes to triage and
to sepsis triage, lab tests already measured (and their values) and IV fluid already given. Nothing at or after the
sepsis-triage time, and no antibiotic information, is used.

`assert_no_leakage` fails loudly if a forbidden column ever reaches the model.

Patients without any recorded antibiotics have no administration time, so they are not part of this question
(the dashboard reports how many are excluded).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from load_data import ACTIVITY, CASE_ID, TIMESTAMP

REGISTRATION, ANTIBIOTICS = "ER Registration", "IV Antibiotics"
TARGET_HOURS = 1.0
GROUP = "org:group"

# model column -> (label shown to users, source column in the log or None if derived)
FEATURES = {
    "age": ("Age", "Age"),
    "arrival_hour": ("Hour of arrival", None),
    "is_night": ("Night arrival (22:00-05:59)", None),
    "is_weekend": ("Weekend arrival", None),
    "er_load_1h": ("Patients registered in the previous hour", None),
    "er_load_6h": ("Patients registered in the previous 6 hours", None),
    "registered_by_team_l": ("Registered by ER team L", GROUP),
    "infection_suspected": ("Infection suspected", "InfectionSuspected"),
    "sirs_tachypnea": ("SIRS: fast breathing", "SIRSCritTachypnea"),
    "sirs_heart_rate": ("SIRS: high heart rate", "SIRSCritHeartRate"),
    "sirs_temperature": ("SIRS: abnormal temperature", "SIRSCritTemperature"),
    "sirs_leucocytes": ("SIRS: abnormal white cell count", "SIRSCritLeucos"),
    "sirs_2_or_more": ("Two or more SIRS criteria", "SIRSCriteria2OrMore"),
    "organ_dysfunction": ("Organ dysfunction", "DisfuncOrg"),
    "hypotension": ("Low blood pressure", "Hypotensie"),
    "hypoxia": ("Low blood oxygen", "Hypoxie"),
    "oliguria": ("Low urine output", "Oligurie"),
}
TRIAGE_FEATURES = {
    "minutes_to_triage": ("Minutes from registration to triage", None),
    "minutes_to_sepsis_triage": ("Minutes from registration to sepsis triage", None),
    "lactate_measured_before": ("Lactate already measured", None),
    "lactate_value": ("Lactate value (0 if not measured yet)", None),
    "crp_measured_before": ("CRP already measured", None),
    "crp_value": ("CRP value (0 if not measured yet)", None),
    "leucocytes_measured_before": ("White cell count already measured", None),
    "leucocytes_value": ("White cell count (0 if not measured yet)", None),
    "fluid_before_triage": ("IV fluid already given", None),
}
STAGES = {"registration": "At ER registration", "triage": "At completion of sepsis triage"}
STAGE_FEATURES = {"registration": list(FEATURES), "triage": list(FEATURES) + list(TRIAGE_FEATURES)}
FEATURE_COLUMNS = STAGE_FEATURES["registration"]  # kept for callers that only know the registration stage
NUMERIC_FEATURES = {"age", "arrival_hour", "er_load_1h", "er_load_6h", "minutes_to_triage", "minutes_to_sepsis_triage",
                    "lactate_value", "crp_value", "leucocytes_value"}
BINARY_FEATURES = [c for c in STAGE_FEATURES["triage"] if c not in NUMERIC_FEATURES]
LABELS = {column: label for column, (label, _) in {**FEATURES, **TRIAGE_FEATURES}.items()}

# Columns that must never be used as inputs (they are recorded after registration or describe treatment).
FORBIDDEN = {
    "hours_to_antibiotics", "delayed", "hours_to_triage", "hours_to_sepsis_triage", "hours_to_fluid", "hours_to_lactate",
    "Infusion", "Diagnose", "DiagnosticBlood", "DiagnosticArtAstrup", "DiagnosticIC", "DiagnosticSputum", "DiagnosticLiquor",
    "DiagnosticOther", "DiagnosticUrinaryCulture", "DiagnosticUrinarySediment", "DiagnosticECG", "DiagnosticLacticAcid",
    "DiagnosticXthorax", "CRP", "LacticAcid", "Leucocytes", "outcome", "duration_days", "returned_to_er",
}


class LeakageError(ValueError):
    """A column that is not known at registration was offered to the model."""


def assert_no_leakage(columns, stage: str = "registration") -> None:
    bad = sorted(set(columns) & FORBIDDEN)
    unknown = sorted(set(columns) - set(STAGE_FEATURES[stage]))
    if bad:
        raise LeakageError(f"Columns not known at registration were used as features: {bad}")
    if unknown:
        raise LeakageError(f"Columns that are not declared features were used: {unknown}")


@dataclass(frozen=True)
class Dataset:
    table: pd.DataFrame  # one row per patient: id, registered, features, hours_to_antibiotics, delayed
    n_patients: int
    n_with_antibiotics: int
    n_without_antibiotics: int
    stage: str = "registration"

    @property
    def feature_columns(self) -> list:
        return STAGE_FEATURES[self.stage]

    @property
    def n_excluded_at_stage(self) -> int:
        """Treated patients who cannot be predicted at this stage (already treated before the prediction point)."""
        return self.n_with_antibiotics - len(self.table)

    @property
    def X(self) -> pd.DataFrame:
        assert_no_leakage(self.feature_columns, self.stage)
        return self.table[self.feature_columns]

    @property
    def y(self) -> pd.Series:
        return self.table["delayed"]


def registration_load(registered: pd.Series, window_hours: float) -> np.ndarray:
    """Number of OTHER patients registered in the `window_hours` before each registration (no future information)."""
    times = (registered - registered.min()).dt.total_seconds().to_numpy() / 3600
    order = np.sort(times)
    before = np.searchsorted(order, times, side="left")  # registrations strictly before this one
    window_start = np.searchsorted(order, times - window_hours, side="left")
    return before - window_start


def _triage_features(log: pd.DataFrame, table: pd.DataFrame, registered: pd.Series) -> pd.DataFrame:
    """Extra features known at the moment sepsis triage is completed (only events strictly before it)."""
    first = log.groupby([CASE_ID, ACTIVITY])[TIMESTAMP].min().unstack()
    triage, sepsis_triage = first["ER Triage"], first["ER Sepsis Triage"]
    cut = sepsis_triage.reindex(table.index)
    out = pd.DataFrame(index=table.index)
    out["minutes_to_triage"] = (triage.reindex(table.index) - registered).dt.total_seconds() / 60
    out["minutes_to_sepsis_triage"] = (cut - registered).dt.total_seconds() / 60
    for activity, column in (("LacticAcid", "lactate"), ("CRP", "crp"), ("Leucocytes", "leucocytes")):
        events = log[log[ACTIVITY] == activity][[CASE_ID, TIMESTAMP, activity]].merge(cut.rename("cut"), left_on=CASE_ID, right_index=True)
        events = events.reset_index(drop=True)  # no index level named like a column, which pandas treats as ambiguous
        before = events[events[TIMESTAMP] < events["cut"]].sort_values(TIMESTAMP).groupby(CASE_ID)[activity].first()
        out[f"{column}_measured_before"] = out.index.isin(before.index).astype(int)
        out[f"{column}_value"] = before.reindex(out.index).fillna(0).astype(float)
    fluid = log[log[ACTIVITY] == "IV Liquid"].groupby(CASE_ID)[TIMESTAMP].min().reindex(table.index)
    out["fluid_before_triage"] = (fluid < cut).fillna(False).astype(int)
    return out


def build_dataset(log: pd.DataFrame, stage: str = "registration") -> Dataset:
    """Build one row per patient from the cleaned log (registration first in every case)."""
    if stage not in STAGES:
        raise ValueError(f"Unknown prediction stage {stage!r}; use one of {list(STAGES)}.")
    log = log.sort_values([CASE_ID, TIMESTAMP], kind="stable")
    registration = log[log[ACTIVITY] == REGISTRATION].groupby(CASE_ID).first()
    antibiotics = log[log[ACTIVITY] == ANTIBIOTICS].groupby(CASE_ID)[TIMESTAMP].min()
    if registration.empty:
        raise ValueError("The log has no ER Registration events.")

    registered = registration[TIMESTAMP]
    table = pd.DataFrame(index=registration.index)
    table["registered"] = registered
    table["age"] = registration["Age"].astype(float)
    hour = registered.dt.hour
    table["arrival_hour"] = hour
    table["is_night"] = ((hour >= 22) | (hour < 6)).astype(int)
    table["is_weekend"] = (registered.dt.dayofweek >= 5).astype(int)
    table["er_load_1h"] = registration_load(registered, 1)
    table["er_load_6h"] = registration_load(registered, 6)
    table["registered_by_team_l"] = (registration[GROUP] == "L").astype(int)
    for column, (_, source) in FEATURES.items():
        if source and column not in table.columns:
            table[column] = registration[source].astype(float).fillna(0).astype(int)
    if stage == "triage":
        table = table.join(_triage_features(log, table, registered))
    table["hours_to_antibiotics"] = (antibiotics.reindex(table.index) - registered).dt.total_seconds() / 3600
    table["delayed"] = (table["hours_to_antibiotics"] > TARGET_HOURS).astype(int)

    # patients with antibiotics recorded at or after registration (earlier ones cannot be predicted)
    treated = table[table["hours_to_antibiotics"].notna() & (table["hours_to_antibiotics"] >= 0)]
    n_with_antibiotics = len(treated)
    if stage == "triage":  # still untreated when sepsis triage is completed
        first = log.groupby([CASE_ID, ACTIVITY])[TIMESTAMP].min().unstack()
        treated = treated[antibiotics.reindex(treated.index) > first["ER Sepsis Triage"].reindex(treated.index)]
    treated = treated.rename_axis(CASE_ID).reset_index()
    assert treated[CASE_ID].is_unique  # one row per patient, so any row-level split is a patient-level split
    return Dataset(treated, len(table), n_with_antibiotics, len(table) - n_with_antibiotics, stage)


def feature_row(values: dict, stage: str = "registration") -> pd.DataFrame:
    """One-row feature table from user-supplied values (missing values default to 0)."""
    columns = STAGE_FEATURES[stage]
    row = {column: values.get(column, 0) for column in columns}
    assert_no_leakage(row, stage)
    return pd.DataFrame([row], columns=columns)


def describe_value(column: str, value) -> str:
    """Human wording for a feature value, for explanations."""
    if column in BINARY_FEATURES:
        return "yes" if value else "no"
    if column == "age":
        return f"{value:.0f} years"
    if column == "arrival_hour":
        return f"{int(value):02d}:00"
    if column.startswith("minutes_"):
        return f"{value:.0f} min"
    return f"{value:.1f}" if column in NUMERIC_FEATURES else f"{value:.0f}"
