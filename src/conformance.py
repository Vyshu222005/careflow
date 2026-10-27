"""Conformance checking: how closely do patients follow an expected sepsis pathway?

Two complementary views, both on the clinical-steps log (repeated lab tests left out):

    formal      every case is aligned to an expected process model; the alignment gives a
                fitness score (1.0 = the case follows the model exactly) and lists the steps
                that are missing or unexpected
    rule-based  plain yes/no checks that explain the deviations (no antibiotics, treatment
                before sepsis triage, no discharge, repeated steps, ...)

The expected pathway is an ASSUMPTION built from the standard route found in the data and the
usual sepsis care sequence; it should be confirmed with the mentor. Change EXPECTED_PATHWAY to
test another definition.

Usage:
    python src/conformance.py

Outputs:
    data/processed/sepsis_conformance.csv   one row per case
    reports/conformance_summary.csv         the deviations and how common each is
    reports/conformance_patterns.csv        the most common deviation patterns
"""
from collections import Counter
from pathlib import Path

import pandas as pd
import pm4py

from clean_data import load_clean_log
from discovery import milestone_log
from load_data import ACTIVITY, CASE_ID, TIMESTAMP
from prepare_log import order_events

ROOT = Path(__file__).resolve().parents[1]
CONFORMANCE_PATH = ROOT / "data" / "processed" / "sepsis_conformance.csv"
SUMMARY_PATH = ROOT / "reports" / "conformance_summary.csv"
PATTERNS_PATH = ROOT / "reports" / "conformance_patterns.csv"

# Process-tree notation: ->( ) is a sequence, +( ) is parallel, X( ) is a choice, tau is "skip".
EXPECTED_PATHWAY = (
    "->( 'ER Registration', 'ER Triage', 'ER Sepsis Triage', "
    "+( X('IV Liquid', tau), 'IV Antibiotics' ), "
    "X( 'Admission NC', ->('Admission IC', 'Admission NC') ), "
    "X('Release A', 'Release B', 'Release C', 'Release D', 'Release E'), "
    "X('Return ER', tau) )"
)
ANY_RELEASE = "Release (any)"
ANY_ADMISSION = "Admission (any)"
ANTIBIOTICS_TARGET_HOURS = 1
FIT_TOLERANCE = 1e-9

# Plain-language names for the rule-based checks
RULES = {
    "no_antibiotics": "No antibiotics recorded",
    "late_antibiotics": "Antibiotics later than 1 hour after registration",
    "treated_before_sepsis_triage": "Treatment before sepsis triage",
    "no_admission": "No ward or ICU admission recorded",
    "no_discharge": "No discharge recorded",
    "repeated_step": "A clinical step is repeated",
}


def expected_model():
    """Petri net of the expected pathway: (net, initial marking, final marking)."""
    return pm4py.convert_to_petri_net(pm4py.parse_process_tree(EXPECTED_PATHWAY))


def _generic(label: str) -> str:
    """Give interchangeable labels one name (the aligner picks among them arbitrarily)."""
    if label.startswith("Release"):
        return ANY_RELEASE
    if label.startswith("Admission"):
        return ANY_ADMISSION
    return label


def align_cases(steps: pd.DataFrame) -> pd.DataFrame:
    """Align every case to the expected model. Returns fitness and missing/unexpected steps per case."""
    log = pm4py.convert_to_event_log(steps[[CASE_ID, ACTIVITY, TIMESTAMP]])
    net, initial, final = expected_model()
    results = pm4py.conformance_diagnostics_alignments(log, net, initial, final)
    rows = []
    for trace, result in zip(log, results):
        moves = result["alignment"]
        missing = [_generic(model) for log_label, model in moves if log_label == ">>" and model is not None]
        extra = [log_label for log_label, model in moves if model == ">>"]
        rows.append({
            CASE_ID: trace.attributes["concept:name"],
            "fitness": result["fitness"],
            "missing_steps": "; ".join(missing),
            "unexpected_steps": "; ".join(extra),
            "n_missing": len(missing),
            "n_unexpected": len(extra),
        })
    table = pd.DataFrame(rows)
    table["conforms"] = table["fitness"] >= 1 - FIT_TOLERANCE
    return table


def rule_checks(steps: pd.DataFrame) -> pd.DataFrame:
    """Yes/no deviation checks per case (True = the deviation occurs)."""
    first = steps.groupby([CASE_ID, ACTIVITY])[TIMESTAMP].min().unstack()
    cases = pd.DataFrame(index=first.index)
    start = first["ER Registration"]
    antibiotics = first["IV Antibiotics"]
    fluid = first["IV Liquid"]
    sepsis_triage = first["ER Sepsis Triage"]
    release_cols = [c for c in first.columns if c.startswith("Release")]
    admission_cols = [c for c in first.columns if c.startswith("Admission")]

    cases["no_antibiotics"] = antibiotics.isna()
    hours = (antibiotics - start).dt.total_seconds() / 3600
    cases["late_antibiotics"] = hours > ANTIBIOTICS_TARGET_HOURS
    first_treatment = pd.concat([antibiotics, fluid], axis=1).min(axis=1)
    cases["treated_before_sepsis_triage"] = first_treatment < sepsis_triage
    cases["no_admission"] = first[admission_cols].isna().all(axis=1)
    cases["no_discharge"] = first[release_cols].isna().all(axis=1)
    counts = steps.groupby(CASE_ID)[ACTIVITY].agg(lambda s: len(s) - s.nunique())
    cases["repeated_step"] = counts.reindex(cases.index) > 0
    return cases.reset_index()


def check_conformance(events: pd.DataFrame) -> pd.DataFrame:
    """One row per case: formal fitness, missing/unexpected steps and the rule-based flags."""
    steps = milestone_log(events)
    table = align_cases(steps).merge(rule_checks(steps), on=CASE_ID)
    table["any_rule_deviation"] = table[list(RULES)].any(axis=1)
    return table


def deviation_summary(table: pd.DataFrame) -> pd.DataFrame:
    """How common each deviation is, as share of all cases."""
    n = len(table)
    rows = [{"deviation": RULES[key], "kind": "rule", "cases": int(table[key].sum())} for key in RULES]
    rows.append({"deviation": "Does not follow the expected pathway exactly", "kind": "formal", "cases": int((~table["conforms"]).sum())})
    out = pd.DataFrame(rows)
    out["share_pct"] = out["cases"] / n * 100
    return out.sort_values(["kind", "cases"], ascending=[True, False]).reset_index(drop=True)


def describe_pattern(missing: str, unexpected: str) -> str:
    """Readable text for a deviation, e.g. 'missing IV Antibiotics and extra Admission NC'."""
    def names(joined: str) -> str:
        counts = Counter(joined.split("; "))
        return ", ".join(f"{name} (x{n})" if n > 1 else name for name, n in counts.items())

    parts = []
    if missing:
        parts.append("missing " + names(missing))
    if unexpected:
        parts.append("extra " + names(unexpected))
    return " and ".join(parts)


def deviation_patterns(table: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """The most common combinations of missing and unexpected steps."""
    deviating = table[~table["conforms"]].copy()
    deviating["description"] = [describe_pattern(m, u) for m, u in zip(deviating["missing_steps"], deviating["unexpected_steps"])]
    counts = deviating["description"].value_counts().head(top).rename_axis("description").reset_index(name="cases")
    counts["share_of_all_pct"] = counts["cases"] / len(table) * 100
    return counts


def load_log() -> pd.DataFrame:
    """The cleaned log in a consistent event order. The rare-activity filter of prepare_log.py is not
    applied, so every discharge code (including Release E) is kept."""
    return order_events(load_clean_log())


def main():
    table = check_conformance(load_log())
    CONFORMANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(CONFORMANCE_PATH, index=False)
    summary, patterns = deviation_summary(table), deviation_patterns(table)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.round(2).to_csv(SUMMARY_PATH, index=False)
    patterns.round(2).to_csv(PATTERNS_PATH, index=False)

    print(f"Cases checked          : {len(table)}")
    print(f"Follow the expected path: {table['conforms'].sum()} ({table['conforms'].mean() * 100:.1f}%)")
    print(f"Average fitness        : {table['fitness'].mean():.3f}")
    print(summary.round(1).to_string(index=False))
    print(f"Saved: {CONFORMANCE_PATH}\n       {SUMMARY_PATH}\n       {PATTERNS_PATH}")
    return table


if __name__ == "__main__":
    main()
