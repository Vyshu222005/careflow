import pandas as pd
import pytest

from validation import DataError, selected_counts, stage_counts, validate_frame

C, A, T = "case:concept:name", "concept:name", "time:timestamp"


def frame(rows):
    return pd.DataFrame(rows, columns=[C, A, T, "org:group", "Age"])


def test_stage_counts_explains_difference():
    clean = frame([("a", "ER Registration", 1, "A", 50), ("a", "Release E", 2, "E", 50), ("b", "ER Registration", 1, "A", 60), ("b", "Release A", 2, "E", 60)])
    prepared = clean[clean[A] != "Release E"]
    counts = stage_counts(clean, prepared)
    assert counts.cleaned_events == 4 and counts.analysis_events == 3
    assert counts.excluded_events == 1 and counts.excluded_activities == ("Release E",) and counts.excluded_patients == 1
    assert counts.patients == 2
    assert "Release E" in counts.explanation


def test_stage_counts_fails_when_events_are_lost_unexplained():
    clean = frame([("a", "ER Registration", 1, "A", 50), ("a", "ER Triage", 2, "C", 50), ("b", "ER Registration", 1, "A", 60), ("b", "ER Triage", 2, "C", 60)])
    prepared = clean.drop(index=3)  # patient b lost an event, yet every activity still exists: nothing explains the loss
    with pytest.raises(DataError):
        stage_counts(clean, prepared)


def test_validate_frame_names_missing_columns():
    with pytest.raises(DataError, match="missing columns"):
        validate_frame(pd.DataFrame({C: ["a"]}), [C, A], "The test table")


def test_validate_frame_rejects_missing_case_ids_and_empty_tables():
    with pytest.raises(DataError, match="without a case ID"):
        validate_frame(pd.DataFrame({C: [None, "a"], A: ["x", "y"]}), [C, A], "table")
    with pytest.raises(DataError, match="empty"):
        validate_frame(pd.DataFrame({C: [], A: []}), [C, A], "table")


def test_selected_counts():
    clean = frame([("a", "ER Registration", 1, "A", 50), ("b", "ER Registration", 1, "A", 60), ("b", "Release E", 2, "E", 60)])
    prepared = clean[clean[A] != "Release E"]
    assert selected_counts(clean, prepared, ["b"]) == (2, 1)
