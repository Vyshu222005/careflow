import numpy as np
import pandas as pd
import pytest

import delay_model
import explain
import insights
import report
from delay_features import LABELS, build_dataset


def healthy_metrics() -> dict:
    """Metrics for a well-performing unit: no rule should fire."""
    return {
        "n_patients": 500, "abx_treated": 450, "abx_untreated": 50, "abx_untreated_pct": 10.0, "abx_median_hours": 0.6, "abx_within_1h": 360, "abx_within_1h_pct": 80.0,
        "missing_discharge_pct": 5.0, "conformance_pct": 80.0, "avg_fitness": 0.97, "top_deviation": None, "repeated_ward_patients": 20, "repeated_ward_pct": 4.0,
        "repeated_ward_records": 25, "ward_transfer_share_pct": 40.0, "released": 400, "returned": 20, "return_30d": 10, "return_30d_pct": 2.5,
        "bundle_complete_bundle_pct": 70.0, "bundle_complete_bundle_n": 500, "bundle_lactate_1h_pct": 90.0, "bundle_lactate_1h_n": 500,
        "bundle_antibiotics_1h_pct": 80.0, "bundle_antibiotics_1h_n": 500, "bundle_repeat_lactate_pct": 90.0, "bundle_repeat_lactate_n": 100,
        "bundle_fluid_3h_pct": 90.0, "bundle_fluid_3h_n": 30, "control_signals": [], "control_months": 0, "top_wait": ("A", "B", 0.5, 100),
        "direct_to_antibiotics": (0.5, 100), "triage_to_fluid": (0.4, 100), "lab_to_discharge": (6.0, 100),
    }


def ids(metrics: dict) -> set:
    return {i.id for i in insights.generate(metrics)}


def test_no_rule_fires_for_healthy_metrics():
    assert ids(healthy_metrics()) == set()


@pytest.mark.parametrize("change, expected", [
    ({"abx_within_1h_pct": 30.0}, "slow_antibiotics"),
    ({"abx_untreated_pct": 30.0}, "no_antibiotics"),
    ({"missing_discharge_pct": 40.0}, "missing_discharge"),
    ({"conformance_pct": 20.0}, "low_conformance"),
    ({"repeated_ward_pct": 40.0}, "ward_records"),
    ({"return_30d_pct": 25.0}, "returns"),
    ({"bundle_complete_bundle_pct": 10.0}, "low_bundle"),
    ({"bundle_repeat_lactate_pct": 20.0}, "repeat_lactate"),
    ({"lab_to_discharge": (40.0, 100)}, "lab_to_discharge"),
    ({"top_wait": ("X", "Y", 3.5, 80)}, "slow_er_step"),
    ({"direct_to_antibiotics": (2.0, 100), "triage_to_fluid": (0.4, 100)}, "direct_antibiotics"),
    ({"control_months": 12}, "control_stable"),
    ({"control_months": 12, "control_signals": [("2014-01", "above the limit", 46.0)]}, "control_signal"),
])
def test_each_rule_fires_on_its_condition(change, expected):
    assert expected in ids({**healthy_metrics(), **change})


def test_severity_and_ordering_and_threshold_boundary():
    bad = {**healthy_metrics(), "abx_within_1h_pct": 20.0, "missing_discharge_pct": 40.0, "conformance_pct": 20.0}
    found = insights.generate(bad)
    assert found[0].severity == "High" and [i.severity for i in found] == sorted((i.severity for i in found), key=insights.SEVERITY_ORDER.get)
    edge = insights.THRESHOLDS["antibiotics_within_1h_low_pct"]
    assert "slow_antibiotics" not in ids({**healthy_metrics(), "abx_within_1h_pct": float(edge)})  # not strictly below the threshold


def test_every_insight_is_explained_and_traceable():
    for item in insights.generate({**healthy_metrics(), "abx_within_1h_pct": 20.0, "missing_discharge_pct": 40.0, "conformance_pct": 20.0}):
        assert item.finding and item.rule and item.explanation and item.action and item.evidence
        # suggestions are about reviewing the process, never prescriptions
        assert not any(word in item.action.lower() for word in ("prescribe", "administer", "dose", "should give", "must give", "start treatment"))


def test_small_samples_switch_rules_off_and_say_so():
    small = {**healthy_metrics(), "n_patients": 10, "abx_treated": 8, "abx_within_1h_pct": 10.0, "conformance_pct": 5.0, "missing_discharge_pct": 90.0}
    assert ids(small) == {"small_sample"}


# ---------------------------------------------------------------- report
def test_ascii_text_handles_symbols_and_unknown_characters():
    assert report.ascii_text("A → B – C ≥ 2") == "A  to  B - C >= 2"
    assert report.ascii_text("emoji \U0001F600").startswith("emoji")


def test_pdf_builds_with_minimal_and_full_data():
    data = report.ReportData(title="Test report", filters_text="all patients", counts_text="counts", kpis=[("Patients", "10", "of 10")])
    pdf = report.build_pdf(data)
    assert pdf.startswith(b"%PDF") and len(pdf) > 1500
    full = report.ReportData(
        title="Full", filters_text="age: 80+", counts_text="counts", kpis=[("Patients", "10", "of 10")] * 6,
        insights=insights.generate({**healthy_metrics(), "abx_within_1h_pct": 20.0}),
        pathways=pd.DataFrame({"Pathway": ["A to B"], "Patients": [5], "Share %": [50.0], "Median days": [1.0]}),
        waits=pd.DataFrame({"Step": ["A to B"], "Patients": [5], "Median wait": ["1.0 h"], "90th percentile": ["2.0 h"], "median_hours": [1.0]}),
        scorecard=pd.DataFrame({"label": ["x"], "met_pct": [50.0]}), rework_lines=["line one", "line → two"],
        model={"stage_label": "At ER registration", "best_model": "XGBoost", "verdict": "weak signal", "base_rate": 0.67,
               "metrics": pd.DataFrame({"Model": ["m"], "ROC-AUC": [0.58]}), "importance": pd.DataFrame({"label": ["Age"], "share_pct": [50.0]})},
        simulation={"scenario_text": "Both improvements", "summary": pd.DataFrame({"Metric": ["m"], "Recorded": [1.0], "Hypothetical": [2.0], "Change": [1.0]}),
                    "newly_within_one_hour": 3, "patients": 10},
    )
    assert report.build_pdf(full).startswith(b"%PDF")


# ---------------------------------------------------------------- explanations
def test_shap_shapes_and_the_planted_feature_ranks_first(synthetic_log):
    model = delay_model.train(synthetic_log, "registration")
    X = model.dataset.X
    values = explain.shap_matrix(model.model, X)
    assert values.shape == X.shape and not np.isnan(values).any()
    importance = explain.global_importance(values, X)
    assert importance.iloc[0]["feature"] == "age" and importance["share_pct"].sum() == pytest.approx(100)
    assert importance.iloc[0]["direction"] > 0  # older patients are treated later in the synthetic data


@pytest.mark.parametrize("name", ["Logistic regression", "Random forest", "XGBoost"])
def test_shap_works_for_every_model_type(synthetic_log, name):
    ds = build_dataset(synthetic_log)
    model = delay_model.make_models(ds.y)[name].fit(ds.X, ds.y)
    assert explain.shap_matrix(model, ds.X).shape == ds.X.shape


def test_explanation_sentences_name_features_and_direction(synthetic_log):
    ds = build_dataset(synthetic_log)
    model = delay_model.make_models(ds.y)["XGBoost"].fit(ds.X, ds.y)
    values = explain.shap_matrix(model, ds.X)
    top = explain.contributions(values[0], ds.X.iloc[0], top=3)
    assert len(top) == 3 and set(top["feature"]) <= set(LABELS)
    sentences = explain.explain_in_words(top, 0.7, 0.67)
    assert len(sentences) == 4 and "70%" in sentences[0] and all(("raises" in s or "lowers" in s) for s in sentences[1:])
