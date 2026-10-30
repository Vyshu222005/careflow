import numpy as np
import pandas as pd
import pytest

import delay_model
from delay_features import FEATURE_COLUMNS, build_dataset

C = "case:concept:name"


@pytest.fixture(scope="module")
def trained(synthetic_log):
    return delay_model.train(synthetic_log, "registration")


def test_evaluation_has_every_model_scheme_and_metric(trained):
    ev = trained.evaluation
    assert set(ev["validation"]) == {"5-fold cross-validation", "80/20 hold-out", "Temporal (earliest 70% to latest 30%)"}
    assert set(ev["model"]) == {delay_model.BASELINE, "Logistic regression", "Random forest", "XGBoost"}
    for column in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert ev[column].between(0, 1).all()


def test_baseline_has_no_discrimination(trained):
    baseline = trained.evaluation[trained.evaluation["model"] == delay_model.BASELINE]
    assert (baseline["roc_auc"] == 0.5).all() and (baseline["recall"] == 1.0).all()


def test_model_finds_a_planted_signal(trained):
    cv = trained.evaluation[(trained.evaluation["validation"] == "5-fold cross-validation") & (trained.evaluation["model"] != delay_model.BASELINE)]
    assert cv["roc_auc"].max() > 0.75  # age drives the delay in the synthetic data
    assert trained.best_name in cv["model"].tolist()


def test_no_skill_on_pure_noise(noise_log):
    result = delay_model.train(noise_log, "registration")
    cv = result.evaluation[(result.evaluation["validation"] == "5-fold cross-validation") & (result.evaluation["model"] == result.best_name)].iloc[0]
    assert cv["roc_auc"] < 0.65  # honest: nothing to learn, so no strong score


def test_risk_groups_are_ordered_by_probability_and_cover_everyone(trained):
    low, high = trained.thresholds
    assert low <= high
    assert trained.classify(low - 1e-6) == "Lower risk" and trained.classify(high) == "Higher risk"
    assert trained.risk_groups["patients"].sum() == len(trained.dataset.table)
    rates = trained.risk_groups["observed_delay_rate_pct"].tolist()
    assert rates[0] < rates[-1]  # planted signal: higher predicted risk means more observed delay


def test_predict_returns_probabilities_and_rejects_leaky_input(trained):
    X = trained.dataset.X
    p = trained.predict(X.head(10))
    assert ((p >= 0) & (p <= 1)).all() and len(p) == 10
    leaky = X.head(3).assign(hours_to_antibiotics=1.0)
    with pytest.raises(Exception):
        trained.predict(leaky)


def test_cross_validation_is_patient_level(synthetic_log):
    ds = build_dataset(synthetic_log)
    assert ds.table[C].is_unique  # one row per patient, so any split of rows is a split of patients


def test_too_few_patients_or_one_class_is_refused(synthetic_log):
    few = synthetic_log[synthetic_log[C].isin(synthetic_log[C].unique()[:50])]
    with pytest.raises(delay_model.ModelError):
        delay_model.train(few, "registration")


def test_metrics_function_matches_hand_calculation():
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.4, 0.6, 0.1])
    m = delay_model.metrics(y, proba)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5) and m["roc_auc"] == pytest.approx(0.75)


def test_cache_round_trip(synthetic_log, tmp_path, monkeypatch):
    monkeypatch.setattr(delay_model, "CACHE_DIR", tmp_path)
    first = delay_model.load_or_train(synthetic_log, "registration")
    assert list(tmp_path.glob("delay_model_registration_*.joblib"))
    second = delay_model.load_or_train(synthetic_log, "registration")
    assert second.best_name == first.best_name
    assert np.allclose(second.predict(second.dataset.X), first.predict(first.dataset.X))
