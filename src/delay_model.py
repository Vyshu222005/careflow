"""Antibiotic-delay prediction: train, validate honestly, and score patients.

Question: will the first antibiotics be given more than 1 hour after ER registration? It is asked at two
prediction points ("registration" and "triage"); see delay_features.py for what is allowed as an input at each.

Validation (nothing is tuned on the test data and no result is invented)
    * 5-fold stratified cross-validation over patients (every row is one patient, so folds are patient-level);
      out-of-fold probabilities give the headline metrics
    * a stratified 80/20 hold-out split, evaluated once
    * a temporal check: train on the earliest 70% of patients, test on the latest 30%
    * a baseline that always predicts "delayed", so the models are judged against doing nothing clever

Class imbalance is handled with class weights in every model. Metrics: precision, recall, F1 (threshold 0.5),
ROC-AUC and PR-AUC. The model with the best cross-validated ROC-AUC is chosen as the final model and refitted on
all patients.

Usage:
    python src/delay_model.py

Outputs:
    reports/delay_model_evaluation.csv   every metric for every model and validation scheme
    reports/delay_model_risk_groups.csv  observed delay rate in the lower/medium/higher-risk groups
"""
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from delay_features import STAGES, Dataset, assert_no_leakage, build_dataset

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_PATH = ROOT / "reports" / "delay_model_evaluation.csv"
RISK_GROUPS_PATH = ROOT / "reports" / "delay_model_risk_groups.csv"

SEED = 42
THRESHOLD = 0.5
CV_FOLDS = 5
TEST_SIZE = 0.2
TEMPORAL_TRAIN_SHARE = 0.7
MIN_PATIENTS = 100
RISK_LABELS = ("Lower risk", "Medium risk", "Higher risk")
BASELINE = "Baseline: always predict delay"


class ModelError(ValueError):
    """The data cannot support model training (too few patients or only one outcome)."""


def make_models(y: pd.Series) -> dict:
    """Fresh, untrained models. Class weights handle the imbalance."""
    positives = max(int(y.sum()), 1)
    negatives = max(int((1 - y).sum()), 1)
    return {
        "Logistic regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))]),
        "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=8, class_weight="balanced_subsample", random_state=SEED, n_jobs=1),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                                 scale_pos_weight=negatives / positives, eval_metric="logloss", random_state=SEED, n_jobs=1, verbosity=0),
    }


def metrics(y_true, proba, threshold: float = THRESHOLD) -> dict:
    predicted = (np.asarray(proba) >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if len(set(y_true)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, proba) if len(set(y_true)) > 1 else np.nan,
    }


def baseline_metrics(y_true) -> dict:
    """Always predict 'delayed': what you get without a model."""
    return metrics(y_true, np.ones(len(y_true)))


def cross_validated(X: pd.DataFrame, y: pd.Series, name: str) -> tuple:
    """Out-of-fold probabilities and fold-wise ROC-AUC for one model."""
    folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    proba = np.zeros(len(y))
    fold_auc = []
    for train, test in folds.split(X, y):
        model = make_models(y.iloc[train])[name]
        model.fit(X.iloc[train], y.iloc[train])
        proba[test] = model.predict_proba(X.iloc[test])[:, 1]
        fold_auc.append(roc_auc_score(y.iloc[test], proba[test]))
    return proba, fold_auc


def evaluate(dataset: Dataset) -> tuple:
    """Run every validation scheme. Returns (evaluation table, out-of-fold probabilities per model)."""
    table, X, y = dataset.table, dataset.X, dataset.y
    rows, oof = [], {}

    def add(model, scheme, result, n_test, extra=None):
        rows.append({"model": model, "validation": scheme, "patients_tested": n_test, **result, **(extra or {})})

    add(BASELINE, "5-fold cross-validation", baseline_metrics(y), len(y))
    for name in make_models(y):
        proba, fold_auc = cross_validated(X, y, name)
        oof[name] = proba
        add(name, "5-fold cross-validation", metrics(y, proba), len(y), {"roc_auc_sd_over_folds": float(np.std(fold_auc))})

    train, test = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, stratify=y, random_state=SEED)
    add(BASELINE, "80/20 hold-out", baseline_metrics(y.iloc[test]), len(test))
    for name, model in make_models(y.iloc[train]).items():
        model.fit(X.iloc[train], y.iloc[train])
        add(name, "80/20 hold-out", metrics(y.iloc[test], model.predict_proba(X.iloc[test])[:, 1]), len(test))

    order = np.argsort(table["registered"].to_numpy())
    cut = int(len(order) * TEMPORAL_TRAIN_SHARE)
    train, test = order[:cut], order[cut:]
    add(BASELINE, "Temporal (earliest 70% to latest 30%)", baseline_metrics(y.iloc[test]), len(test))
    for name, model in make_models(y.iloc[train]).items():
        model.fit(X.iloc[train], y.iloc[train])
        add(name, "Temporal (earliest 70% to latest 30%)", metrics(y.iloc[test], model.predict_proba(X.iloc[test])[:, 1]), len(test))
    return pd.DataFrame(rows), oof


def risk_thresholds(oof_proba: np.ndarray) -> tuple:
    """Probabilities that split patients into thirds: lower, medium and higher risk."""
    low, high = np.quantile(oof_proba, [1 / 3, 2 / 3])
    return float(low), float(high)


def risk_group(probability: float, thresholds: tuple) -> str:
    low, high = thresholds
    return RISK_LABELS[0] if probability < low else RISK_LABELS[2] if probability >= high else RISK_LABELS[1]


def risk_group_table(y: pd.Series, oof_proba: np.ndarray, thresholds: tuple) -> pd.DataFrame:
    """Observed delay rate in each risk group, from out-of-fold predictions (the honest check of the groups)."""
    groups = pd.Series([risk_group(p, thresholds) for p in oof_proba], index=y.index)
    table = pd.DataFrame({"group": groups, "delayed": y}).groupby("group")["delayed"].agg(patients="size", delayed_patients="sum")
    table["observed_delay_rate_pct"] = table["delayed_patients"] / table["patients"] * 100
    return table.reindex(RISK_LABELS).reset_index()


@dataclass
class DelayModel:
    dataset: Dataset
    evaluation: pd.DataFrame
    best_name: str
    model: object  # fitted on all patients
    oof_proba: np.ndarray
    thresholds: tuple
    risk_groups: pd.DataFrame

    @property
    def stage(self) -> str:
        return self.dataset.stage

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert_no_leakage(X.columns, self.stage)
        return self.model.predict_proba(X[self.dataset.feature_columns])[:, 1]

    def classify(self, probability: float) -> str:
        return risk_group(probability, self.thresholds)

    @property
    def base_rate(self) -> float:
        return float(self.dataset.y.mean())


def train(log: pd.DataFrame, stage: str = "registration") -> DelayModel:
    """Build the dataset, validate all models, pick the best by cross-validated ROC-AUC, refit on all patients."""
    dataset = build_dataset(log, stage)
    y = dataset.y
    if len(y) < MIN_PATIENTS:
        raise ModelError(f"Only {len(y)} patients have antibiotics recorded; at least {MIN_PATIENTS} are needed to train a model.")
    if y.nunique() < 2 or y.value_counts().min() < CV_FOLDS * 2:
        raise ModelError("The outcome has too few examples of one class to validate a model.")
    evaluation, oof = evaluate(dataset)
    cv = evaluation[(evaluation["validation"] == "5-fold cross-validation") & (evaluation["model"] != BASELINE)]
    best = cv.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    model = make_models(y)[best]
    model.fit(dataset.X, y)
    thresholds = risk_thresholds(oof[best])
    return DelayModel(dataset, evaluation, best, model, oof[best], thresholds, risk_group_table(y, oof[best], thresholds))


CACHE_DIR = ROOT / "data" / "processed"


def _cache_path(dataset: Dataset) -> Path:
    """Cache file named after the data and library versions, so a stale model is never loaded."""
    import sklearn
    import xgboost

    key = joblib.hash((dataset.table, dataset.stage, sklearn.__version__, xgboost.__version__, SEED, CV_FOLDS))
    return CACHE_DIR / f"delay_model_{dataset.stage}_{key[:12]}.joblib"


def load_or_train(log: pd.DataFrame, stage: str = "registration") -> DelayModel:
    """Train (about 20 seconds) or load the identical trained model from a cache file in data/processed."""
    dataset = build_dataset(log, stage)
    path = _cache_path(dataset)
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:  # noqa: BLE001 - a damaged cache file just means we train again
            path.unlink(missing_ok=True)
    result = train(log, stage)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for old in CACHE_DIR.glob(f"delay_model_{stage}_*.joblib"):
        old.unlink(missing_ok=True)
    joblib.dump(result, path)
    return result


def main():
    from conformance import load_log

    log = load_log()
    tables, groups = [], []
    for stage, title in STAGES.items():
        result = load_or_train(log, stage)
        d = result.dataset
        print(f"\n=== Prediction point: {title} ===")
        print(f"Patients: {d.n_patients} | with antibiotics: {d.n_with_antibiotics} | modelled at this point: {len(d.table)} "
              f"(already treated before this point: {d.n_excluded_at_stage})")
        print(f"Delayed (>1 h) rate among modelled patients: {result.base_rate * 100:.1f}%")
        show = result.evaluation.copy()
        for column in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
            show[column] = show[column].round(3)
        print(show[["model", "validation", "patients_tested", "precision", "recall", "f1", "roc_auc", "pr_auc"]].to_string(index=False))
        print(f"Best model by cross-validated ROC-AUC: {result.best_name}")
        print(result.risk_groups.round(1).to_string(index=False))
        tables.append(result.evaluation.assign(stage=stage))
        groups.append(result.risk_groups.assign(stage=stage, model=result.best_name))
    EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables).round(4).to_csv(EVALUATION_PATH, index=False)
    pd.concat(groups).round(2).to_csv(RISK_GROUPS_PATH, index=False)
    print(f"\nSaved: {EVALUATION_PATH}\n       {RISK_GROUPS_PATH}")


if __name__ == "__main__":
    main()
