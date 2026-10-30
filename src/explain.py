"""Explainable AI for the antibiotic-delay model, using SHAP.

SHAP splits a prediction into contributions, one per input: how much each feature pushed this patient's predicted
risk up or down compared with the average patient. The functions here return plain tables and sentences, so the
dashboard can show them without knowing anything about SHAP.

Units: for tree models the contributions are in the model's raw score (log-odds for XGBoost, probability for a random
forest); for logistic regression they are in log-odds of the standardised inputs. Only their sign and relative size
matter for the explanations. SHAP explains what the MODEL does; it does not show what causes a delay.
"""
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from delay_features import LABELS, describe_value

TOP_FEATURES = 5
MAX_BACKGROUND = 200


def _positive_class(values) -> np.ndarray:
    """SHAP output for the 'delayed' class as an (n_patients, n_features) array, whatever the library version returns."""
    if isinstance(values, list):  # older shap: one array per class
        values = values[1]
    values = np.asarray(values)
    return values[:, :, 1] if values.ndim == 3 else values


def shap_matrix(model, X: pd.DataFrame, background: pd.DataFrame | None = None) -> np.ndarray:
    """Contribution of every feature to every patient's predicted delay risk."""
    background = X if background is None else background
    background = background.sample(min(len(background), MAX_BACKGROUND), random_state=42)
    if isinstance(model, Pipeline):  # logistic regression: explain the linear model on standardised inputs
        scaler, classifier = model.steps[0][1], model.steps[-1][1]
        explainer = shap.LinearExplainer(classifier, scaler.transform(background))
        return _positive_class(explainer.shap_values(scaler.transform(X)))
    explainer = shap.TreeExplainer(model)
    return _positive_class(explainer.shap_values(X))


def global_importance(values: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """Average size of each feature's contribution, and the direction in which higher values push the risk."""
    rows = []
    for index, column in enumerate(X.columns):
        contribution = values[:, index]
        feature = X[column].to_numpy(dtype=float)
        direction = np.nan
        if np.std(feature) > 0 and np.std(contribution) > 0:
            direction = float(np.corrcoef(feature, contribution)[0, 1])
        rows.append({"feature": column, "label": LABELS.get(column, column), "mean_abs_contribution": float(np.mean(np.abs(contribution))),
                     "direction": direction})
    table = pd.DataFrame(rows).sort_values("mean_abs_contribution", ascending=False).reset_index(drop=True)
    total = table["mean_abs_contribution"].sum()
    table["share_pct"] = table["mean_abs_contribution"] / total * 100 if total else 0.0
    table["effect"] = np.where(table["direction"].isna(), "no clear direction",
                               np.where(table["direction"] > 0.1, "higher values raise the risk",
                                        np.where(table["direction"] < -0.1, "higher values lower the risk", "mixed effect")))
    return table


def contributions(values_row: np.ndarray, row: pd.Series, top: int = TOP_FEATURES) -> pd.DataFrame:
    """The features that moved one patient's prediction the most, with the patient's value for each."""
    table = pd.DataFrame({
        "feature": row.index, "label": [LABELS.get(c, c) for c in row.index], "value": [describe_value(c, row[c]) for c in row.index],
        "contribution": values_row,
    })
    table["size"] = table["contribution"].abs()
    return table.sort_values("size", ascending=False).head(top).drop(columns="size").reset_index(drop=True)


def explain_in_words(top: pd.DataFrame, probability: float, base_rate: float) -> list:
    """Short, plain sentences for the biggest contributions."""
    sentences = [f"The predicted chance of antibiotics after more than 1 hour is {probability * 100:.0f}% "
                 f"(the average for all treated patients is {base_rate * 100:.0f}%)."]
    for _, item in top.iterrows():
        effect = "raises" if item["contribution"] > 0 else "lowers"
        sentences.append(f"{item['label']} ({item['value']}) {effect} the predicted risk.")
    return sentences
