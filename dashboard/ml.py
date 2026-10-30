"""Cached model and explanation loaders, shared by the Predictive and Reports tabs."""
import streamlit as st

import conformance
import delay_model
import explain


@st.cache_resource(show_spinner="Training and validating the models (the first time only, about 30 seconds)...")
def get_model(stage: str):
    return delay_model.load_or_train(conformance.load_log(), stage)


@st.cache_data(show_spinner="Explaining the model with SHAP...")
def get_explanations(stage: str):
    model = get_model(stage)
    X = model.dataset.X
    values = explain.shap_matrix(model.model, X)
    return values, explain.global_importance(values, X)


def signal_strength(roc_auc: float) -> tuple:
    """(label, streamlit message kind) describing how much the model can tell patients apart."""
    if roc_auc < 0.55:
        return "no usable signal", "warning"
    if roc_auc < 0.65:
        return "weak signal", "warning"
    if roc_auc < 0.75:
        return "moderate signal", "info"
    return "good signal", "success"
