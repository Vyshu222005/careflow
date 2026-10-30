"""Dashboard tab: Predictive Intelligence (antibiotic-delay prediction with SHAP explanations)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

import delay_model
import explain
import ui
from dashboard.ml import get_explanations, get_model, signal_strength
from dashboard.common import BLUE, INK, INK_SOFT, ORANGE, show, style
from delay_features import BINARY_FEATURES, LABELS, STAGE_FEATURES, STAGES, TARGET_HOURS, feature_row
from load_data import CASE_ID

CV = "5-fold cross-validation"
METRIC_NAMES = {"precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}


def metric_table(evaluation: pd.DataFrame, validation: str) -> pd.DataFrame:
    part = evaluation[evaluation["validation"] == validation]
    table = part[["model", "patients_tested", *METRIC_NAMES]].rename(columns={"model": "Model", "patients_tested": "Patients tested", **METRIC_NAMES})
    return table.reset_index(drop=True)


def _curve_figure(x, y, baseline_x, baseline_y, title_x, title_y, label) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=baseline_x, y=baseline_y, mode="lines", line=dict(color="#98a2b3", dash="dash", width=1.5), name="No skill", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=BLUE, width=2.5), name=label, hovertemplate=f"{title_x} %{{x:.2f}}, {title_y} %{{y:.2f}}<extra></extra>"))
    fig.update_xaxes(title=title_x, range=[0, 1])
    fig.update_yaxes(title=title_y, range=[0, 1.02])
    fig = style(fig, 320, top=10)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def _contribution_figure(top: pd.DataFrame) -> go.Figure:
    top = top.iloc[::-1]
    colors = [ORANGE if c > 0 else BLUE for c in top["contribution"]]
    labels = [f"{label} = {value}" for label, value in zip(top["label"], top["value"])]
    fig = go.Figure(go.Bar(x=top["contribution"], y=labels, orientation="h", marker_color=colors,
                           hovertemplate="%{y}<br>contribution %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color=INK_SOFT, line_width=1)
    fig.update_xaxes(title="Pushes the predicted risk down  ←   →  up")
    return style(fig, 60 + 46 * len(top), top=6)


def _hypothetical_inputs(model) -> dict:
    """Widgets for a made-up patient. Defaults are typical values from the training data."""
    X = model.dataset.X
    stage = model.stage
    typical = {c: (float(X[c].median()) if c not in BINARY_FEATURES else int(X[c].mode().iloc[0])) for c in X.columns}
    values = {}
    left, middle, right = st.columns(3)
    with left:
        values["age"] = st.slider("Age", 18, 100, int(typical["age"]), key=f"h_age_{stage}")
        values["arrival_hour"] = st.slider("Hour of arrival", 0, 23, int(typical["arrival_hour"]), key=f"h_hour_{stage}")
        values["is_weekend"] = int(st.checkbox("Weekend arrival", bool(typical["is_weekend"]), key=f"h_we_{stage}"))
        values["er_load_1h"] = st.number_input("Patients registered in the previous hour", 0, 20, int(typical["er_load_1h"]), key=f"h_l1_{stage}")
        values["er_load_6h"] = st.number_input("Patients registered in the previous 6 hours", 0, 40, int(typical["er_load_6h"]), key=f"h_l6_{stage}")
        values["registered_by_team_l"] = int(st.checkbox("Registered by ER team L", False, key=f"h_l_{stage}"))
    with middle:
        for column in ("infection_suspected", "sirs_2_or_more", "sirs_tachypnea", "sirs_heart_rate", "sirs_temperature", "sirs_leucocytes"):
            values[column] = int(st.checkbox(LABELS[column], bool(typical[column]), key=f"h_{column}_{stage}"))
    with right:
        for column in ("organ_dysfunction", "hypotension", "hypoxia", "oliguria"):
            values[column] = int(st.checkbox(LABELS[column], bool(typical[column]), key=f"h_{column}_{stage}"))
        if stage == "triage":
            values["minutes_to_triage"] = st.slider("Minutes from registration to triage", 0, 120, int(round(typical["minutes_to_triage"])), key="h_mt")
            values["minutes_to_sepsis_triage"] = st.slider("Minutes from registration to sepsis triage", 0, 300, int(round(typical["minutes_to_sepsis_triage"])), key="h_mst")
            values["fluid_before_triage"] = int(st.checkbox(LABELS["fluid_before_triage"], False, key="h_fluid"))
            for lab, name in (("lactate", "Lactate"), ("crp", "CRP"), ("leucocytes", "White cell count")):
                measured = st.checkbox(f"{name} already measured", False, key=f"h_{lab}_m")
                values[f"{lab}_measured_before"] = int(measured)
                column = f"{lab}_value"
                nonzero = X.loc[X[column] > 0, column]
                values[column] = float(nonzero.median()) if measured and len(nonzero) else 0.0
    hour = values["arrival_hour"]
    values["is_night"] = int(hour >= 22 or hour < 6)
    return values


def render(ctx) -> None:
    ui.callout(
        "<b>Can we tell in advance which patients will wait more than 1 hour for antibiotics?</b> A machine-learning model is trained to answer "
        "that, with a strict rule: it may only use information that already exists at the chosen moment. Results are reported honestly, "
        "including against a do-nothing baseline. <b>This is a research demonstration, not a clinical tool.</b>"
    )
    stage_label = st.radio("Prediction point", list(STAGES.values()), index=1, horizontal=True,
                           help="'At ER registration' uses only arrival and presentation information. 'At completion of sepsis triage' also uses "
                                "what happened before triage finished (waiting times to triage, lab tests already done, fluid already given).")
    stage = next(key for key, label in STAGES.items() if label == stage_label)

    try:
        model = get_model(stage)
    except delay_model.ModelError as error:
        st.warning(f"A model cannot be trained: {error}")
        return
    values, importance = get_explanations(stage)
    dataset, evaluation = model.dataset, model.evaluation
    X, table = dataset.X, dataset.table
    cv_best = evaluation[(evaluation["validation"] == CV) & (evaluation["model"] == model.best_name)].iloc[0]
    baseline = evaluation[(evaluation["validation"] == CV) & (evaluation["model"] == delay_model.BASELINE)].iloc[0]
    verdict, kind = signal_strength(cv_best["roc_auc"])

    message = (f"**Verdict at this prediction point: {verdict}.** Best model: {model.best_name}, cross-validated ROC-AUC {cv_best['roc_auc']:.2f} "
               f"(0.50 = no better than chance), PR-AUC {cv_best['pr_auc']:.2f} against {baseline['pr_auc']:.2f} for always predicting a delay.")
    getattr(st, kind)(message)

    columns = st.columns(4)
    ui.kpi(columns[0], "Patients modelled", f"{len(table):,}", f"of {dataset.n_patients:,}; {dataset.n_without_antibiotics} have no antibiotics", "users")
    ui.kpi(columns[1], "Delayed > 1 hour", f"{model.base_rate * 100:.0f}%", "of modelled patients", "clock")
    ui.kpi(columns[2], "Best model", model.best_name, "chosen by cross-validated ROC-AUC", "route")
    ui.kpi(columns[3], "ROC-AUC / PR-AUC", f"{cv_best['roc_auc']:.2f} / {cv_best['pr_auc']:.2f}",
           f"± {cv_best['roc_auc_sd_over_folds']:.2f} across folds; baseline PR-AUC {baseline['pr_auc']:.2f}", "check")

    ui.spacer(1.0)
    with st.container(key="card_pred_models"):
        ui.section("Model comparison", "5-fold cross-validation over patients. Threshold 0.5; class weights handle imbalance. The first row is the do-nothing baseline.")
        shown = metric_table(evaluation, CV)
        st.dataframe(shown.style.format({m: "{:.3f}" for m in METRIC_NAMES.values()}), hide_index=True, width="stretch")
        with st.expander("Other validation schemes (hold-out and temporal)"):
            for validation in evaluation["validation"].unique():
                if validation == CV:
                    continue
                st.markdown(f"**{validation}**")
                st.dataframe(metric_table(evaluation, validation).style.format({m: "{:.3f}" for m in METRIC_NAMES.values()}), hide_index=True, width="stretch")
            st.caption("The temporal scheme trains on the earliest 70% of patients and tests on the latest 30%, which is closest to real use. "
                       "Small test sets make these numbers move a lot.")

    ui.spacer(1.0)
    y = dataset.y.to_numpy()
    left, middle, right = st.columns(3)
    with left, st.container(key="card_pred_roc"):
        ui.section("ROC curve", "Out-of-fold predictions of the best model.")
        fpr, tpr, _ = roc_curve(y, model.oof_proba)
        show(_curve_figure(fpr, tpr, [0, 1], [0, 1], "False positive rate", "True positive rate", model.best_name))
    with middle, st.container(key="card_pred_pr"):
        ui.section("Precision-recall curve", "The dashed line is the base rate.")
        precision, recall, _ = precision_recall_curve(y, model.oof_proba)
        show(_curve_figure(recall, precision, [0, 1], [model.base_rate] * 2, "Recall", "Precision", model.best_name))
    with right, st.container(key="card_pred_groups"):
        ui.section("Do the risk groups separate?", "Observed delay rate in thirds of predicted risk.")
        groups = model.risk_groups
        fig = go.Figure(go.Bar(x=groups["group"], y=groups["observed_delay_rate_pct"], marker_color=BLUE, customdata=groups["patients"],
                               hovertemplate="%{x}: %{y:.0f}% delayed (n=%{customdata})<extra></extra>"))
        fig.add_hline(y=model.base_rate * 100, line_dash="dash", line_color=INK_SOFT, annotation_text="all patients", annotation_position="top left")
        fig.update_yaxes(title="Observed delay rate (%)", range=[0, 100])
        show(style(fig, 320, top=20))
    st.caption("A useful model would show a clearly rising bar from lower to higher risk. Flat bars mean the model cannot tell the groups apart.")

    ui.spacer(1.0)
    with st.container(key="card_pred_shap"):
        ui.section("What drives the model? (SHAP feature importance)",
                   "Average size of each feature's contribution across all modelled patients. It describes the model, not causes.")
        top = importance.head(12).iloc[::-1]
        fig = go.Figure(go.Bar(x=top["share_pct"], y=top["label"], orientation="h", marker_color=BLUE, customdata=top["effect"],
                               hovertemplate="%{y}<br>%{x:.1f}% of total effect; %{customdata}<extra></extra>"))
        fig.update_xaxes(title="Share of the model's total effect (%)")
        show(style(fig, 90 + 30 * len(top)))
        lead = importance.iloc[0]
        st.caption(f"The biggest influence is **{lead['label'].lower()}** ({lead['share_pct']:.0f}% of the total); for it, {lead['effect']}.")

    ui.spacer(1.0)
    with st.container(key="card_pred_selection"):
        ui.section("Your selected patients", "Filters from the sidebar apply here. The model itself is always trained on all patients.")
        in_selection = table[CASE_ID].isin(ctx.cases[CASE_ID]).to_numpy()
        if in_selection.sum() < 10:
            st.info("Fewer than 10 patients in the current selection have antibiotics recorded at this prediction point.")
        else:
            observed = y[in_selection].mean() * 100
            predicted = model.oof_proba[in_selection].mean() * 100
            a, b, c = st.columns(3)
            a.metric("Modelled patients in selection", f"{int(in_selection.sum()):,}")
            b.metric("Observed delayed share", f"{observed:.0f}%")
            c.metric("Average predicted risk", f"{predicted:.0f}%", help="Out-of-fold predictions: the model had not seen these patients.")

    ui.spacer(1.0)
    with st.container(key="card_pred_patient"):
        ui.section("Explain one patient", "Pick a patient from your selection. The explanation shows which facts moved the prediction up or down.")
        candidates = table.loc[in_selection, CASE_ID].tolist()
        if not candidates:
            st.info("No modelled patients in the current selection.")
        else:
            case_id = st.selectbox("Patient ID", candidates, key=f"pred_patient_{stage}")
            index = int(np.flatnonzero(table[CASE_ID].to_numpy() == case_id)[0])
            probability = float(model.predict(X.iloc[[index]])[0])
            group = model.classify(probability)
            observed_hours = table["hours_to_antibiotics"].iloc[index]
            a, b, c = st.columns(3)
            a.metric("Predicted chance of delay", f"{probability * 100:.0f}%")
            b.metric("Risk group", group)
            c.metric("What actually happened", f"antibiotics after {observed_hours:.1f} h", "delayed" if observed_hours > TARGET_HOURS else "within 1 hour", delta_color="off")
            contribution = explain.contributions(values[index], X.iloc[index])
            show(_contribution_figure(contribution))
            st.markdown("\n".join(f"- {s}" for s in explain.explain_in_words(contribution, probability, model.base_rate)))
            st.caption("This patient was part of the training data, so the prediction is not an honest test. See the metrics above for real accuracy.")

    with st.expander("Try a hypothetical patient"):
        st.caption("Change the inputs to see how the prediction and its explanation move. These patients are made up.")
        chosen = _hypothetical_inputs(model)
        row = feature_row(chosen, stage)
        probability = float(model.predict(row)[0])
        hyp_values = explain.shap_matrix(model.model, row, background=X)[0]
        a, b = st.columns(2)
        a.metric("Predicted chance of delay", f"{probability * 100:.0f}%")
        b.metric("Risk group", model.classify(probability))
        hyp_top = explain.contributions(hyp_values, row.iloc[0])
        show(_contribution_figure(hyp_top))
        st.markdown("\n".join(f"- {s}" for s in explain.explain_in_words(hyp_top, probability, model.base_rate)))

    with st.expander("How this model is built, and its limits"):
        st.markdown(
            f"- **Question:** will the first antibiotics be given more than {TARGET_HOURS:.0f} hour after ER registration? Only the {dataset.n_with_antibiotics:,} patients with antibiotics recorded "
            f"can be modelled; {dataset.n_without_antibiotics} have none.\n"
            f"- **Inputs ({len(STAGE_FEATURES[stage])}):** age, arrival hour, night and weekend arrival, ER load, registering team, and clinical flags stored on the registration event"
            + (", plus waiting time to triage, lab tests and IV fluid that happened before sepsis triage finished." if stage == "triage" else ".") + "\n"
            "- **Left out on purpose (would leak the answer):** anything after the prediction point, lab results from later, the diagnosis code, test-order flags and the infusion flag.\n"
            "- **Validation:** patient-level 5-fold cross-validation, an 80/20 hold-out, and a temporal split, compared with always predicting a delay.\n"
            "- **Limits:** a small dataset from one hospital, timestamps entered retrospectively, and no record of when sepsis was recognised. Weak performance means "
            "delays are mostly **not explained by the recorded patient and arrival information**, which points to process factors the log does not capture."
        )
