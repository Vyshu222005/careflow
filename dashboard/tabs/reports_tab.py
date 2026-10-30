"""Dashboard tab: Reports (downloadable PDF of the current selection)."""
from datetime import datetime

import streamlit as st

import insights
import report
import simulator
import ui
from dashboard.ml import get_explanations, get_model, signal_strength
from delay_features import STAGES


def _model_summary(stage: str) -> dict:
    model = get_model(stage)
    _, importance = get_explanations(stage)
    cv = model.evaluation[model.evaluation["validation"] == "5-fold cross-validation"]
    best = cv[cv["model"] == model.best_name].iloc[0]
    metrics = cv[["model", "patients_tested", "precision", "recall", "f1", "roc_auc", "pr_auc"]].rename(columns={
        "model": "Model", "patients_tested": "Tested", "precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"})
    return {"stage_label": STAGES[stage], "best_model": model.best_name, "verdict": signal_strength(best["roc_auc"])[0], "metrics": metrics,
            "importance": importance, "base_rate": model.base_rate}


def _simulation_summary(ctx, preset: str) -> dict:
    scenario = simulator.PRESETS[preset]
    result = simulator.simulate(ctx.hours, scenario)
    text = (f"Scenario: {preset}. Segment 1 shortened by {scenario.cut_segment1_pct:.0f}%, segment 2 by {scenario.cut_segment2_pct:.0f}% and "
            f"{scenario.cut_segment2_minutes:.0f} minutes, for {scenario.adoption_pct:.0f}% of patients; segment 2 not below {scenario.floor_minutes:.0f} minutes.")
    return {"scenario_text": text, "summary": result.summary().rename(columns={"metric": "Metric", "baseline": "Recorded", "simulated": "Hypothetical", "change": "Change"}),
            "newly_within_one_hour": result.newly_within_one_hour, "patients": len(result.baseline_hours)}


def render(ctx) -> None:
    ui.callout(
        "<b>Take the findings with you.</b> Build a PDF of the patients currently selected in the sidebar: key figures, patient pathways, bottlenecks, "
        "quality scorecard, department flow, recommendations and limitations. Optionally add the prediction model summary and a what-if scenario."
    )
    with st.container(key="card_report_options"):
        ui.section("Report contents", "The sidebar filters decide which patients are included.")
        a, b = st.columns(2)
        with a:
            include_model = st.checkbox("Include the prediction model summary", value=True)
            stage_label = st.radio("Prediction point to report", list(STAGES.values()), index=1, disabled=not include_model, key="report_stage")
        with b:
            include_simulation = st.checkbox("Include a what-if scenario (clearly marked hypothetical)", value=False)
            preset = st.selectbox("Scenario", [name for name in simulator.PRESETS if name != "Baseline (no change)"], disabled=not include_simulation, key="report_preset")
        st.caption(f"Patients in this report: {len(ctx.cases):,} ({ctx.filters_text}).")

        if st.button("Build the PDF report", type="primary"):
            with st.spinner("Building the report..."):
                try:
                    metrics = insights.collect_from_context(ctx)
                    found = insights.generate(metrics)
                    model = _model_summary(next(k for k, v in STAGES.items() if v == stage_label)) if include_model else None
                    simulation = _simulation_summary(ctx, preset) if include_simulation else None
                    data = report.collect_report_data(ctx, metrics, found, model, simulation)
                    st.session_state["report_pdf"] = report.build_pdf(data)
                    st.session_state["report_name"] = f"careflow_report_{datetime.now():%Y%m%d_%H%M}.pdf"
                    st.session_state["report_for"] = (len(ctx.cases), ctx.filters_text)
                except Exception as error:  # noqa: BLE001 - show a readable message instead of a stack trace
                    st.error(f"The report could not be built: {error}")

    pdf = st.session_state.get("report_pdf")
    if pdf:
        ui.spacer(0.8)
        with st.container(key="card_report_ready"):
            ui.section("Your report is ready")
            patients, filters = st.session_state["report_for"]
            st.markdown(f"**{st.session_state['report_name']}**, {len(pdf) / 1024:.0f} KB, for {patients:,} patients ({filters}).")
            st.download_button("Download the PDF", pdf, file_name=st.session_state["report_name"], mime="application/pdf")
            if (patients, filters) != (len(ctx.cases), ctx.filters_text):
                st.info("The sidebar filters have changed since this report was built. Build it again to match the current selection.")
