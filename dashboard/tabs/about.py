"""Dashboard tab: About (dataset, methodology, technologies, cleaning, limitations, references)."""
import streamlit as st

import ui


def render(ctx) -> None:
    counts = ctx.counts
    left, right = st.columns(2)
    with left, st.container(key="card_about_data"):
        ui.section("Dataset")
        st.markdown(
            "*Sepsis Cases - Event Log* (Mannhardt, 2016, 4TU.ResearchData): real, anonymised hospital data from the emergency room of a Dutch hospital, "
            "1,050 patients and 15,214 events between 2013 and 2015. A **case** is one patient's journey, an **activity** one recorded step."
        )
        ui.section("Methodology")
        st.markdown(
            "1. **Clean** the event log and validate every change.\n"
            "2. **Prepare** it for process mining (consistent event order, waiting-time features).\n"
            "3. **Analyse** pathways, process maps, waiting times, conformance, rework, a care-bundle scorecard and department handovers (pm4py, pandas).\n"
            "4. **Predict** antibiotic delay with validated machine-learning models and explain them with SHAP.\n"
            "5. **Simulate** hypothetical waiting-time changes and **summarise** findings with transparent rules and PDF reports."
        )
        ui.section("Data cleaning")
        st.markdown(
            f"- 1,050 patients and 15,214 events in the raw log.\n"
            f"- 7 duplicate events and 55 patients whose events are stamped before registration were removed, leaving **{counts.patients:,} patients and {counts.cleaned_events:,} cleaned events**.\n"
            f"- Rare steps (under 1% of patients) are left out of pathway, process-map and waiting-time analysis: {counts.excluded_events} events "
            f"({', '.join(counts.excluded_activities) or 'none'}), leaving **{counts.analysis_events:,} analysed events**. Patient counts do not change.\n"
            "- Patient details recorded once per patient were copied to every event; blood-test values stay empty when no test was done.\n"
            "- Full detail: `reports/data_quality_report.md`."
        )
    with right, st.container(key="card_about_tech"):
        ui.section("Technologies")
        st.markdown(
            "Python, pandas, NumPy, **pm4py** (process mining), **scikit-learn** and **XGBoost** (models), **SHAP** (explanations), "
            "Streamlit and Plotly (dashboard), ReportLab and Matplotlib (PDF reports), pytest, Git and GitHub, GitHub Actions and Docker."
        )
        ui.section("Reading the numbers")
        st.markdown(
            "- Waits are medians over retrospective timestamps. They show where delays happen, not why.\n"
            "- **Clinical steps** leave out repeated lab tests (Leucocytes, CRP, LacticAcid).\n"
            "- **Release A to E** are anonymised discharge codes: the way a stay ended. The data does not say what each letter stands for. "
            "Release A is the most common; 'No release recorded' means no discharge event exists.\n"
            "- Antibiotic timing is measured from **ER registration**. The log has no time of sepsis recognition, so the true share treated in time is probably higher.\n"
            "- 'Treated within 1 hour' is a share of patients **with antibiotics recorded**; the care bundle counts **all** patients, so the two figures differ on purpose."
        )
        ui.section("Project limitations")
        st.markdown(
            "- A public dataset from one hospital, with timestamps entered retrospectively.\n"
            "- 26% of patients have no discharge event, and the reason is not in the data.\n"
            "- The expected pathway and the care-bundle scorecard are adapted for this project and are not official measures.\n"
            "- Department codes are anonymised; roles are inferred.\n"
            "- IV fluid and antibiotics are often recorded at the same moment."
        )
        ui.section("Model limitations")
        st.markdown(
            "- Only information available at the prediction point is used, and patients with no antibiotics cannot be modelled.\n"
            "- With registration-time information the models cannot beat chance; with triage-time information the signal is weak. This is reported openly.\n"
            "- SHAP explains what the **model** does, not what causes a delay. The simulator is arithmetic, not a forecast.\n"
            "- Nothing here is clinical advice."
        )
        ui.section("References")
        st.markdown(
            "- Mannhardt, F. (2016). *Sepsis Cases - Event Log.* 4TU.ResearchData. doi:10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460\n"
            "- Mannhardt, F. and Blinde, D. (2017). Analyzing the trajectories of patients with sepsis using process mining.\n"
            "- van der Aalst, W. (2016). *Process Mining: Data Science in Action.* Springer.\n"
            "- Berti, A., van Zelst, S., van der Aalst, W. (2019). Process Mining for Python (PM4Py).\n"
            "- Lundberg, S. and Lee, S. (2017). A unified approach to interpreting model predictions (SHAP). NeurIPS.\n"
            "- Evans, L. et al. (2021). Surviving Sepsis Campaign: international guidelines for management of sepsis and septic shock."
        )
