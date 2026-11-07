# Final presentation outline

CareFlow: Clinical Pathway Process Mining. About 12 slides, 8 to 10 minutes plus the live dashboard demonstration. Every number comes from the dashboard and `docs/project_report.md`.

## 1. Title
CareFlow: Clinical Pathway Process Mining. Name, role (Data Associate L1 Intern, Data Analytics), Infotact Solutions.

## 2. Problem
- Sepsis is life-threatening, and antibiotics are recommended within 1 hour.
- Hospitals record every step of care, but the records are rarely analysed.
- Questions: how do patients move, where do they wait, do they follow the expected pathway?

## 3. Objective
1. Discover the pathways patients actually follow.
2. Identify delays, mainly the time to antibiotics.
3. Assess conformance to the expected pathway and a care bundle.
4. Present the findings in an interactive dashboard with reports.

## 4. Dataset
- *Sepsis Cases - Event Log* (4TU.ResearchData), real anonymised emergency-room data, 2013 to 2015.
- One row per step: patient, activity, timestamp.
- Raw 1,050 patients and 15,214 events; cleaned **995 patients and 14,214 events**; 14,208 analysed after leaving out 6 rare-step events.

## 5. Approach
Collect and clean, prepare the event log, discover pathways, analyse waits, check conformance, analyse rework and handovers, predict delay, build the dashboard. Tools: Python, pandas, pm4py, Streamlit, Plotly, scikit-learn, XGBoost.

## 6. Pathways
- 96 distinct clinical pathways; the top 10 cover 73% of patients.
- Most common pathway: 22.5% (224 patients).
- Visual: the most common pathways chart or the process map.

## 7. Bottlenecks
- 776 patients received antibiotics; median wait 2.0 hours.
- 33% treated within 1 hour, 72% within 3 hours.
- Early steps are fast (triage 8 minutes, sepsis triage 11 minutes); the delay is before antibiotics.
- Visual: the registration-to-antibiotics histogram.

## 8. Quality and conformance
- About 33% follow the expected pathway (average fitness 0.90).
- Biggest deviation: antibiotics later than 1 hour (about half of patients).
- Only 18% complete the care bundle.
- Visual: the deviations chart.

## 9. Rework, returns and department flow
- 30% of patients repeat a step; 278 of 739 discharged patients return (median 46.5 days).
- 26 departments, 6,642 handovers; 89% of repeated ward records are transfers.
- Visual: the handover network.

## 10. Prediction (extra)
- Question: will antibiotics come more than 1 hour after registration?
- Leakage-controlled, patient-level validation, three models.
- No skill at registration; weak at triage (ROC-AUC 0.58, PR-AUC 0.74 against 0.67). Reported honestly; research demonstration only.

## 11. Live demonstration
Dashboard: Overview, Patient Pathways, Bottleneck Analysis, Quality & Conformance, Department Flow, then Predictive Intelligence, What-if Simulator and Reports.

## 12. Conclusion and next steps
- Treatment is slower than recommended, care varies, repeats and returns are common, delay is hard to predict.
- 87 automated tests, continuous integration, documentation.
- Next: confirm definitions with the mentor, deploy with Docker, test on other logs.
- Thank you and questions.

## Speaker tips
- One idea per slide and one number per chart.
- Say what the chart shows, the key number, and what it means.
- Keep the limitations slide ready if asked: one hospital, clock starts at registration, adapted pathway, anonymised codes.
