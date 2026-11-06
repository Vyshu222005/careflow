# CareFlow: Clinical Pathway Process Mining

Project report (draft). Infotact Solutions, Data Analytics internship.

## 1. Introduction
Hospitals record every step of a patient's care with a timestamp, but these event records are rarely analysed as a whole. Process mining turns such an event log into a picture of how care actually happens: which routes patients take, where they wait, and where the care departs from what was expected.

This project applies process mining to a public log of patients with suspected sepsis in a hospital emergency room, and delivers the results as an interactive dashboard.

## 2. Problem statement
Sepsis is a life-threatening reaction to infection, and guidelines recommend giving antibiotics within one hour. Without analysis of the event records it is hard to see:
- how patients really move through the emergency room,
- where they wait longest, especially before antibiotics,
- whether they follow the expected care pathway, and
- how often they repeat steps or return to the emergency room.

## 3. Objective
To apply process mining to real sepsis emergency-room data in order to:
1. discover the pathways patients actually follow,
2. identify delays, mainly the time from registration to antibiotics,
3. assess conformance to the expected pathway and a care-bundle checklist,
4. present the findings in an interactive dashboard with reports.

## 4. Dataset
The data is the *Sepsis Cases - Event Log* (Mannhardt, 2016, 4TU.ResearchData, DOI 10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460): real, anonymised data from the emergency room of a Dutch hospital, covering 2013 to 2015.

An event log has one row per recorded step, with a case (patient), an activity and a timestamp. Typical activities are ER Registration, ER Triage, ER Sepsis Triage, blood tests (Leucocytes, CRP, LacticAcid), IV Liquid, IV Antibiotics, ward or ICU admission, Release (discharge) and Return ER.

### 4.1 Cleaning and validation
| Step | Result |
|---|---|
| Raw log | 1,050 patients and 15,214 events |
| Duplicate events removed | 7 |
| Patients with events stamped before registration removed | 55 |
| Cleaned log | **995 patients and 14,214 events** |
| Rare steps (under 1% of patients) left out of pathway, process-map and waiting-time analysis | 6 events, leaving **14,208 analysed events** |

Patient counts do not change when rare steps are left out. The full detail is in `reports/data_quality_report.md`.

### 4.2 Limitations of the data
- One hospital, public data, timestamps entered retrospectively.
- The log has no time of sepsis recognition, so the antibiotic clock starts at ER registration and the true share treated in time is probably higher.
- Department codes and the discharge codes (Release A to E) are anonymised.
- 26% of patients have no discharge event, and the reason is not in the data.

## 5. Method
### 5.1 Tools
Python, pandas and NumPy for data handling; pm4py for process mining, with Graphviz for the process maps; scikit-learn, XGBoost and SHAP for the prediction model; Streamlit and Plotly for the dashboard; ReportLab and Matplotlib for the PDF report; pytest, Git, GitHub Actions and Docker for quality and delivery.

### 5.2 Steps
| Step | What was done | Code |
|---|---|---|
| Collect and clean | Downloaded the log, removed duplicates and patients with events before registration, validated every change | `src/download_data.py`, `src/clean_data.py` |
| Prepare the event log | Ordered events, added case-level features such as duration and waiting times | `src/prepare_log.py` |
| Pathways | Grouped patients by the sequence of steps. Repeated lab tests are left out at the "clinical step" level | `src/variants.py` |
| Process discovery | Discovered process models (directly-follows graph, heuristics net, inductive miner) | `src/discovery.py` |
| Bottlenecks | Median and 90th-percentile waits between consecutive steps, time to antibiotics, monthly trends | `src/bottlenecks.py` |
| Conformance | Compared each patient with an expected sepsis pathway and scored fitness (1.0 is an exact match) | `src/conformance.py` |
| Rework | Repeated steps, loop-backs and returns to the ER | `src/rework.py` |
| Care bundle | Checked the recommended sepsis steps on time, with monthly control charts | `src/bundle.py` |
| Handovers | Moves between departments, with department roles inferred from the steps they record | `src/handover.py` |
| Prediction | Predicted whether antibiotics come more than one hour after registration | `src/delay_features.py`, `src/delay_model.py`, `src/explain.py` |
| Dashboard | Ten pages in four groups, with patient filters | `app.py`, `dashboard/` |

### 5.3 Prediction model
- **Question:** will a patient's first antibiotics be given more than 1 hour after ER registration? Only patients with antibiotics recorded can be modelled.
- **Leakage control:** a model may use only information available at its prediction point (ER registration, or completion of sepsis triage). Antibiotic information, the diagnosis code and test-order flags are never used, and a guard in the code fails if a forbidden column reaches a model.
- **Models:** logistic regression, random forest and XGBoost, with class weights for the imbalance.
- **Validation:** 5-fold stratified cross-validation over patients, an 80/20 hold-out and a temporal check, compared with the baseline of always predicting a delay. Metrics: precision, recall, F1, ROC-AUC and PR-AUC.

### 5.4 Quality assurance
87 automated tests (including checks of every dashboard page and filter), input validation that reconciles event counts, and GitHub Actions that run the tests on every push.

## 6. Results
### 6.1 Overview
| Measure | Result |
|---|---|
| Patients / cleaned events | 995 / 14,214 |
| Median case duration | 5.2 days |
| Reach intensive care | 10% |
| Median time from registration to first antibiotics | 2.0 hours |
| Patients with no antibiotics recorded | 22% (219 patients) |

### 6.2 Pathways
There are **96 distinct clinical pathways**. The top 10 cover 73% of patients, and 4% follow a route nobody else follows. The most common pathway (Registration, Triage, Sepsis triage, IV fluid, Antibiotics, Ward, Release A) is followed by 22.5% of patients (224).

### 6.3 Bottlenecks
776 patients received antibiotics. **33% were treated within 1 hour and 72% within 3 hours.** Registration to triage is fast (median 8 minutes), sepsis triage follows at 11 minutes and IV fluid at 59 minutes, but antibiotics come at a median of 2.0 hours. The month-by-month median delay rises from under one hour in early 2014 to about 2.5 hours in early 2015; months with few patients move a lot, so the trend is read with caution.

### 6.4 Conformance and care bundle
About **33%** of patients follow the expected pathway exactly (average fitness 0.90). The most frequent deviation is antibiotics later than 1 hour after registration (about half of patients). Only **18%** complete the full care bundle on time.

### 6.5 Rework, returns and department flow
| Measure | Result |
|---|---|
| Patients with rework (a repeated step) | 30% (300) |
| Patients with repeated ward records | 294 (356 extra records) |
| Loop-backs to an earlier step | 38 patients |
| Discharged patients who return to the ER | 278 of 739 (28% of all 995 patients); median 46.5 days after discharge |
| Returns by day 7 / 30 / 90 | 16% / 40% / 66% |
| Departments / handovers | 26 (20 ward units) / 6,642 |
| Repeated ward records that move to a different unit | 89% (317 of 356), so they look like transfers |

### 6.6 Prediction model
- **At ER registration** the models have **no skill**: cross-validated ROC-AUC is 0.48 to 0.50, no better than chance.
- **At completion of sepsis triage** the signal is **weak**. The best model (XGBoost, 775 patients modelled) reaches a cross-validated ROC-AUC of 0.58 (0.50 is chance) and a PR-AUC of 0.74 against 0.67 for always predicting a delay.
- The full tables, the hold-out and temporal checks, and the risk-group analysis are in `docs/model_evaluation.md`.
- The honest conclusion is that early information does not predict antibiotic delay reliably. The model is a research demonstration, not a clinical tool.

## 7. Conclusion and future work
### 7.1 Conclusion
Process mining on the sepsis event log shows how care actually runs in the emergency room, and where it falls short of the one-hour antibiotic recommendation.

- **Treatment is slower than recommended.** Only 33% of treated patients receive antibiotics within 1 hour of registration (72% within 3 hours), and the median wait is 2.0 hours. The early steps are fast, so the delay sits between sepsis triage and antibiotics.
- **Care varies.** Patients take 96 distinct pathways, only about 33% follow the expected pathway exactly, and only 18% complete the full care bundle.
- **Repeated steps and returns are common.** 30% of patients repeat a step, and 278 of 739 discharged patients return to the emergency room, a median of 46.5 days later. Most repeated ward records are transfers between units.
- **The delay is hard to predict.** With information available at registration the models have no skill, and at completion of sepsis triage the signal is weak (cross-validated ROC-AUC 0.58). This is reported as it is, and the model is a research demonstration, not a clinical tool.

The findings are delivered in an interactive dashboard (ten pages in four groups, with patient filters), a PDF report, 87 automated tests and continuous integration. All numbers come from one public dataset and are measured from ER registration, so they describe this log and are not clinical advice.

### 7.2 Limitations
- One hospital, public data, timestamps entered retrospectively.
- The antibiotic clock starts at registration because the log has no time of sepsis recognition, so the true share treated in time is probably higher.
- The expected pathway and the care bundle are adapted for this project and are not official clinical measures.
- Department and discharge codes are anonymised, so department roles are inferred.
- The prediction model has little skill, so it should not be used to support decisions.

### 7.3 Future work
- Confirm the expected pathway and the care-bundle definitions with a clinical expert.
- Test the approach on other event logs, and on logs that record the time of sepsis recognition.
- Add richer information to the prediction model, such as clinical measurements at triage, and re-evaluate it.
- Monitor the delay over time with the control charts as new data arrives.
- Build and run the Docker image so the dashboard can be deployed on any machine.

## References
- Mannhardt, F. (2016). *Sepsis Cases - Event Log.* 4TU.ResearchData.
- Mannhardt, F. and Blinde, D. (2017). Analyzing the trajectories of patients with sepsis using process mining.
- van der Aalst, W. (2016). *Process Mining: Data Science in Action.* Springer.
- Berti, A., van Zelst, S., van der Aalst, W. (2019). Process Mining for Python (PM4Py).
