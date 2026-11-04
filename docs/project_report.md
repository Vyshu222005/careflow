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
To be completed: tools, event-log preparation, pathway discovery, waiting-time analysis, conformance checking, rework and handover analysis, care bundle, and the prediction model.

## 6. Results
To be completed: pathways, bottlenecks, conformance, rework and returns, department flow, model evaluation.

## 7. Conclusion and future work
To be completed.

## References
- Mannhardt, F. (2016). *Sepsis Cases - Event Log.* 4TU.ResearchData.
- Mannhardt, F. and Blinde, D. (2017). Analyzing the trajectories of patients with sepsis using process mining.
- van der Aalst, W. (2016). *Process Mining: Data Science in Action.* Springer.
- Berti, A., van Zelst, S., van der Aalst, W. (2019). Process Mining for Python (PM4Py).
