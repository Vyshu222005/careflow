# Model evaluation: antibiotic-delay prediction

Generated from `reports/delay_model_evaluation.csv` and `reports/delay_model_risk_groups.csv` (run `python src/delay_model.py`). No number in this file is typed by hand.

## Question

Will a patient's first antibiotics be given **more than 1 hour after ER registration**? Only patients with antibiotics recorded can be modelled (776 of 995; 219 have none). 66.6% of them were delayed, so *always predicting a delay* is a strong baseline for precision, recall and F1.

## Prediction points and leakage control

| Prediction point | Inputs |
|---|---|
| At ER registration | Age, arrival hour, night and weekend arrival, ER load (patients registered in the previous 1 and 6 hours), registering team, and the clinical flags the dataset stores on the registration event (suspected infection, SIRS criteria, organ dysfunction, hypotension, hypoxia, oliguria). 17 features. |
| At completion of sepsis triage | The same, plus what happened strictly **before** sepsis triage finished: minutes to triage and sepsis triage, lab tests already measured and their values, IV fluid already given. 26 features. Patients already treated before sepsis triage are left out (1 patient). |

**Never used:** anything at or after the prediction point, antibiotic information, the diagnosis code, the test-order flags (`Diagnostic*`) and the infusion flag (they describe decisions or treatment, not presentation). `assert_no_leakage` in `src/delay_features.py` fails loudly if a forbidden column reaches a model, and is covered by tests.

## Validation

* **5-fold stratified cross-validation** over patients (every row is one patient, so folds are patient-level). Headline metrics come from the out-of-fold predictions.
* **80/20 hold-out**, evaluated once.
* **Temporal check:** train on the earliest 70% of patients, test on the latest 30%.
* **Baseline:** always predict a delay.
* Class imbalance is handled with class weights. Precision, recall and F1 use a 0.5 threshold. The final model is the one with the best cross-validated ROC-AUC, refitted on all patients.

## Results: At ER registration

**5-fold cross-validation**

| Model | Tested | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Baseline: always predict delay | 776 | 0.666 | 1.000 | 0.800 | 0.500 | 0.666 |
| Logistic regression | 776 | 0.656 | 0.576 | 0.614 | 0.493 | 0.652 |
| Random forest | 776 | 0.659 | 0.605 | 0.631 | 0.500 | 0.673 |
| XGBoost | 776 | 0.661 | 0.571 | 0.613 | 0.478 | 0.655 |

**80/20 hold-out**

| Model | Tested | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Baseline: always predict delay | 156 | 0.667 | 1.000 | 0.800 | 0.500 | 0.667 |
| Logistic regression | 156 | 0.667 | 0.500 | 0.571 | 0.475 | 0.641 |
| Random forest | 156 | 0.637 | 0.490 | 0.554 | 0.456 | 0.656 |
| XGBoost | 156 | 0.675 | 0.519 | 0.587 | 0.479 | 0.650 |

**Temporal (earliest 70% to latest 30%)**

| Model | Tested | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Baseline: always predict delay | 233 | 0.708 | 1.000 | 0.829 | 0.500 | 0.708 |
| Logistic regression | 233 | 0.696 | 0.473 | 0.563 | 0.497 | 0.710 |
| Random forest | 233 | 0.694 | 0.521 | 0.595 | 0.507 | 0.737 |
| XGBoost | 233 | 0.738 | 0.479 | 0.581 | 0.521 | 0.738 |

**Do the risk groups separate?** (best model: Random forest, out-of-fold predictions split into thirds)

| Group | Patients | Observed delay rate |
|---|---|---|
| Lower risk | 259 | 68.3% |
| Medium risk | 258 | 64.3% |
| Higher risk | 259 | 67.2% |

## Results: At completion of sepsis triage

**5-fold cross-validation**

| Model | Tested | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Baseline: always predict delay | 775 | 0.667 | 1.000 | 0.800 | 0.500 | 0.667 |
| Logistic regression | 775 | 0.678 | 0.561 | 0.614 | 0.556 | 0.717 |
| Random forest | 775 | 0.685 | 0.712 | 0.698 | 0.568 | 0.735 |
| XGBoost | 775 | 0.702 | 0.634 | 0.667 | 0.577 | 0.739 |

**80/20 hold-out**

| Model | Tested | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Baseline: always predict delay | 155 | 0.664 | 1.000 | 0.798 | 0.500 | 0.664 |
| Logistic regression | 155 | 0.634 | 0.437 | 0.517 | 0.477 | 0.686 |
| Random forest | 155 | 0.694 | 0.660 | 0.677 | 0.537 | 0.721 |
| XGBoost | 155 | 0.686 | 0.573 | 0.624 | 0.552 | 0.727 |

**Temporal (earliest 70% to latest 30%)**

| Model | Tested | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Baseline: always predict delay | 233 | 0.712 | 1.000 | 0.832 | 0.500 | 0.712 |
| Logistic regression | 233 | 0.734 | 0.548 | 0.628 | 0.559 | 0.772 |
| Random forest | 233 | 0.755 | 0.687 | 0.719 | 0.558 | 0.772 |
| XGBoost | 233 | 0.733 | 0.596 | 0.658 | 0.555 | 0.773 |

**Do the risk groups separate?** (best model: XGBoost, out-of-fold predictions split into thirds)

| Group | Patients | Observed delay rate |
|---|---|---|
| Lower risk | 258 | 61.2% |
| Medium risk | 258 | 66.7% |
| Higher risk | 259 | 72.2% |

## How to read this honestly

* **At registration** the best cross-validated ROC-AUC is **0.50**, which is no better than chance (0.50), and the risk groups have the same observed delay rate. The recorded patient and arrival information cannot predict antibiotic delay.
* **At sepsis triage** the best ROC-AUC is **0.58**: a weak signal. The observed delay rate rises from the lower- to the higher-risk group, mainly because waiting longer for triage predicts a longer wait for antibiotics (see SHAP in the dashboard).
* In the hold-out and temporal checks the same models sit close to 0.5 to 0.56, and test sets are small, so any single number is uncertain.
* The practical conclusion is about the process: delays are mostly **not explained by who the patient is or when they arrive**, which points to process factors the log does not record. This is a research demonstration, **not a clinical tool**.

## Explainability

SHAP (`src/explain.py`) shows how much each feature moved a prediction: global importance across patients, and the top contributions for one patient, in plain sentences. It explains what the **model** does. It does not show what causes a delay.

## Limitations

* One hospital, 776 modelled patients, timestamps entered retrospectively, no time of sepsis recognition (the clock starts at registration).
* The clinical flags are treated as available at registration because the dataset stores them on the registration event; if they were really entered later, performance at that point would be optimistic (it is already at chance).
* Results depend on the random seed (42) and the chosen threshold (0.5). Models are not tuned beyond fixed, conservative settings.
