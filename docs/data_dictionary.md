# Data Dictionary: Sepsis Cases Event Log

## Source
- **Dataset:** Sepsis Cases - Event Log (Mannhardt, F., 2016), 4TU.ResearchData
- **DOI:** 10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460
- **Content:** Real hospital data on patients with suspected sepsis, from the emergency room (ER) registration through treatment, admission and discharge. Data is anonymised.
- **Format:** XES (`sepsis.xes`), loaded into a pandas DataFrame with pm4py.
- **Approximate size:** about 1,050 cases, about 15,000 events, 16 distinct activities. Exact counts are confirmed in the data-loading step.
- **Location:** `data/raw/sepsis.xes` (not committed; fetch with `python src/download_data.py`).

## Process Mining Mapping
Every process mining algorithm needs three fields. They map to this dataset as follows:

| Process mining concept | XES column | Meaning |
|---|---|---|
| **Case ID** | `case:concept:name` | One patient pathway (a patient's sepsis episode) |
| **Activity** | `concept:name` | The care step performed, for example `ER Triage` or `IV Antibiotics` |
| **Timestamp** | `time:timestamp` | When the event happened |
| Resource (optional) | `org:group` | Hospital department/group that performed the activity |

## Event-Level Columns
| Column | Type | Description |
|---|---|---|
| `case:concept:name` | string | Case identifier |
| `concept:name` | string | Activity name |
| `time:timestamp` | datetime | Event timestamp |
| `org:group` | string | Responsible group/department |
| `lifecycle:transition` | string | Event lifecycle state (for example `complete`) |

## Activities
| Activity | Description |
|---|---|
| ER Registration | Patient registered at the emergency room |
| ER Triage | Initial triage assessment |
| ER Sepsis Triage | Sepsis-specific triage |
| Leucocytes | Leucocyte (white blood cell) lab measurement |
| CRP | C-reactive protein lab measurement |
| LacticAcid | Lactic acid lab measurement |
| IV Liquid | Intravenous fluid administered |
| IV Antibiotics | Intravenous antibiotics administered |
| Admission NC | Admission to normal care ward |
| Admission IC | Admission to intensive care |
| Release A to E | Discharge, with the code indicating the reason or destination |
| Return ER | Patient returned to the emergency room |

## Case-Level Attributes
These values are recorded on events and describe the patient or clinical state.

| Column | Type | Description |
|---|---|---|
| `Age` | integer | Patient age |
| `Diagnose` | string | Diagnosis code |
| `InfectionSuspected` | boolean | Infection suspected at triage |
| `DiagnosticBlood`, `DiagnosticArtAstrup`, `DiagnosticIC`, `DiagnosticSputum`, `DiagnosticLiquor`, `DiagnosticOther`, `DiagnosticUrinaryCulture`, `DiagnosticUrinarySediment`, `DiagnosticECG`, `DiagnosticLacticAcid`, `DiagnosticXthorax` | boolean | Whether the named diagnostic test was ordered |
| `DisfuncOrg` | boolean | Organ dysfunction present |
| `Hypotensie` | boolean | Low blood pressure |
| `Hypoxie` | boolean | Low blood oxygen |
| `Oligurie` | boolean | Low urine output |
| `Infusion` | boolean | Infusion administered |
| `SIRSCritHeartRate`, `SIRSCritLeucos`, `SIRSCritTachypnea`, `SIRSCritTemperature` | boolean | Individual SIRS criteria met |
| `SIRSCriteria2OrMore` | boolean | Two or more SIRS criteria met |

## Measurement Columns
| Column | Type | Description |
|---|---|---|
| `CRP` | float | C-reactive protein value (mg/L) |
| `LacticAcid` | float | Lactic acid value (mmol/L) |
| `Leucocytes` | float | Leucocyte count (10^9/L) |

## Known Data Considerations (to handle in the cleaning step)
- Many attribute columns are only filled on certain events, so they have many missing values.
- Timestamps must be parsed and events sorted by time within each case.
- Some cases are very short or very long and may need filtering.
- Rare activities may be treated as noise during discovery.

> Column names and types are taken from the dataset documentation and will be verified against the loaded data in the loading step.
