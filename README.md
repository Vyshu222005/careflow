# CareFlow: Clinical Pathway Process Mining

Process mining of real sepsis emergency-room data: which pathways patients actually follow, and where they wait.
Data Analytics internship project at Infotact Solutions (Ref. ITS/DA/1370).

## Overview
Hospitals record every step of a patient's journey (registration, tests, treatment, admission, discharge) but rarely study how patients really move through care. CareFlow applies **process mining** to a public hospital event log to:

1. clean and validate the event log,
2. discover the real patient pathways and how common each one is,
3. find where patients wait (bottlenecks),
4. present the results in an interactive dashboard.

Process mining needs three fields: a **case** (one patient's journey), an **activity** (one step) and a **timestamp**.

## Key results
Cleaned data: **995 patients, 14,208 events, 15 activities** (from 1,050 patients and 15,214 events in the raw log).

| Finding | Result |
|---|---|
| Typical case length | median 5.2 days; 26% finish within a day, 19% last over 30 days |
| Pathway variety | 619 exact pathways (54% of patients follow one nobody else did), but only **96** once repeated lab tests are left out; the top 10 cover **73%** of patients |
| Standard pathway | Registration, Triage, Sepsis triage, IV fluid, Antibiotics, Ward, Release A: 22.5% of patients |
| Treatment speed | median 8 minutes to triage, but only **33%** receive antibiotics within 1 hour of registration (72% within 3 hours) |
| Where time goes | about 80% of in-hospital time is on the ward; waits between ER steps are under 5% |
| Return visits | 28% of patients return to the ER, a median of 46.5 days after discharge |
| Missing endings | 26% of patients have no discharge event recorded |
| Rework | 30% of patients repeat a step, almost always the ward admission record (294 patients); true loops are rare (38 patients). 37.6% of discharged patients return, 15% within 30 days |
| Conformance | 33% of patients follow the assumed expected pathway exactly (average fitness 0.90); the largest deviation pattern is an extra `Admission NC` record (20% of patients) |

Waits are medians over retrospective timestamps and the antibiotics clock starts at ER registration, not at recognition of sepsis. The results show where delays happen, not why.

## Quick start
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell; in Git Bash use: source .venv/Scripts/activate
pip install -r requirements.txt
winget install Graphviz.Graphviz  # only needed to redraw the process maps (src/discovery.py, src/bottlenecks.py)

python src/download_data.py       # fetch the raw event log into data/raw/
python src/clean_data.py          # clean it
python src/prepare_log.py         # prepare it for process mining

streamlit run app.py              # open the dashboard
```
`src/discovery.py` finds Graphviz in `C:\Program Files\Graphviz\bin` even if it is not on your PATH.

## The pipeline
Run the scripts in this order. Each writes its results to `data/processed/` (not committed) or `reports/`.

| Step | Script | What it does | Main outputs |
|---|---|---|---|
| 1 | [src/download_data.py](src/download_data.py) | Downloads the Sepsis event log | `data/raw/sepsis.xes` |
| 2 | [src/load_data.py](src/load_data.py) | Loads the XES file into a table | (used by later steps) |
| 3 | [src/clean_data.py](src/clean_data.py) | Removes duplicates and 55 cases with impossible event order, fills case details | `sepsis_clean.csv`, [data quality report](reports/data_quality_report.md) |
| 4 | [src/prepare_log.py](src/prepare_log.py) | Fixes tie order, filters rare activities, adds waiting-time features | `sepsis_events.csv`, `sepsis_cases.csv`, [report](reports/event_log_preparation.md) |
| 5 | [src/variants.py](src/variants.py) | Counts pathway variants at two levels of detail | [top variants](reports/top_variants.csv) |
| 6 | [src/discovery.py](src/discovery.py) | Draws process maps and scores the models | [model evaluation](reports/model_evaluation.csv), maps in `reports/figures/` |
| 7 | [src/bottlenecks.py](src/bottlenecks.py) | Measures waits between steps | [transitions table](reports/bottleneck_transitions.csv), performance map |
| 8 | [src/conformance.py](src/conformance.py) | Checks each patient against an expected sepsis pathway | `sepsis_conformance.csv`, [deviation summary](reports/conformance_summary.csv), [patterns](reports/conformance_patterns.csv) |
| 9 | [src/rework.py](src/rework.py) | Finds repeated steps, loops and return visits | `sepsis_rework.csv`, [repeated steps](reports/rework_steps.csv), [summary](reports/rework_summary.csv) |

Read the processed files with `load_clean_log()` ([src/clean_data.py](src/clean_data.py)) and `load_events()` / `load_cases()` ([src/prepare_log.py](src/prepare_log.py)). One case ID is literally `NA`, which a plain `pd.read_csv` would turn into a missing value.

### Notebooks
Each notebook has its outputs saved, so it can be read without re-running.

| Notebook | Content |
|---|---|
| [01_load_data](notebooks/01_load_data.ipynb) | Shape, columns, missing values of the raw log |
| [02_eda](notebooks/02_eda.ipynb) | Exploratory analysis: volume, activities, case length, outcomes |
| [03_variants](notebooks/03_variants.ipynb) | Pathway variants, coverage, duration by pathway |
| [04_process_discovery](notebooks/04_process_discovery.ipynb) | Process maps and model quality |
| [05_bottlenecks](notebooks/05_bottlenecks.ipynb) | Waiting times, time to antibiotics, where the time goes |
| [06_conformance](notebooks/06_conformance.ipynb) | Conformance to the expected pathway: fitness, deviations, patterns |
| [07_rework](notebooks/07_rework.ipynb) | Repeated steps, loops, ward records and return visits |

## Dashboard
```bash
streamlit run app.py
```
Seven tabs, all driven by the sidebar filters (registration dates, how the case ended, age group, returned to the ER):

| Tab | What it shows |
|---|---|
| Overview | Key numbers, four insight cards, patients per month, how cases end, case duration |
| Pathways | Most common pathways at clinical-step or exact level, coverage, per-pathway duration |
| Process map | A live map of patient flow with counts or median waits, in either direction |
| Waiting times | Time to antibiotics against the 1-hour and 3-hour marks, waits between ER steps, all transitions |
| Conformance | How closely patients follow the expected pathway: fitness, deviations, deviation patterns |
| Rework | Repeated steps, gaps between repeated ward records, ward stay, and how soon patients return |
| About | Data source, method, and how to read the numbers |

The charts and logic are in [app.py](app.py). The look is defined in [assets/style.css](assets/style.css) (stylesheet) and [src/ui.py](src/ui.py) (HTML for the banner and cards).

## Project structure
```
careflow/
├── app.py                 # Streamlit dashboard
├── assets/style.css       # dashboard stylesheet
├── data/
│   ├── raw/               # original event log (not committed)
│   └── processed/         # cleaned and prepared logs (not committed)
├── docs/
│   └── data_dictionary.md # columns, activities, case/activity/timestamp mapping
├── notebooks/             # five analysis notebooks
├── reports/
│   ├── figures/           # charts and process maps
│   └── *.md, *.csv        # quality report, variants, model scores, waiting times
├── src/                   # pipeline code (load, clean, prepare, variants, discovery, bottlenecks, ui)
├── .streamlit/config.toml # dashboard theme
├── pyrightconfig.json     # tells VS Code where src/ is
└── requirements.txt
```

## Dataset
*Sepsis Cases - Event Log* (Mannhardt, 2016, 4TU.ResearchData, DOI 10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460): real, anonymised hospital data. See [docs/data_dictionary.md](docs/data_dictionary.md).

## Limitations and open questions
- The dataset is public research data, not data supplied by the company.
- The expected sepsis pathway used for conformance checking is an assumption (the standard route in the data plus the usual care sequence); a different definition gives different conformance numbers.
- 26% of patients have no discharge event, and the reason is not in the data.
- `Admission NC` is recorded more than once for many patients (320 immediate repeats); transfers or repeated records are both possible.
- IV fluid and antibiotics are often recorded at the same moment, so they may be entered together rather than given together.
- Process-model scores involve a trade-off: the heuristics model is the most precise (0.94) but only 15% of cases replay perfectly, while the inductive model with noise filtering replays 61%.

## Status
- [x] Project setup, dataset and data dictionary
- [x] Data loading and cleaning
- [x] Exploratory data analysis
- [x] Event log preparation
- [x] Variant analysis
- [x] Process discovery and model evaluation
- [x] Bottleneck analysis
- [x] Interactive dashboard
- [x] Conformance checking against an expected sepsis pathway ([src/conformance.py](src/conformance.py), [notebook](notebooks/06_conformance.ipynb)); the expected pathway is an assumption to confirm with the mentor
- [x] Rework and loop analysis ([src/rework.py](src/rework.py), [notebook](notebooks/07_rework.ipynb))
- [ ] Final report, presentation and demo
