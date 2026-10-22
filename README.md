# CareFlow: Clinical Pathway Process Mining

Data Analytics internship project at Infotact Solutions (Ref. ITS/DA/1370).

## Problem Statement
Hospitals record every step of a patient's journey (registration, tests, treatment, discharge), but rarely analyse how patients actually move through care. CareFlow applies **process mining** to hospital event logs to discover real clinical pathways, find delays and bottlenecks, and highlight deviations from the expected path.

## Objectives
1. Collect, clean and validate a clinical event log.
2. Discover the actual patient pathways and the most common variants.
3. Identify bottlenecks and slow transitions between care steps.
4. Check conformance against an expected pathway and detect rework/loops.
5. Present findings in an interactive dashboard and report.

## Key Questions
- What are the most common patient pathways?
- Where do patients wait the longest?
- Which cases deviate from the expected path, and why?

## Planned Tech Stack
Python, pandas, pm4py, matplotlib/plotly, Jupyter, Streamlit.

## Project Structure
```
careflow/
├── data/
│   ├── raw/          # original event logs (not committed)
│   └── processed/    # cleaned logs (not committed)
├── notebooks/        # exploration and analysis notebooks
├── src/              # reusable loading, cleaning and analysis code
├── reports/
│   └── figures/      # exported charts and process maps
├── docs/             # data dictionary, summaries, documentation
├── requirements.txt
└── README.md
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python src/download_data.py   # fetches the Sepsis event log into data/raw/
python src/load_data.py       # loads the log and prints its shape and sample rows
python src/clean_data.py      # cleans the log, writes data/processed/sepsis_clean.csv and the quality report
python src/prepare_log.py     # filters rare activities, adds features, writes sepsis_events.csv and sepsis_cases.csv
```

Read the processed files with `load_clean_log()` (`src/clean_data.py`), `load_events()` and `load_cases()` (`src/prepare_log.py`). One case ID is literally `NA`, which a plain `pd.read_csv` would turn into a missing value.

## Dataset
Sepsis Cases - Event Log (4TU.ResearchData, DOI 10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460). See [docs/data_dictionary.md](docs/data_dictionary.md) for columns and the case/activity/timestamp mapping.

## Status
- [x] Project scaffold
- [x] Dataset and data dictionary
- [x] Data loading ([src/load_data.py](src/load_data.py), [notebook](notebooks/01_load_data.ipynb))
- [x] Data cleaning ([src/clean_data.py](src/clean_data.py), [quality report](reports/data_quality_report.md))
- [x] Exploratory data analysis ([notebook](notebooks/02_eda.ipynb), charts in [reports/figures](reports/figures))
- [x] Event log preparation ([src/prepare_log.py](src/prepare_log.py), [report](reports/event_log_preparation.md))
- [ ] Variant analysis, process discovery
- [ ] Bottleneck analysis
- [ ] Conformance checking, dashboard, final report
