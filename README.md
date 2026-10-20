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
```

## Status
- [x] Project scaffold
- [ ] Dataset and data dictionary
- [ ] Data loading and cleaning
- [ ] EDA, variant analysis, process discovery
- [ ] Bottleneck analysis
- [ ] Conformance checking, dashboard, final report
