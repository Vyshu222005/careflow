# Progress log

Weekly summary of the work on CareFlow: Clinical Pathway Process Mining. Dates are taken from the Git history.

## Week 1: Data and process discovery (20 to 26 October 2026)
- Set up the project: folder structure, README and requirements.
- Downloaded the *Sepsis Cases* event log (4TU.ResearchData) and wrote the data dictionary.
- Loaded the data with a loading module and a notebook.
- Cleaned the data and produced a data-quality report.
- Explored the data with charts (exploratory data analysis).
- Prepared the event log and derived features such as case duration.
- Analysed pathway variants: 96 distinct clinical pathways.
- Discovered process maps and evaluated the models (pm4py).
- Analysed bottlenecks and built a performance map.
- Built the first Streamlit dashboard.

## Week 2: Quality analysis and dashboard (27 October to 2 November 2026)
- Added conformance checking against the expected sepsis pathway.
- Added rework and loop analysis: repeated steps and returns to the ER.
- Added the conformance and rework pages to the dashboard.
- Added the care-bundle check and the department and handover analysis.
- Rebuilt the dashboard with a left sidebar menu (four groups, ten pages) and collapsible patient filters.
- Added delay prediction with explanations, a what-if simulator, rule-based insights and PDF reports.
- Added input validation, 87 automated tests, GitHub Actions and Docker configuration.
- Documented the project: README, architecture diagram, model evaluation, glossary and notes on the Release codes.

## Week 3: Reporting and final deliverables (3 to 10 November 2026, in progress)
- Added this weekly progress log to the documentation.
- Planned: project report, final presentation, mentor confirmation of the expected-pathway and care-bundle definitions, and the Docker build check.
