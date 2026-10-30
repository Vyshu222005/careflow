"""CareFlow 2.0: clinical pathway monitoring, predictive analytics and explainable decision support.

Run:
    streamlit run app.py

Needs the processed data, built with:
    python src/clean_data.py
    python src/prepare_log.py

Structure
    app.py                  page set-up and the page flow (this file)
    dashboard/navigation.py left sidebar menu: four groups, ten pages
    dashboard/tabs/         one module per tab
    dashboard/sidebar.py    patient filters
    assets/style.css        look and feel;  src/ui.py: HTML cards and header
"""
import dashboard  # noqa: F401  (puts src/ on the import path)
from dashboard import ROOT

import warnings

import streamlit as st

warnings.filterwarnings("ignore")

import ui  # noqa: E402
from dashboard.data import build_context, data_ready, load_all  # noqa: E402
from dashboard.navigation import render_navigation  # noqa: E402
from dashboard.sidebar import render_sidebar  # noqa: E402
from validation import DataError  # noqa: E402

st.set_page_config(page_title="CareFlow 2.0 | Clinical Pathway Intelligence", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")
ui.inject_css()
st.logo(str(ROOT / "assets" / "logo.svg"), icon_image=str(ROOT / "assets" / "logo_icon.svg"), size="large")

if not data_ready():
    st.title("CareFlow")
    st.error("The processed data files were not found. Build them first, then reload this page:")
    st.code("python src/download_data.py\npython src/clean_data.py\npython src/prepare_log.py", language="bash")
    st.stop()

try:
    loaded = load_all()
except DataError as error:
    st.title("CareFlow")
    st.error(f"The data could not be validated: {error}")
    st.stop()

cases_all = loaded[1]
cases, filters_text = render_sidebar(cases_all)

first_day, last_day = cases_all["start_date"].min(), cases_all["start_date"].max()
chips = [f"{len(cases_all):,} patients", f"{len(loaded[2]):,} cleaned events", f"{first_day:%b %Y} to {last_day:%b %Y}",
         "Source: Sepsis Cases event log (4TU)"]
if len(cases) != len(cases_all):
    chips.append(f"Filtered: {len(cases):,} patients")
HERO_TEXT = ("Intelligent clinical pathway monitoring: how sepsis patients move through the emergency room, where they wait, "
             "and what could be improved.")

if cases.empty:
    ui.hero(HERO_TEXT, chips)
    st.warning("No patients match these filters. Widen the filters in the sidebar or press Reset filters.")
    st.stop()

ctx = build_context(loaded, cases, filters_text)
render_navigation(ctx, HERO_TEXT, chips)

ui.footer("CareFlow 2.0: clinical pathway intelligence. Data: Sepsis Cases event log (4TU.ResearchData). "
          "Research and education use only; not clinical advice.")
