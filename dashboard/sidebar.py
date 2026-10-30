"""Sidebar content below the navigation menu: the selected-patient counter and a collapsible Filters section.

The CareFlow logo and the page menu are drawn above this by Streamlit (st.logo and st.navigation). Filter widget keys are
unchanged, so every page keeps responding to the same filters.
"""
import pandas as pd
import streamlit as st

import ui
from dashboard.common import AGE_ORDER, RETURN_OPTIONS


def _defaults(first_day, last_day) -> dict:
    return {"f_dates": (first_day, last_day), "f_outcomes": [], "f_ages": [], "f_returned": "All"}  # empty list = no filter


def render_sidebar(cases_all: pd.DataFrame):
    """Draw the counter and the filters. Returns (selected cases, short text describing the active filters)."""
    first_day, last_day = cases_all["start_date"].min(), cases_all["start_date"].max()
    outcomes = cases_all["outcome"].value_counts().index.tolist()
    defaults = _defaults(first_day, last_day)
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    def reset_filters() -> None:
        for key, value in defaults.items():
            st.session_state[key] = value

    counter = st.sidebar.container()  # filled in below, once the selection is known; sits above the Filters section
    with st.sidebar.expander("Patient filters", expanded=False, icon=":material/filter_alt:"):
        st.caption("Applied to every analysis page. The prediction model is always trained on all patients.")
        date_range = st.date_input("Registered between", min_value=first_day, max_value=last_day, key="f_dates", help="Date of ER registration.")
        chosen_outcomes = st.multiselect("How the case ended", outcomes, key="f_outcomes", placeholder="All outcomes",
                                         help="'No release recorded' means the log has no discharge event.")
        chosen_ages = st.multiselect("Age group", AGE_ORDER, key="f_ages", placeholder="All ages")
        returned = st.radio("Returned to the ER", RETURN_OPTIONS, key="f_returned", help="Whether the patient has a Return ER event.")
        st.button("Reset filters", on_click=reset_filters, width="stretch")

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_day, end_day = date_range
    else:
        start_day, end_day = first_day, last_day

    mask = cases_all["start_date"].between(start_day, end_day)
    parts = []
    if (start_day, end_day) != (first_day, last_day):
        parts.append(f"registered {start_day:%d %b %Y} to {end_day:%d %b %Y}")
    if chosen_outcomes:
        mask &= cases_all["outcome"].isin(chosen_outcomes)
        parts.append("outcome: " + ", ".join(chosen_outcomes))
    if chosen_ages:
        mask &= cases_all["age_group"].isin(chosen_ages)
        parts.append("age: " + ", ".join(chosen_ages))
    if returned == "Returned":
        mask &= cases_all["returned_to_er"]
        parts.append("returned to the ER")
    elif returned == "Did not return":
        mask &= ~cases_all["returned_to_er"]
        parts.append("did not return to the ER")

    cases = cases_all[mask]
    filters_text = "; ".join(parts) if parts else "all patients"
    with counter:
        ui.selected_box(len(cases), len(cases_all), "Filters applied" if parts else "No filters applied")
    return cases, filters_text
