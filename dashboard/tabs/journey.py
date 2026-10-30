"""Dashboard tab (part of Patient Pathways): explore one patient's journey step by step."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from bottlenecks import format_hours
from dashboard.common import BLUE, ORANGE, show, style
from load_data import ACTIVITY, CASE_ID, TIMESTAMP

GROUP = "org:group"
LABS = {"Leucocytes", "CRP", "LacticAcid"}


def journey_table(clean_log: pd.DataFrame, case_id: str) -> pd.DataFrame:
    """One patient's events in time order, with hours since registration and since the previous event."""
    events = clean_log[clean_log[CASE_ID] == case_id].sort_values(TIMESTAMP, kind="stable").reset_index(drop=True)
    start = events[TIMESTAMP].iloc[0]
    events["hours_since_registration"] = (events[TIMESTAMP] - start).dt.total_seconds() / 3600
    events["hours_since_previous"] = events[TIMESTAMP].diff().dt.total_seconds() / 3600
    return events[[TIMESTAMP, ACTIVITY, GROUP, "hours_since_registration", "hours_since_previous"]]


def render(ctx) -> None:
    ui.callout("<b>Follow one patient.</b> Pick a patient to see every recorded step in time order, how long each step took to arrive, "
               "and how the patient compares with the expected pathway and the care bundle.")
    cases = ctx.cases.sort_values("duration_days", ascending=False)
    choice = st.radio("Which patient?", ["Longest journey in the selection", "Shortest journey", "Choose a patient ID"], horizontal=True)
    if choice.startswith("Longest"):
        case_id = cases[CASE_ID].iloc[0]
    elif choice.startswith("Shortest"):
        case_id = cases[CASE_ID].iloc[-1]
    else:
        case_id = st.selectbox("Patient ID", cases[CASE_ID].tolist(), help="IDs are anonymised case codes from the dataset.")

    row = cases[cases[CASE_ID] == case_id].iloc[0]
    journey = journey_table(ctx.clean_log, case_id)
    conf = ctx.conformance_all[ctx.conformance_all[CASE_ID] == case_id]
    bundle_row = ctx.bundle_all[ctx.bundle_all[CASE_ID] == case_id]

    columns = st.columns(4)
    ui.kpi(columns[0], "Patient", str(case_id), f"age {row['age']:.0f}, {row['age_group']}", "users")
    ui.kpi(columns[1], "Journey length", f"{row['duration_days']:.1f} days", f"{len(journey)} recorded events", "clock")
    ui.kpi(columns[2], "Ended with", str(row["outcome"]), "first discharge code", "check")
    fitness = f"{conf['fitness'].iloc[0]:.2f}" if len(conf) else "n/a"
    ui.kpi(columns[3], "Fits expected pathway", fitness, "1.00 = follows it exactly", "route")

    ui.spacer(1.0)
    with st.container(key="card_journey_timeline"):
        ui.section("Timeline", "Each dot is one recorded event. Orange dots are repeated blood tests; hover for details.")
        order = list(dict.fromkeys(journey[ACTIVITY]))
        is_lab = journey[ACTIVITY].isin(LABS)
        fig = go.Figure()
        for mask, color, name in ((~is_lab, BLUE, "Clinical step"), (is_lab, ORANGE, "Blood test")):
            part = journey[mask]
            if part.empty:
                continue
            fig.add_trace(go.Scatter(
                x=part[TIMESTAMP], y=part[ACTIVITY], mode="markers", name=name, marker=dict(color=color, size=10, line=dict(color="white", width=1.5)),
                customdata=list(zip(part[GROUP], part["hours_since_registration"].map(format_hours))),
                hovertemplate="%{y}<br>%{x|%d %b %Y %H:%M}<br>department %{customdata[0]}, %{customdata[1]} after registration<extra></extra>",
            ))
        fig.update_yaxes(categoryorder="array", categoryarray=order[::-1], title="")
        fig.update_xaxes(title="Time")
        fig = style(fig, 90 + 32 * len(order), top=10)
        fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        show(fig)

    ui.spacer(0.8)
    left, right = st.columns(2)
    with left:
        ui.section("How this patient compares")
        if len(bundle_row):
            b = bundle_row.iloc[0]
            checks = [("Lactate within 1 hour", b["lactate_1h"]), ("Antibiotics within 1 hour", b["antibiotics_1h"]), ("Complete care bundle", b["complete_bundle"])]
            st.markdown("\n".join(f"- {'✅' if bool(ok) else '❌'} {label}" for label, ok in checks))
        if len(conf) and conf["missing_steps"].iloc[0]:
            st.markdown(f"- Missing steps: **{conf['missing_steps'].iloc[0]}**")
        if len(conf) and conf["unexpected_steps"].iloc[0]:
            st.markdown(f"- Extra steps: **{conf['unexpected_steps'].iloc[0]}**")
    with right:
        ui.section("Event list")
        view = journey.assign(**{
            "Time": journey[TIMESTAMP].dt.strftime("%d %b %H:%M"),
            "Since registration": journey["hours_since_registration"].map(format_hours),
            "Since previous": journey["hours_since_previous"].map(lambda h: "" if pd.isna(h) else format_hours(h)),
        }).rename(columns={ACTIVITY: "Step", GROUP: "Dept"})[["Time", "Step", "Dept", "Since registration", "Since previous"]]
        st.dataframe(view, hide_index=True, width="stretch", height=300)
