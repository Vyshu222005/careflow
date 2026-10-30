"""Dashboard tab (part of Bottleneck Analysis): registration-to-triage waits and monthly delay trends."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from bottlenecks import format_hours
from dashboard.common import BLUE, INK, show, style
from load_data import CASE_ID

MIN_MONTH_PATIENTS = 10
MINUTES_CAP = 60


def monthly_delay_table(cases: pd.DataFrame, hours: pd.DataFrame, step: str = "IV Antibiotics", target_hours: float = 1.0) -> pd.DataFrame:
    """Per registration month: patients with the step, median hours from registration, and share within the target."""
    month = cases.set_index(CASE_ID)["start_month"]
    data = pd.DataFrame({"hours": hours[step], "month": month.reindex(hours.index)}).dropna()
    table = data.groupby("month")["hours"].agg(patients="size", median_hours="median", within=lambda s: (s <= target_hours).mean() * 100)
    return table[table["patients"] >= MIN_MONTH_PATIENTS].reset_index().rename(columns={"within": "within_target_pct"})


def render(ctx) -> None:
    cases, hours = ctx.cases, ctx.hours
    ui.spacer(1.2)
    ui.section("Early steps from registration", "Time from ER registration to the first time each early step is recorded.")
    steps = [("ER Triage", "Registration to triage", "users"), ("ER Sepsis Triage", "Registration to sepsis triage", "alert"),
             ("IV Liquid", "Registration to IV fluid", "drop"), ("IV Antibiotics", "Registration to antibiotics", "clock")]
    columns = st.columns(4)
    for column, (step, label, icon) in zip(columns, steps):
        values = hours[step].dropna()
        ui.kpi(column, label, format_hours(values.median()) if len(values) else "n/a",
               f"90th percentile {format_hours(values.quantile(0.9))}; {len(values):,} patients" if len(values) else "no patients", icon)

    ui.spacer(1.0)
    left, right = st.columns(2)
    with left, st.container(key="card_triage_dist"):
        ui.section("Registration to triage", f"Minutes from registration to the first triage (waits over {MINUTES_CAP} minutes grouped at {MINUTES_CAP}).")
        minutes = (hours["ER Triage"].dropna() * 60)
        if minutes.empty:
            st.info("No triage times in this selection.")
        else:
            fig = go.Figure(go.Histogram(x=minutes.clip(upper=MINUTES_CAP), xbins=dict(start=0, end=MINUTES_CAP, size=3), marker_color=BLUE,
                                         hovertemplate="%{x:.0f} min: %{y} patients<extra></extra>"))
            fig.add_vline(x=min(minutes.median(), MINUTES_CAP), line_dash="dash", line_color=INK,
                          annotation_text=f"median {minutes.median():.0f} min", annotation_position="top right")
            fig.update_xaxes(title="Minutes")
            fig.update_yaxes(title="Patients")
            show(style(fig, 320, top=30))
    with right, st.container(key="card_arrival_period"):
        ui.section("Antibiotic delay by arrival time", "Median hours from registration to antibiotics, by the hour of day the patient registered.")
        reg = ctx.cases.set_index(CASE_ID)["start"].dt.hour
        data = pd.DataFrame({"hours": hours["IV Antibiotics"], "hour": reg.reindex(hours.index)}).dropna()
        if data.empty:
            st.info("No antibiotic times in this selection.")
        else:
            data["period"] = pd.cut(data["hour"], [-1, 5, 11, 17, 23], labels=["00:00-05:59", "06:00-11:59", "12:00-17:59", "18:00-23:59"])
            by_period = data.groupby("period", observed=True)["hours"].agg(["median", "size"])
            fig = go.Figure(go.Bar(x=by_period.index.astype(str), y=by_period["median"], marker_color=BLUE, customdata=by_period["size"],
                                   hovertemplate="%{x}: median %{y:.1f} h (n=%{customdata})<extra></extra>"))
            fig.update_yaxes(title="Median hours")
            show(style(fig, 320))

    ui.spacer(1.0)
    with st.container(key="card_monthly_delay"):
        ui.section("Monthly delay trends", f"Antibiotic timing by registration month. Months with fewer than {MIN_MONTH_PATIENTS} treated patients are left out.")
        trend = monthly_delay_table(cases, hours)
        if len(trend) < 2:
            st.info("Not enough months with enough patients in this selection to show a trend.")
        else:
            months = pd.to_datetime(trend["month"] + "-01").dt.strftime("%b %y").tolist()
            a, b = st.columns(2)
            with a:
                st.markdown("**Median hours to antibiotics**")
                fig = go.Figure(go.Scatter(x=months, y=trend["median_hours"], mode="lines+markers", line=dict(color=BLUE, width=2.5), customdata=trend["patients"],
                                           hovertemplate="%{x}: %{y:.1f} h (n=%{customdata})<extra></extra>"))
                fig.update_xaxes(type="category", tickangle=-45)
                fig.update_yaxes(title="Hours", rangemode="tozero")
                show(style(fig, 300))
            with b:
                st.markdown("**Share treated within 1 hour**")
                fig = go.Figure(go.Scatter(x=months, y=trend["within_target_pct"], mode="lines+markers", line=dict(color=BLUE, width=2.5), customdata=trend["patients"],
                                           hovertemplate="%{x}: %{y:.0f}% (n=%{customdata})<extra></extra>"))
                fig.update_xaxes(type="category", tickangle=-45)
                fig.update_yaxes(title="% of treated patients", range=[0, 100])
                show(style(fig, 300))
            st.caption("Single months with few patients move a lot. See Quality & Conformance for a control chart that separates real change from chance.")
