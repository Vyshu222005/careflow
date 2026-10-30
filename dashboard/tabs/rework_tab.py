"""Dashboard tab: rework tab (moved unchanged from app.py)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import rework
import ui
from load_data import ACTIVITY, CASE_ID
from dashboard.common import BLUE, show, style


def render(ctx) -> None:
    cases = ctx.cases
    clean_log = ctx.clean_log
    rework_all = ctx.rework_all
    ui.callout(
        "<b>Which steps do patients repeat, and who comes back?</b> A <b>repeat</b> is a step that occurs more than once for the same "
        "patient; a <b>loop back</b> is a step that reappears after a different step. Repeated lab tests are left out, because "
        "repeating blood tests is routine monitoring."
    )
    ids = cases[CASE_ID]
    rw = rework_all[rework_all[CASE_ID].isin(ids)]
    log_sel = clean_log[clean_log[CASE_ID].isin(ids)]
    steps_sel = rework.clinical_steps(log_sel)
    gaps = rework.ward_record_gaps(steps_sel)
    stays = rework.ward_stay_by_records(log_sel, steps_sel)
    returns = rework.return_intervals(log_sel)
    released = log_sel[log_sel[ACTIVITY].str.startswith("Release")][CASE_ID].nunique()
    ward_repeaters = int(rw["repeated_steps"].str.contains(rework.WARD).sum())
    within_30 = int((returns["days_to_return"] <= 30).sum())

    columns = st.columns(4)
    ui.kpi(columns[0], "Patients with rework", f"{rw['has_rework'].mean() * 100:.0f}%", f"{int(rw['has_rework'].sum()):,} repeat a step", "repeat")
    ui.kpi(columns[1], "Repeated ward record", f"{ward_repeaters:,}", f"patients; {len(gaps):,} extra records", "list")
    ui.kpi(columns[2], "Loop back", f"{int((rw['loop_backs'] > 0).sum()):,}", "patients return to an earlier step", "route")
    ui.kpi(columns[3], "Back within 30 days", f"{within_30 / released * 100:.0f}%" if released else "n/a", f"of {released:,} discharged patients", "clock")

    ui.spacer(1.0)
    left, right = st.columns(2)
    with left, st.container(key="card_rework_gaps"):
        ui.section("Time between repeated ward records", "Gap between consecutive Admission NC records of one patient.")
        if gaps.empty:
            st.info("No repeated ward records in this selection.")
        else:
            edges = [0, 6, 12, 24, 48, 120, np.inf]
            labels = ["under 6 h", "6 to 12 h", "12 to 24 h", "1 to 2 days", "2 to 5 days", "over 5 days"]
            buckets = pd.cut(gaps, bins=edges, labels=labels, right=False).value_counts().reindex(labels)
            fig = go.Figure(go.Bar(
                x=buckets.index, y=buckets.values, marker_color=BLUE, customdata=buckets.values / len(gaps) * 100,
                hovertemplate="%{x}: %{y} gaps (%{customdata:.0f}%)<extra></extra>",
            ))
            fig.update_yaxes(title="Gaps between records")
            show(style(fig, 330))
    with right, st.container(key="card_rework_stay"):
        ui.section("Ward stay by number of ward records", "Median days from the first ward record to discharge.")
        if stays.empty:
            st.info("No discharged patients with a ward record in this selection.")
        else:
            stays = stays.assign(group=stays["records"].clip(upper=3).map({1: "1 record", 2: "2 records", 3: "3 or more"}))
            by_group = stays.groupby("group")["stay_days"].agg(["median", "size"]).reindex(["1 record", "2 records", "3 or more"]).dropna()
            fig = go.Figure(go.Bar(
                x=by_group.index, y=by_group["median"], marker_color=BLUE, customdata=by_group["size"],
                hovertemplate="%{x}: median %{y:.1f} days (n=%{customdata})<extra></extra>",
            ))
            fig.update_yaxes(title="Days")
            show(style(fig, 330))

    ui.spacer(1.0)
    with st.container(key="card_rework_returns"):
        ui.section("How soon do patients return to the ER?", "Share of all returns that have happened by each day after discharge.")
        if returns.empty:
            st.info("No returns to the ER in this selection.")
        else:
            days = returns["days_to_return"].sort_values().to_numpy()
            share = np.arange(1, len(days) + 1) / len(days) * 100
            fig = go.Figure(go.Scatter(
                x=np.maximum(days, 0.5), y=share, mode="lines", line=dict(color=BLUE, width=2.5, shape="hv"),
                hovertemplate="%{y:.0f}% of returns within %{x:.0f} days<extra></extra>",
            ))
            for window in rework.RETURN_WINDOWS_DAYS:
                value = (days <= window).mean() * 100
                fig.add_trace(go.Scatter(
                    x=[window], y=[value], mode="markers+text", text=[f"{value:.0f}% by day {window}"], textposition="bottom right",
                    marker=dict(color=BLUE, size=10, line=dict(color="white", width=2)), hoverinfo="skip",
                ))
            fig.update_xaxes(type="log", tickvals=[1, 7, 30, 90, 365], ticktext=["1 day", "1 week", "30 days", "90 days", "1 year"],
                             title="Time since discharge (log scale)")
            fig.update_yaxes(title="Share of returns so far (%)", range=[0, 102])
            show(style(fig, 340))
            st.caption(f"{len(returns):,} of {released:,} discharged patients return; the median is {returns['days_to_return'].median():.1f} days after discharge.")
