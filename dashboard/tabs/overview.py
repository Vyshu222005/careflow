"""Dashboard tab: overview (moved unchanged from app.py)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from bottlenecks import clinical_steps, er_waits, format_hours, transition_stats
from load_data import CASE_ID
from dashboard.common import BLUE, INK, median_return_days, show, style


def render(ctx) -> None:
    cases_all = ctx.cases_all
    cases = ctx.cases
    events = ctx.events
    antibiotics = ctx.antibiotics
    milestone_table = ctx.milestone_table
    has_abx = ctx.has_abx
    clean_log = ctx.clean_log
    columns = st.columns(4)
    ui.kpi(columns[0], "Patients", f"{len(cases):,}", f"of {len(cases_all):,} in total", "users")
    excluded = ctx.cleaned_events_selected - len(events)
    ui.kpi(columns[1], "Events", f"{ctx.cleaned_events_selected:,}", f"cleaned; {excluded} rare-step events not analysed" if excluded else "cleaned events, all analysed", "list")
    ui.kpi(columns[2], "Median duration", f"{cases['duration_days'].median():.1f} days", "first to last event", "clock")
    ui.kpi(columns[3], "Time to antibiotics", format_hours(antibiotics["median_hours"]) if has_abx else "n/a",
           "median from registration", "drop")

    ids = set(cases[CASE_ID])
    conf = ctx.conformance_all[ctx.conformance_all[CASE_ID].isin(ids)]
    bundle_rows = ctx.bundle_all[ctx.bundle_all[CASE_ID].isin(ids)]
    without_antibiotics = len(cases) - antibiotics["cases"]
    columns = st.columns(4)
    ui.kpi(columns[0], "Intensive care", f"{cases['reached_icu'].mean() * 100:.0f}%", "of patients reach ICU", "heart")
    ui.kpi(columns[1], "No antibiotics recorded", f"{without_antibiotics / len(cases) * 100:.0f}%", f"{without_antibiotics:,} patients", "alert")
    ui.kpi(columns[2], "Follow expected pathway", f"{conf['conforms'].mean() * 100:.0f}%", f"average fitness {conf['fitness'].mean():.2f}", "route")
    ui.kpi(columns[3], "Complete care bundle", f"{bundle_rows['complete_bundle'].mean() * 100:.0f}%", "all applicable elements on time", "check")

    ui.spacer(0.8)
    ui.section("Key insights", "Calculated for the patients selected in the sidebar.")
    insight = st.columns(4)
    ui.insight(insight[0], "Treatment speed", f"{antibiotics['within_1h_pct']:.0f}%" if has_abx else "n/a",
               "of patients receive antibiotics within 1 hour of registration (commonly quoted sepsis target).", "drop")
    return_days = median_return_days(clean_log[clean_log[CASE_ID].isin(cases[CASE_ID])])  # same log as the Rework tab
    ui.insight(insight[1], "Return visits", f"{cases['returned_to_er'].mean() * 100:.0f}%",
               "of patients return to the ER" + (f", a median of {return_days:.1f} days after discharge." if return_days else "."),
               "repeat")
    ui.insight(insight[2], "Missing endings", f"{(cases['outcome'] == 'No release recorded').mean() * 100:.0f}%",
               "of patients have no discharge event recorded, so their pathway stops early in the data.", "alert")
    top3 = milestone_table["cumulative_pct"].iloc[min(3, len(milestone_table)) - 1]
    ui.insight(insight[3], "Common pathways", f"{top3:.0f}%",
               "of patients follow one of the 3 most common clinical pathways.", "route")

    ui.spacer(0.8)
    left, right = st.columns(2)
    with left, st.container(key="card_month"):
        ui.section("Patients registered per month")
        monthly = cases.groupby("start_month").size()
        labels = pd.to_datetime(monthly.index + "-01").strftime("%b %y")
        fig = go.Figure(go.Bar(x=labels, y=monthly.values, marker_color=BLUE,
                               hovertemplate="%{x}: %{y} patients<extra></extra>"))
        peak = int(monthly.values.argmax())
        fig.add_annotation(x=labels[peak], y=monthly.values[peak], text=f"peak: {monthly.values[peak]}",
                           showarrow=False, yshift=12, font=dict(color=INK, size=11))
        fig.update_xaxes(type="category", tickangle=-45)
        fig.update_yaxes(title="Patients")
        show(style(fig))
    with right, st.container(key="card_outcome"):
        ui.section("How cases end", "No release recorded = no discharge event in the data.")
        ended = cases["outcome"].value_counts().sort_values()
        fig = go.Figure(go.Bar(
            x=ended.values, y=ended.index, orientation="h", marker_color=BLUE,
            customdata=ended.values / len(cases) * 100,
            hovertemplate="%{y}: %{x} patients (%{customdata:.1f}%)<extra></extra>",
        ))
        fig.update_xaxes(title="Patients")
        show(style(fig, 322))

    ui.spacer(1.0)
    with st.container(key="card_duration"):
        ui.section("How long a case lasts", "Time from the first to the last recorded event.")
        duration = cases["duration_days"]
        fig = go.Figure(go.Histogram(x=duration.clip(upper=120), xbins=dict(start=0, end=120, size=4),
                                     marker_color=BLUE, hovertemplate="%{x} days: %{y} patients<extra></extra>"))
        fig.add_vline(x=min(duration.median(), 120), line_dash="dash", line_color=INK,
                      annotation_text=f"median {duration.median():.1f} days", annotation_position="top right")
        fig.update_xaxes(title="Days (cases over 120 days are grouped at 120)")
        fig.update_yaxes(title="Patients")
        show(style(fig, 320, top=30))

    ui.spacer(1.0)
    with st.container(key="card_top_bottlenecks"):
        ui.section("Top operational bottlenecks",
                   "Where selected patients wait longest before the next step. Full detail is in Bottleneck Analysis.")
        rows = []
        if has_abx:
            late = (1 - antibiotics["within_1h_pct"] / 100) * 100
            rows.append({"Where": "Registration → first antibiotics", "Median wait": format_hours(antibiotics["median_hours"]),
                         "Patients": antibiotics["cases"], "Why it matters": f"{late:.0f}% of treated patients wait more than 1 hour"})
        waits = er_waits(transition_stats(clinical_steps(events)), 30).head(3)
        for _, row in waits.iterrows():
            rows.append({"Where": f"{row['from_step']} → {row['to_step']}", "Median wait": format_hours(row["median_hours"]),
                         "Patients": int(row["cases"]), "Why it matters": f"9 in 10 patients wait less than {format_hours(row['p90_hours'])}"})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.info("Not enough patients in this selection to rank bottlenecks.")
