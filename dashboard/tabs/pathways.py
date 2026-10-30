"""Dashboard tab: pathways (moved unchanged from app.py)."""
import plotly.graph_objects as go
import streamlit as st

import ui
from variants import SEPARATOR, shorten, variant_table
from dashboard.common import BLUE, show, style


def render(ctx) -> None:
    cases = ctx.cases
    ui.callout(
        "A <b>pathway</b> is the sequence of steps one patient went through. At the <b>clinical-step</b> level the repeated "
        "lab tests are left out; at the <b>exact</b> level every event counts, and almost every patient ends up unique."
    )
    level = st.radio("Level of detail", ["Clinical steps (lab tests left out)", "Exact (every event)"], horizontal=True)
    column = "milestone_variant" if level.startswith("Clinical") else "variant"
    table = variant_table(cases, column)
    once = table["cases"] == 1

    ui.spacer(0.4)
    columns = st.columns(4)
    ui.kpi(columns[0], "Distinct pathways", f"{len(table):,}", "different routes found", "route")
    ui.kpi(columns[1], "Covered by the top 10", f"{table['cumulative_pct'].iloc[min(10, len(table)) - 1]:.0f}%",
           "of selected patients", "check")
    ui.kpi(columns[2], "In a one-off pathway", f"{table.loc[once, 'share_pct'].sum():.0f}%",
           "followed by nobody else", "alert")
    ui.kpi(columns[3], "Most common pathway", f"{table['share_pct'].iloc[0]:.1f}%", f"{int(table['cases'].iloc[0]):,} patients follow it", "users")

    ui.spacer(1.0)
    with st.container(key="card_paths"):
        ui.section("Most common pathways", "Hover a bar to see the full pathway, the number of patients, and the median duration.")
        top_n = st.slider("Pathways to show", 5, 20, 10)
        top = table.head(top_n).iloc[::-1]
        labels = [shorten(v) for v in top["variant"]] if column == "milestone_variant" else [f"Pathway #{r}" for r in top["rank"]]
        fig = go.Figure(go.Bar(
            x=top["share_pct"], y=labels, orientation="h", marker_color=BLUE,
            customdata=list(zip(top["cases"], top["median_duration_days"], top["variant"].str.replace(SEPARATOR, " → "))),
            hovertemplate="<b>%{customdata[0]} patients (%{x:.1f}%)</b><br>Median duration: %{customdata[1]:.1f} days"
                          "<br>%{customdata[2]}<extra></extra>",
        ))
        fig.update_xaxes(title="Share of selected patients (%)")
        show(style(fig, 90 + 34 * top_n))

    ui.spacer(0.8)
    shown = table.head(top_n).copy()
    if column == "milestone_variant":
        shown["variant"] = shown["variant"].map(shorten)
    else:
        shown["variant"] = shown["variant"].str.replace(SEPARATOR, " → ")
    st.dataframe(
        shown[["variant", "cases", "share_pct", "cumulative_pct", "median_duration_days"]].rename(columns={
            "variant": "Pathway", "cases": "Patients", "share_pct": "Share",
            "cumulative_pct": "Cumulative %", "median_duration_days": "Median days",
        }),
        hide_index=True, width="stretch",
        column_config={
            "Pathway": st.column_config.TextColumn(width=470),
            "Patients": st.column_config.NumberColumn(width=75),
            "Share": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=float(shown["share_pct"].max()), width=135),
            "Cumulative %": st.column_config.NumberColumn(format="%.0f", width=100),
            "Median days": st.column_config.NumberColumn(format="%.1f", width=95),
        },
    )
