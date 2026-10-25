"""CareFlow dashboard: clinical pathway process mining on the Sepsis event log.

Run:
    streamlit run app.py

Needs the processed data, built with:
    python src/clean_data.py
    python src/prepare_log.py
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pm4py.visualization.dfg import visualizer as dfg_visualizer  # noqa: E402

from bottlenecks import (  # noqa: E402
    TARGET_HOURS, clinical_steps, er_waits, filtered_dfgs, format_hours, target_summary,
    time_from_registration, transition_stats,
)
from discovery import milestone_log  # noqa: E402
from load_data import ACTIVITY, CASE_ID, TIMESTAMP  # noqa: E402
from prepare_log import CASES_PATH, EVENTS_PATH, load_cases, load_events  # noqa: E402
from variants import SEPARATOR, add_milestone_variant, shorten, variant_table  # noqa: E402

BLUE = "#2a78d6"
AGE_ORDER = ["under 45", "45-64", "65-79", "80+"]
FIGURES = ROOT / "reports" / "figures"

st.set_page_config(page_title="CareFlow", page_icon="🩺", layout="wide")


# ---------- data ----------
@st.cache_data(show_spinner="Loading data...")
def load_data():
    events = load_events()
    cases = add_milestone_variant(events, load_cases())
    cases["start_date"] = cases["start"].dt.date
    return events, cases


def style(fig: go.Figure, height: int = 360) -> go.Figure:
    """Shared look: transparent background so it fits light and dark themes, light grid, no clutter."""
    fig.update_layout(
        height=height, margin=dict(l=0, r=10, t=10, b=0), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", bargap=0.35,
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


if not (EVENTS_PATH.exists() and CASES_PATH.exists()):
    st.title("CareFlow")
    st.error("The processed data files were not found. Build them first, then reload this page:")
    st.code("python src/download_data.py\npython src/clean_data.py\npython src/prepare_log.py", language="bash")
    st.stop()

events_all, cases_all = load_data()

# ---------- filters ----------
st.sidebar.title("Filters")
first_day, last_day = cases_all["start_date"].min(), cases_all["start_date"].max()
date_range = st.sidebar.date_input("Registered between", (first_day, last_day), min_value=first_day, max_value=last_day)
outcomes = cases_all["outcome"].value_counts().index.tolist()
chosen_outcomes = st.sidebar.multiselect("How the case ended", outcomes, default=outcomes)
chosen_ages = st.sidebar.multiselect("Age group", AGE_ORDER, default=AGE_ORDER)
returned = st.sidebar.radio("Returned to the ER", ["All", "Returned", "Did not return"], horizontal=True)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_day, end_day = date_range
else:
    start_day, end_day = first_day, last_day

mask = (
    cases_all["start_date"].between(start_day, end_day)
    & cases_all["outcome"].isin(chosen_outcomes)
    & cases_all["age_group"].isin(chosen_ages)
)
if returned == "Returned":
    mask &= cases_all["returned_to_er"]
elif returned == "Did not return":
    mask &= ~cases_all["returned_to_er"]

cases = cases_all[mask]
st.sidebar.metric("Patients selected", f"{len(cases):,} of {len(cases_all):,}")

st.title("CareFlow: Clinical Pathway Process Mining")
st.caption("Sepsis patients in a hospital emergency room: the real pathways they follow and where they wait.")

if cases.empty:
    st.warning("No patients match these filters. Widen the filters in the sidebar.")
    st.stop()
if len(cases) < 30:
    st.info(f"Only {len(cases)} patients are selected, so the numbers below can vary a lot.")

events = events_all[events_all[CASE_ID].isin(cases[CASE_ID])]
hours = time_from_registration(events)
antibiotics = target_summary(hours["IV Antibiotics"])

tab_overview, tab_paths, tab_map, tab_wait, tab_about = st.tabs(
    ["Overview", "Pathways", "Process map", "Waiting times", "About"]
)

# ---------- overview ----------
with tab_overview:
    columns = st.columns(5)
    columns[0].metric("Patients", f"{len(cases):,}")
    columns[1].metric("Median case duration", f"{cases['duration_days'].median():.1f} days")
    columns[2].metric("Returned to the ER", f"{cases['returned_to_er'].mean() * 100:.0f}%")
    columns[3].metric("No discharge recorded", f"{(cases['outcome'] == 'No release recorded').mean() * 100:.0f}%")
    columns[4].metric("Reached intensive care", f"{cases['reached_icu'].mean() * 100:.0f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Patients registered per month")
        monthly = cases.groupby("start_month").size()
        fig = go.Figure(go.Bar(x=monthly.index, y=monthly.values, marker_color=BLUE,
                               hovertemplate="%{x}: %{y} patients<extra></extra>"))
        fig.update_xaxes(type="category", tickangle=-45)
        show(style(fig))
    with right:
        st.subheader("How cases end")
        ended = cases["outcome"].value_counts().sort_values()
        fig = go.Figure(go.Bar(
            x=ended.values, y=ended.index, orientation="h", marker_color=BLUE,
            customdata=ended.values / len(cases) * 100,
            hovertemplate="%{y}: %{x} patients (%{customdata:.1f}%)<extra></extra>",
        ))
        show(style(fig))
        st.caption("`No release recorded` means the case has no discharge event in the data.")

    st.subheader("How long a case lasts")
    duration = cases["duration_days"]
    fig = go.Figure(go.Histogram(x=duration.clip(upper=120), xbins=dict(start=0, end=120, size=4),
                                 marker_color=BLUE, hovertemplate="%{x} days: %{y} patients<extra></extra>"))
    fig.update_xaxes(title="Days from first to last event (cases over 120 days grouped at 120)")
    fig.update_yaxes(title="Patients")
    show(style(fig, 300))

# ---------- pathways ----------
with tab_paths:
    level = st.radio(
        "Level of detail", ["Clinical steps (lab tests left out)", "Exact (every event)"], horizontal=True,
        help="Repeated lab tests make almost every exact pathway unique. The clinical-step level groups patients sensibly.",
    )
    column = "milestone_variant" if level.startswith("Clinical") else "variant"
    table = variant_table(cases, column)
    once = table["cases"] == 1

    columns = st.columns(3)
    columns[0].metric("Distinct pathways", f"{len(table):,}")
    columns[1].metric("Covered by the top 10", f"{table['cumulative_pct'].iloc[min(10, len(table)) - 1]:.0f}%")
    columns[2].metric("Patients in a one-off pathway", f"{table.loc[once, 'share_pct'].sum():.0f}%")

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
    show(style(fig, 80 + 34 * top_n))

    shown = table.head(top_n).copy()
    shown["variant"] = shown["variant"].str.replace(SEPARATOR, " → ")
    st.dataframe(
        shown[["rank", "variant", "cases", "share_pct", "cumulative_pct", "median_duration_days"]].rename(columns={
            "rank": "Rank", "variant": "Pathway", "cases": "Patients", "share_pct": "Share %",
            "cumulative_pct": "Cumulative %", "median_duration_days": "Median days",
        }),
        hide_index=True, width="stretch",
        column_config={"Share %": st.column_config.NumberColumn(format="%.1f"),
                       "Cumulative %": st.column_config.NumberColumn(format="%.1f"),
                       "Median days": st.column_config.NumberColumn(format="%.1f")},
    )

# ---------- process map ----------
with tab_map:
    steps = clinical_steps(events)
    controls = st.columns([1, 2])
    measure = controls[0].radio("Show on the arrows", ["Number of patients", "Median wait"])
    path_share = controls[1].slider("Share of most frequent paths drawn", 20, 100, 50, step=10, format="%d%%") / 100

    frequency, performance, start, end = filtered_dfgs(steps, path_share)
    if measure == "Number of patients":
        graph = dfg_visualizer.apply(
            frequency, variant=dfg_visualizer.Variants.FREQUENCY,
            parameters={"start_activities": start, "end_activities": end, "rankdir": "LR", "bgcolor": "transparent"},
        )
    else:
        graph = dfg_visualizer.apply(
            performance, variant=dfg_visualizer.Variants.PERFORMANCE,
            parameters={"start_activities": start, "end_activities": end, "rankdir": "LR",
                        "bgcolor": "transparent", "aggregationMeasure": "median"},
        )
    st.graphviz_chart(graph.source, width="stretch")
    st.caption(
        "Each box is a clinical step and each arrow means one step directly followed another. Lab tests are left out. "
        "Only the most frequent paths are drawn, so numbers can be slightly below the full totals. "
        "Wait labels: s = seconds, m = minutes, h = hours, D = days, MO = months."
    )

    with st.expander("Process models from the discovery step"):
        st.image(str(FIGURES / "process_heuristics_milestone.png"), caption="Heuristics net (exact counts)", width=520)
        st.image(str(FIGURES / "process_petri_inductive.png"), caption="Petri net (inductive miner, 20% noise filter)")
        evaluation = ROOT / "reports" / "model_evaluation.csv"
        if evaluation.exists():
            st.dataframe(pd.read_csv(evaluation), hide_index=True, width="stretch")

# ---------- waiting times ----------
with tab_wait:
    st.subheader("Time to antibiotics")
    columns = st.columns(4)
    columns[0].metric("Patients with antibiotics", f"{antibiotics['cases']:,}")
    columns[1].metric("Median from registration", format_hours(antibiotics["median_hours"]) if antibiotics["cases"] else "n/a")
    columns[2].metric(f"Within {TARGET_HOURS[0]} hour", f"{antibiotics['within_1h_pct']:.0f}%" if antibiotics["cases"] else "n/a")
    columns[3].metric(f"Within {TARGET_HOURS[1]} hours", f"{antibiotics['within_3h_pct']:.0f}%" if antibiotics["cases"] else "n/a")
    st.caption(
        "Antibiotics within 1 hour is the commonly quoted sepsis target. Here the clock starts at ER registration, "
        "not when sepsis was recognised, so the real share meeting the target is probably higher."
    )

    abx = hours["IV Antibiotics"].dropna()
    if len(abx):
        fig = go.Figure(go.Histogram(x=abx.clip(upper=10), xbins=dict(start=0, end=10, size=0.5), marker_color=BLUE,
                                     hovertemplate="%{x} h: %{y} patients<extra></extra>"))
        for target in TARGET_HOURS:
            fig.add_vline(x=target, line_dash="dash", line_color="gray", annotation_text=f"{target} h", annotation_position="top")
        fig.update_xaxes(title="Hours from ER registration (waits over 10 h grouped at 10)")
        fig.update_yaxes(title="Patients")
        show(style(fig, 300))

    st.subheader("Waits between ER steps")
    min_patients = st.slider("Minimum patients per step", 5, 100, 30, help="Hides rare transitions with too few patients.")
    transitions = transition_stats(clinical_steps(events))
    waits = er_waits(transitions, min_patients).sort_values("median_hours")
    if waits.empty:
        st.info("No transitions have enough patients with these filters.")
    else:
        fig = go.Figure(go.Bar(
            x=waits["median_hours"], y=[f"{a} → {b}" for a, b in zip(waits["from_step"], waits["to_step"])],
            orientation="h", marker_color=BLUE,
            customdata=list(zip(waits["cases"], waits["p90_hours"].map(format_hours), waits["median_hours"].map(format_hours))),
            hovertemplate="<b>Median %{customdata[2]}</b><br>90th percentile %{customdata[1]}<br>%{customdata[0]} patients<extra></extra>",
        ))
        fig.update_xaxes(title="Median wait (hours)")
        show(style(fig, 80 + 34 * len(waits)))

    with st.expander("All transitions"):
        view = transitions.rename(columns={
            "from_step": "From", "to_step": "To", "cases": "Patients", "median_hours": "Median (h)",
            "mean_hours": "Mean (h)", "p90_hours": "90th pct (h)", "kind": "Kind",
        })[["From", "To", "Kind", "Patients", "Median (h)", "Mean (h)", "90th pct (h)"]].round(2)
        st.dataframe(view, hide_index=True, width="stretch")
        st.download_button("Download as CSV", view.to_csv(index=False), "transitions.csv", "text/csv")
        st.caption("Time on the ward, in intensive care and before a return to the ER are long stays, not ER waits, "
                   "and are labelled in the Kind column.")

# ---------- about ----------
with tab_about:
    st.markdown(
        """
**Data:** *Sepsis Cases - Event Log* (Mannhardt, 2016, 4TU.ResearchData). Real, anonymised hospital data.
After cleaning, 995 patients and 14,208 events remain (see `reports/data_quality_report.md`).

**Method:** the event log is cleaned and prepared, then analysed with process mining (pm4py):
pathway variants, process discovery, and waiting times between steps.

**Reading the numbers**
- A *case* is one patient's journey; a *pathway* (variant) is the sequence of steps it followed.
- *Clinical steps* leave out the repeated lab tests (Leucocytes, CRP, LacticAcid); with them almost every patient has a unique pathway.
- Waits are medians over retrospective timestamps. They show where delays happen, not why.
- 26% of patients have no discharge event recorded; the reason is not in the data.

**Filters** in the sidebar apply to every tab.
"""
    )
