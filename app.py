"""CareFlow dashboard: clinical pathway process mining on the Sepsis event log.

Run:
    streamlit run app.py

Needs the processed data, built with:
    python src/clean_data.py
    python src/prepare_log.py

Look and feel: assets/style.css (stylesheet) and src/ui.py (HTML cards and header).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pm4py.visualization.dfg import visualizer as dfg_visualizer  # noqa: E402

import conformance  # noqa: E402
import rework  # noqa: E402
import ui  # noqa: E402
from bottlenecks import (  # noqa: E402
    TARGET_HOURS, clinical_steps, er_waits, filtered_dfgs, format_hours, target_summary,
    time_from_registration, transition_stats,
)
from clean_data import CLEAN_PATH  # noqa: E402
from load_data import ACTIVITY, CASE_ID, TIMESTAMP  # noqa: E402
from prepare_log import CASES_PATH, EVENTS_PATH, load_cases, load_events  # noqa: E402
from variants import SEPARATOR, add_milestone_variant, shorten, variant_table  # noqa: E402

BLUE = "#2a78d6"
INK = "#1d2939"
INK_SOFT = "#475467"
GRID = "rgba(102,112,133,0.18)"
AGE_ORDER = ["under 45", "45-64", "65-79", "80+"]
RETURN_OPTIONS = ["All", "Returned", "Did not return"]
FIGURES = ROOT / "reports" / "figures"

st.set_page_config(page_title="CareFlow | Clinical Pathway Process Mining", page_icon="🩺", layout="wide")
ui.inject_css()


# ---------- data and chart helpers ----------
@st.cache_data(show_spinner="Loading data...")
def load_data():
    events = load_events()
    cases = add_milestone_variant(events, load_cases())
    cases["start_date"] = cases["start"].dt.date
    return events, cases


@st.cache_data(show_spinner="Checking conformance and rework...")
def load_conformance_rework():
    """Conformance and rework use the cleaned log with every discharge code kept (see src/conformance.py)."""
    log = conformance.load_log()
    conformance_table = conformance.check_conformance(log)
    rework_table = rework.rework_per_case(rework.clinical_steps(log))
    return log, conformance_table, rework_table


def style(fig: go.Figure, height: int = 340, top: int = 8) -> go.Figure:
    """Shared chart look: transparent background, light grid, soft text."""
    fig.update_layout(
        height=height, margin=dict(l=0, r=12, t=top, b=0), showlegend=False, bargap=0.35,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, Roboto, sans-serif", size=12, color=INK_SOFT),
        hoverlabel=dict(bgcolor="white", font_size=12, font_color=INK, bordercolor="#d0d5dd"),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, automargin=True)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def median_return_days(events: pd.DataFrame):
    """Median days from the first discharge to a return to the ER."""
    release = events[events[ACTIVITY].str.startswith("Release")].groupby(CASE_ID)[TIMESTAMP].min()
    back = events[events[ACTIVITY] == "Return ER"].groupby(CASE_ID)[TIMESTAMP].min()
    gap = (back - release.reindex(back.index)).dropna().dt.total_seconds() / 86400
    return gap.median() if len(gap) else None


if not (EVENTS_PATH.exists() and CASES_PATH.exists() and CLEAN_PATH.exists()):
    st.title("CareFlow")
    st.error("The processed data files were not found. Build them first, then reload this page:")
    st.code("python src/download_data.py\npython src/clean_data.py\npython src/prepare_log.py", language="bash")
    st.stop()

events_all, cases_all = load_data()
clean_log, conformance_all, rework_all = load_conformance_rework()

# ---------- sidebar: brand and filters ----------
first_day, last_day = cases_all["start_date"].min(), cases_all["start_date"].max()
outcomes = cases_all["outcome"].value_counts().index.tolist()
DEFAULTS = {"f_dates": (first_day, last_day), "f_outcomes": [], "f_ages": [], "f_returned": "All"}  # empty list = no filter
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def reset_filters() -> None:
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


ui.sidebar_brand()
ui.sidebar_title("Filters")
date_range = st.sidebar.date_input("Registered between", min_value=first_day, max_value=last_day, key="f_dates")
chosen_outcomes = st.sidebar.multiselect("How the case ended", outcomes, key="f_outcomes", placeholder="All outcomes")
chosen_ages = st.sidebar.multiselect("Age group", AGE_ORDER, key="f_ages", placeholder="All ages")
returned = st.sidebar.radio("Returned to the ER", RETURN_OPTIONS, key="f_returned")
st.sidebar.button("Reset filters", on_click=reset_filters, width="stretch")

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_day, end_day = date_range
else:
    start_day, end_day = first_day, last_day

mask = cases_all["start_date"].between(start_day, end_day)
if chosen_outcomes:
    mask &= cases_all["outcome"].isin(chosen_outcomes)
if chosen_ages:
    mask &= cases_all["age_group"].isin(chosen_ages)
if returned == "Returned":
    mask &= cases_all["returned_to_er"]
elif returned == "Did not return":
    mask &= ~cases_all["returned_to_er"]

cases = cases_all[mask]
ui.selected_box(len(cases), len(cases_all))

# ---------- header ----------
chips = [f"{len(cases_all):,} patients", f"{len(events_all):,} events", f"{first_day:%b %Y} to {last_day:%b %Y}",
         "Source: Sepsis Cases event log (4TU)"]
if len(cases) != len(cases_all):
    chips.append(f"Filtered: {len(cases):,} patients")
ui.hero("How sepsis patients move through the emergency room, which pathways they follow, and where they wait.", chips)

if cases.empty:
    st.warning("No patients match these filters. Widen the filters in the sidebar or press Reset filters.")
    st.stop()
if len(cases) < 30:
    st.info(f"Only {len(cases)} patients are selected, so the numbers below can vary a lot.")

events = events_all[events_all[CASE_ID].isin(cases[CASE_ID])]
hours = time_from_registration(events)
antibiotics = target_summary(hours["IV Antibiotics"])
milestone_table = variant_table(cases, "milestone_variant")
has_abx = antibiotics["cases"] > 0

tab_overview, tab_paths, tab_map, tab_wait, tab_conf, tab_rework, tab_about = st.tabs(
    ["Overview", "Pathways", "Process map", "Waiting times", "Conformance", "Rework", "About"]
)

# ---------- overview ----------
with tab_overview:
    columns = st.columns(5)
    ui.kpi(columns[0], "Patients", f"{len(cases):,}", f"of {len(cases_all):,} in total", "users")
    ui.kpi(columns[1], "Events", f"{len(events):,}", "recorded steps", "list")
    ui.kpi(columns[2], "Median duration", f"{cases['duration_days'].median():.1f} days", "first to last event", "clock")
    ui.kpi(columns[3], "Time to antibiotics", format_hours(antibiotics["median_hours"]) if has_abx else "n/a",
           "median from registration", "drop")
    ui.kpi(columns[4], "Intensive care", f"{cases['reached_icu'].mean() * 100:.0f}%", "of patients reach ICU", "heart")

    ui.spacer(1.2)
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

    ui.spacer(1.2)
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

# ---------- pathways ----------
with tab_paths:
    ui.callout(
        "A <b>pathway</b> is the sequence of steps one patient went through. At the <b>clinical-step</b> level the repeated "
        "lab tests are left out; at the <b>exact</b> level every event counts, and almost every patient ends up unique."
    )
    level = st.radio("Level of detail", ["Clinical steps (lab tests left out)", "Exact (every event)"], horizontal=True)
    column = "milestone_variant" if level.startswith("Clinical") else "variant"
    table = variant_table(cases, column)
    once = table["cases"] == 1

    ui.spacer(0.4)
    columns = st.columns(3)
    ui.kpi(columns[0], "Distinct pathways", f"{len(table):,}", "different routes found", "route")
    ui.kpi(columns[1], "Covered by the top 10", f"{table['cumulative_pct'].iloc[min(10, len(table)) - 1]:.0f}%",
           "of selected patients", "check")
    ui.kpi(columns[2], "In a one-off pathway", f"{table.loc[once, 'share_pct'].sum():.0f}%",
           "followed by nobody else", "alert")

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

# ---------- process map ----------
with tab_map:
    ui.callout(
        "<b>How to read the map:</b> each box is a clinical step and each arrow means one step directly followed another. "
        "Darker boxes and thicker arrows mean more patients. Lab tests are left out so the main route stays visible."
    )
    steps = clinical_steps(events)
    controls = st.columns([1, 1, 1.4])
    measure = controls[0].radio("Show on the arrows", ["Number of patients", "Median wait"])
    direction = controls[1].radio("Direction", ["Top to bottom", "Left to right"])
    path_share = controls[2].slider("Share of most frequent paths drawn", 20, 100, 50, step=10, format="%d%%") / 100
    rankdir = "TB" if direction == "Top to bottom" else "LR"

    frequency, performance, start, end = filtered_dfgs(steps, path_share)
    if measure == "Number of patients":
        graph = dfg_visualizer.apply(
            frequency, variant=dfg_visualizer.Variants.FREQUENCY,
            parameters={"start_activities": start, "end_activities": end, "rankdir": rankdir, "bgcolor": "transparent"},
        )
    else:
        graph = dfg_visualizer.apply(
            performance, variant=dfg_visualizer.Variants.PERFORMANCE,
            parameters={"start_activities": start, "end_activities": end, "rankdir": rankdir,
                        "bgcolor": "transparent", "aggregationMeasure": "median"},
        )
    with st.container(key="card_map"):
        ui.section("Patient flow between clinical steps", "Use the arrows at the top right of the map to view it full screen.")
        st.graphviz_chart(graph.source, width="stretch")
    st.caption(
        "Only the most frequent paths are drawn, so numbers can be slightly below the full totals. "
        "Wait labels: s = seconds, m = minutes, h = hours, D = days, MO = months."
    )

    with st.expander("Process models from the discovery step"):
        st.image(str(FIGURES / "process_heuristics_milestone.png"), caption="Heuristics net (exact counts)", width=520)
        st.image(str(FIGURES / "process_petri_inductive.png"), caption="Petri net (inductive miner, 20% noise filter)")
        evaluation = ROOT / "reports" / "model_evaluation.csv"
        if evaluation.exists():
            st.markdown("**How well each model fits the data**")
            scores = pd.read_csv(evaluation).replace({"milestone": "clinical steps"}).rename(columns={
                "model": "Model", "log": "Log", "fitness": "Fitness", "perfectly_fitting_cases_pct": "Perfectly fitting cases (%)",
                "precision": "Precision", "f_score": "F-score", "places": "Places", "transitions": "Transitions", "arcs": "Arcs",
            })
            st.dataframe(scores, hide_index=True, width="stretch")

# ---------- waiting times ----------
with tab_wait:
    ui.callout(
        "<b>Where do patients wait?</b> A wait is the time between two consecutive clinical steps. "
        "The clock for antibiotics starts at <b>ER registration</b>, not at recognition of sepsis, so the real share "
        "meeting the 1-hour target is probably higher."
    )
    columns = st.columns(4)
    ui.kpi(columns[0], "With antibiotics", f"{antibiotics['cases']:,}", "patients treated", "users")
    ui.kpi(columns[1], "Median wait", format_hours(antibiotics["median_hours"]) if has_abx else "n/a",
           "registration to antibiotics", "clock")
    ui.kpi(columns[2], f"Within {TARGET_HOURS[0]} hour", f"{antibiotics['within_1h_pct']:.0f}%" if has_abx else "n/a",
           "of treated patients", "drop")
    ui.kpi(columns[3], f"Within {TARGET_HOURS[1]} hours", f"{antibiotics['within_3h_pct']:.0f}%" if has_abx else "n/a",
           "of treated patients", "check")

    ui.spacer(1.0)
    abx = hours["IV Antibiotics"].dropna()
    with st.container(key="card_abx"):
        ui.section("Time from registration to first antibiotics", "Dashed lines mark 1 hour and 3 hours.")
        if len(abx):
            fig = go.Figure(go.Histogram(x=abx.clip(upper=10), xbins=dict(start=0, end=10, size=0.5), marker_color=BLUE,
                                         hovertemplate="%{x} h: %{y} patients<extra></extra>"))
            for target in TARGET_HOURS:
                fig.add_vline(x=target, line_dash="dash", line_color=INK, annotation_text=f"{target} h",
                              annotation_position="top")
            fig.update_xaxes(title="Hours from ER registration (waits over 10 h are grouped at 10)")
            fig.update_yaxes(title="Patients")
            show(style(fig, 320, top=34))
        else:
            st.info("No patients with recorded antibiotics in this selection.")

    ui.spacer(1.0)
    with st.container(key="card_waits"):
        ui.section("Waits between ER steps", "Median wait before the next step. Hover a bar for the 90th percentile.")
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
                hovertemplate="<b>Median %{customdata[2]}</b><br>90th percentile %{customdata[1]}"
                              "<br>%{customdata[0]} patients<extra></extra>",
            ))
            fig.update_xaxes(title="Median wait (hours)")
            show(style(fig, 90 + 34 * len(waits)))

    ui.spacer(0.8)
    with st.expander("All transitions"):
        view = transitions.rename(columns={
            "from_step": "From", "to_step": "To", "cases": "Patients", "median_hours": "Median (h)",
            "mean_hours": "Mean (h)", "p90_hours": "90th pct (h)", "kind": "Kind",
        })[["From", "To", "Kind", "Patients", "Median (h)", "Mean (h)", "90th pct (h)"]].round(2)
        st.dataframe(view, hide_index=True, width="stretch")
        st.download_button("Download as CSV", view.to_csv(index=False), "transitions.csv", "text/csv")
        st.caption("Time on the ward, in intensive care and before a return to the ER are long stays, not ER waits, "
                   "and are labelled in the Kind column.")

# ---------- conformance ----------
with tab_conf:
    ui.callout(
        "<b>Do patients follow the expected sepsis pathway?</b> Each patient is compared with an expected route: "
        "registration, triage, sepsis triage, antibiotics (IV fluid optional), ward admission, then discharge. "
        "This expected route is an <b>assumption</b> that still has to be confirmed with the mentor."
    )
    conf = conformance_all[conformance_all[CASE_ID].isin(cases[CASE_ID])]
    treated = conf[~conf["no_antibiotics"]]
    late_share = treated["late_antibiotics"].mean() * 100 if len(treated) else None

    columns = st.columns(4)
    ui.kpi(columns[0], "Follow the pathway", f"{conf['conforms'].mean() * 100:.0f}%", f"{int(conf['conforms'].sum()):,} of {len(conf):,} patients", "check")
    ui.kpi(columns[1], "Average fitness", f"{conf['fitness'].mean():.2f}", "1.00 means an exact match", "route")
    ui.kpi(columns[2], "No discharge recorded", f"{conf['no_discharge'].mean() * 100:.0f}%", "of selected patients", "alert")
    ui.kpi(columns[3], "Late antibiotics", f"{late_share:.0f}%" if late_share is not None else "n/a", "of treated patients, after 1 hour", "clock")

    ui.spacer(1.0)
    left, right = st.columns([3, 2])
    with left, st.container(key="card_conf_dev"):
        ui.section("Deviations from the expected pathway", "Share of selected patients. One patient can have several deviations.")
        rules = conformance.deviation_summary(conf).query("kind == 'rule'").sort_values("share_pct")
        fig = go.Figure(go.Bar(
            x=rules["share_pct"], y=rules["deviation"], orientation="h", marker_color=BLUE, customdata=rules["cases"],
            hovertemplate="%{y}<br>%{customdata} patients (%{x:.1f}%)<extra></extra>",
        ))
        fig.update_xaxes(title="Share of patients (%)")
        fig.update_yaxes(automargin=True)
        show(style(fig, 340))
    with right, st.container(key="card_conf_fit"):
        ui.section("How closely patients follow it", "Fitness: 1.0 means the patient follows the pathway exactly.")
        edges = [0, 0.7, 0.8, 0.9, 0.9999999, 1.01]
        labels = ["under 0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0", "exactly 1.0"]
        groups = pd.cut(conf["fitness"], bins=edges, labels=labels, right=False).value_counts().reindex(labels)
        fig = go.Figure(go.Bar(
            x=groups.index, y=groups.values, marker_color=BLUE, customdata=groups.values / max(len(conf), 1) * 100,
            hovertemplate="%{x}: %{y} patients (%{customdata:.0f}%)<extra></extra>",
        ))
        fig.update_yaxes(title="Patients")
        show(style(fig, 340))

    ui.spacer(1.0)
    with st.container(key="card_conf_patterns"):
        ui.section("Most common deviation patterns", "Steps that are missing (expected but not recorded) and extra (recorded but not expected).")
        patterns = conformance.deviation_patterns(conf, top=8).iloc[::-1]
        if patterns.empty:
            st.info("Every selected patient follows the expected pathway exactly.")
        else:
            fig = go.Figure(go.Bar(
                x=patterns["share_of_all_pct"], y=patterns["description"], orientation="h", marker_color=BLUE, customdata=patterns["cases"],
                hovertemplate="%{y}<br>%{customdata} patients (%{x:.1f}%)<extra></extra>",
            ))
            fig.update_xaxes(title="Share of selected patients (%)")
            fig.update_yaxes(automargin=True)
            show(style(fig, 90 + 36 * len(patterns)))

    with st.expander("The expected pathway and how to read the results"):
        st.markdown(
            "Written as a process tree: `->` is a sequence, `+` runs steps in parallel, `X` is a choice, `tau` means the step can be skipped."
        )
        st.code(conformance.EXPECTED_PATHWAY, language="text", wrap_lines=True)
        st.markdown(
            "- **Missing** steps are expected but not recorded; **extra** steps are recorded but not expected.\n"
            "- Timing is **not** part of the formal check, only the order of steps. Late antibiotics are shown separately.\n"
            "- Changing the pathway in `src/conformance.py` changes every number on this tab."
        )
        st.download_button("Download the per-patient results (CSV)", conf.to_csv(index=False), "conformance.csv", "text/csv")

# ---------- rework ----------
with tab_rework:
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

# ---------- about ----------
with tab_about:
    left, right = st.columns(2)
    with left, st.container(key="card_about_data"):
        ui.section("The data")
        st.markdown(
            "*Sepsis Cases - Event Log* (Mannhardt, 2016, 4TU.ResearchData): real, anonymised hospital data. "
            "After cleaning, **995 patients** and **14,208 events** remain (see `reports/data_quality_report.md`)."
        )
        ui.section("The method")
        st.markdown(
            "The event log is cleaned and prepared, then analysed with process mining (pm4py): "
            "pathway variants, process discovery, and waiting times between steps."
        )
    with right, st.container(key="card_about_read"):
        ui.section("Reading the numbers")
        st.markdown(
            "- A **case** is one patient's journey; a **pathway** is the sequence of steps it followed.\n"
            "- **Clinical steps** leave out the repeated lab tests (Leucocytes, CRP, LacticAcid).\n"
            "- Waits are medians over retrospective timestamps. They show where delays happen, not why.\n"
            "- 26% of patients have no discharge event recorded; the reason is not in the data.\n"
            "- IV fluid and antibiotics are often recorded at the same moment, so they may be entered together rather than given together.\n"
            "- **Conformance** compares each patient with an expected pathway that is an assumption, to be confirmed with the mentor.\n"
            "- **Rework** is mostly repeated ward admission records; true loops back to an earlier step are rare."
        )

ui.footer("CareFlow: clinical pathway process mining. Data: Sepsis Cases event log (4TU.ResearchData).")
