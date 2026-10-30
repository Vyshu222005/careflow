"""Dashboard tab: process map (moved unchanged from app.py)."""
import pandas as pd
import streamlit as st

import ui
from pm4py.visualization.dfg import visualizer as dfg_visualizer
from bottlenecks import clinical_steps, filtered_dfgs
from dashboard.common import FIGURES, ROOT


def render(ctx) -> None:
    events = ctx.events
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
