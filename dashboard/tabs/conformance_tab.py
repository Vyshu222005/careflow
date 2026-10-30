"""Dashboard tab: conformance tab (moved unchanged from app.py)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import conformance
import ui
from load_data import CASE_ID
from dashboard.common import BLUE, show, style


def render(ctx) -> None:
    cases = ctx.cases
    conformance_all = ctx.conformance_all
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
