"""Dashboard tab: What-if Simulator (hypothetical scenarios for the antibiotic wait)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import simulator
import ui
from bottlenecks import format_hours
from dashboard.common import BLUE, INK, INK_SOFT, ORANGE, show, style
from load_data import CASE_ID

CUSTOM = "Custom scenario"


def render(ctx) -> None:
    st.warning("**Hypothetical results.** This tool changes recorded waiting times by arithmetic. It does not predict what would happen in a "
               "hospital and does not show that any change would improve patient outcomes.")
    ui.callout(
        "<b>If the early waits were shorter, how many more patients would be treated within the hour?</b> Each treated patient's time to antibiotics "
        "is split into <b>segment 1</b> (registration to sepsis triage) and <b>segment 2</b> (sepsis triage to antibiotics). Choose how much shorter each "
        "segment becomes and for how many patients; everything else stays as recorded."
    )
    hours = ctx.hours
    if simulator.ANTIBIOTICS not in hours or hours[simulator.ANTIBIOTICS].notna().sum() < 10:
        st.info("Fewer than 10 selected patients have antibiotics recorded, so a simulation would not be meaningful.")
        return

    choice = st.radio("Scenario", [*simulator.PRESETS, CUSTOM], horizontal=True, index=3)
    base = simulator.PRESETS.get(choice, simulator.Scenario())
    left, middle, right = st.columns(3)
    with left:
        cut1 = st.slider("Shorten segment 1 (registration to sepsis triage) by", 0, 80, int(base.cut_segment1_pct), step=5, format="%d%%", key=f"s1_{choice}")
        st.caption("Percent of the recorded wait.")
    with middle:
        cut2 = st.slider("Shorten segment 2 (sepsis triage to antibiotics) by", 0, 80, int(base.cut_segment2_pct), step=5, format="%d%%", key=f"s2_{choice}")
        minutes = st.slider("... and by a further", 0, 120, int(base.cut_segment2_minutes), step=5, format="%d min", key=f"m2_{choice}")
    with right:
        adoption = st.slider("Share of patients the change applies to", 10, 100, int(base.adoption_pct), step=10, format="%d%%", key=f"ad_{choice}")
        floor = st.slider("Segment 2 cannot go below", 0, 30, int(base.floor_minutes), step=1, format="%d min", key=f"fl_{choice}",
                          help="A lower limit so the scenario does not assume instant treatment.")
    scenario = simulator.Scenario(cut1, cut2, minutes, adoption, floor)
    result = simulator.simulate(hours, scenario)
    summary = result.summary().set_index("metric")

    ui.spacer(0.8)
    columns = st.columns(4)
    within = summary.loc["Treated within 1 hour (%)"]
    ui.kpi(columns[0], "Within 1 hour (hypothetical)", f"{within['simulated']:.0f}%", f"baseline {within['baseline']:.0f}%, change {within['change']:+.0f} points", "clock")
    ui.kpi(columns[1], "Newly within 1 hour", f"{result.newly_within_one_hour:,}", f"of {len(result.baseline_hours):,} treated patients", "check")
    median = summary.loc["Median hours to antibiotics"]
    ui.kpi(columns[2], "Median time (hypothetical)", format_hours(median["simulated"]), f"baseline {format_hours(median['baseline'])}", "drop")
    bundle = ctx.bundle_all[ctx.bundle_all[CASE_ID].isin(ctx.cases[CASE_ID])]
    base_bundle, sim_bundle = simulator.complete_bundle_share(bundle, result)
    ui.kpi(columns[3], "Complete bundle (hypothetical)", f"{sim_bundle:.0f}%", f"baseline {base_bundle:.0f}%; only antibiotic timing changed", "heart")

    ui.spacer(1.0)
    a, b = st.columns(2)
    with a, st.container(key="card_sim_dist"):
        ui.section("Time to antibiotics: baseline and hypothetical", "Hours from registration. Waits over 8 hours are grouped at 8.")
        fig = go.Figure()
        for series, name, color, opacity in ((result.baseline_hours, "Recorded (baseline)", "#98a2b3", 0.75), (result.simulated_hours, "Hypothetical", BLUE, 0.7)):
            fig.add_trace(go.Histogram(x=series.clip(upper=8), xbins=dict(start=0, end=8, size=0.5), name=name, marker_color=color, opacity=opacity))
        fig.update_layout(barmode="overlay")
        for target in simulator.TARGETS_HOURS:
            fig.add_vline(x=target, line_dash="dash", line_color=INK, annotation_text=f"{target:.0f} h", annotation_position="top")
        fig.update_xaxes(title="Hours from ER registration")
        fig.update_yaxes(title="Patients")
        fig = style(fig, 340, top=34)
        fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0))
        show(fig)
    with b, st.container(key="card_sim_sensitivity"):
        ui.section("How much does each extra minute saved help?", "Segment 2 shortened by a fixed number of minutes, with your other settings.")
        sweep = simulator.sensitivity(hours, list(range(0, 125, 5)), adoption_pct=adoption, cut_segment1_pct=cut1)
        fig = go.Figure(go.Scatter(x=sweep["minutes_saved"], y=sweep["within_1h_pct"], mode="lines+markers", line=dict(color=BLUE, width=2.5),
                                   hovertemplate="%{x} min saved: %{y:.0f}% within 1 hour<extra></extra>"))
        fig.add_hline(y=float(within["baseline"]), line_dash="dash", line_color=INK_SOFT, annotation_text="recorded", annotation_position="bottom right")
        fig.update_xaxes(title="Minutes saved in segment 2")
        fig.update_yaxes(title="Treated within 1 hour (%)", range=[0, 100])
        show(style(fig, 340, top=20))

    ui.spacer(1.0)
    with st.container(key="card_sim_table"):
        ui.section("Baseline against hypothetical")
        view = summary.reset_index().rename(columns={"metric": "Metric", "baseline": "Recorded", "simulated": "Hypothetical", "change": "Change"})
        st.dataframe(view.style.format({"Recorded": "{:.1f}", "Hypothetical": "{:.1f}", "Change": "{:+.1f}"}), hide_index=True, width="stretch")
        if scenario.is_baseline:
            st.caption("No change is selected, so the hypothetical equals the recorded data.")

    with st.expander("Assumptions of this simulation"):
        st.markdown(
            "- Time to antibiotics = **segment 1** (registration to sepsis triage) + **segment 2** (sepsis triage to antibiotics). Patients treated before sepsis triage keep their whole time in segment 1.\n"
            "- A segment is never made longer; segment 2 never falls below the floor you set (unless it was already shorter).\n"
            "- The change applies to a fixed, seeded subset of patients, so the same settings always give the same result.\n"
            "- Lab tests, IV fluid, ward stay and patient outcomes are left as recorded. The complete-bundle figure changes only through antibiotic timing.\n"
            "- Real changes to a process have costs, side effects and limits that this arithmetic ignores."
        )
