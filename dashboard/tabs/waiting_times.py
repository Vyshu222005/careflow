"""Dashboard tab: waiting times (moved unchanged from app.py)."""
import plotly.graph_objects as go
import streamlit as st

import ui
from bottlenecks import clinical_steps, er_waits, format_hours, TARGET_HOURS, transition_stats
from dashboard.common import BLUE, INK, show, style


def render(ctx) -> None:
    events = ctx.events
    hours = ctx.hours
    antibiotics = ctx.antibiotics
    has_abx = ctx.has_abx
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
