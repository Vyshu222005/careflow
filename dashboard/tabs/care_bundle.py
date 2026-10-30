"""Dashboard tab: care bundle (moved unchanged from app.py)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bundle
import ui
from load_data import CASE_ID
from dashboard.common import BLUE, INK_SOFT, show, style


def render(ctx) -> None:
    cases = ctx.cases
    bundle_all = ctx.bundle_all
    ui.callout(
        "<b>How many patients get the key early sepsis care on time, and is it changing?</b> The scorecard is <b>adapted</b> from the "
        "Surviving Sepsis Campaign hour-1 bundle using only what the log records. It is <b>not an official quality measure</b>: the clock "
        "starts at ER registration, and blood cultures and vasopressors are not recorded."
    )
    bt = bundle_all[bundle_all[CASE_ID].isin(cases[CASE_ID])]
    card_table = bundle.scorecard(bt).set_index("element")

    def pct_text(element: str) -> str:
        row = card_table.loc[element]
        return f"{row['met_pct']:.0f}%" if row["applicable"] else "n/a"

    columns = st.columns(4)
    ui.kpi(columns[0], "Complete bundle", pct_text("complete_bundle"), "every applicable element on time", "check")
    ui.kpi(columns[1], "Lactate in 1 hour", pct_text("lactate_1h"), "measured within an hour", "drop")
    ui.kpi(columns[2], "Antibiotics in 1 hour", pct_text("antibiotics_1h"), "of all selected patients", "clock")
    ui.kpi(columns[3], "IV fluid in 3 hours", pct_text("fluid_3h"),
           f"of {int(card_table.loc['fluid_3h', 'applicable'])} patients with hypotension", "heart")

    ui.spacer(1.0)
    with st.container(key="card_bundle_score"):
        ui.section("Scorecard", "Share of patients each element applies to. Hover for the numbers.")
        score = card_table.reset_index().iloc[::-1]
        fig = go.Figure(go.Bar(
            x=score["met_pct"], y=score["label"], orientation="h", marker_color=BLUE, customdata=list(zip(score["met"], score["applicable"])),
            hovertemplate="%{y}<br>%{x:.0f}% (%{customdata[0]} of %{customdata[1]} patients)<extra></extra>",
        ))
        fig.update_xaxes(title="Share of patients (%)", range=[0, 100])
        fig.update_yaxes(automargin=True)
        show(style(fig, 330))

    ui.spacer(1.0)
    with st.container(key="card_bundle_trend"):
        ui.section("Is care improving? Monthly control chart",
                   "Dashed line: the overall average. Grey lines: control limits. A month is flagged only when it differs by more than chance would explain.")
        labels_by_element = {e: bundle.ELEMENTS[e][0] for e in bundle.TREND_ELEMENTS}
        chosen = st.selectbox("Element", list(labels_by_element), format_func=labels_by_element.get)
        chart = bundle.control_chart(bt, chosen)
        if chart.empty:
            st.info(f"Not enough patients per month (at least {bundle.MIN_MONTH_N}) in this selection to draw a control chart.")
        else:
            months = pd.to_datetime(chart["month"] + "-01").dt.strftime("%b %y").tolist()
            fig = go.Figure()
            for column, name, show_legend in (("ucl", "Control limits", True), ("lcl", "Control limits", False)):
                fig.add_trace(go.Scatter(x=months, y=chart[column], mode="lines", line=dict(color="#98a2b3", width=1.2, shape="hv"),
                                         name=name, showlegend=show_legend, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=months, y=[chart["center"].iloc[0]] * len(chart), mode="lines", line=dict(color=INK_SOFT, dash="dash", width=1.2),
                                     name=f"Average ({chart['center'].iloc[0]:.0f}%)", hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=months, y=chart["pct"], mode="lines+markers", name="Monthly rate", line=dict(color=BLUE, width=2.5), marker=dict(size=8),
                customdata=list(zip(chart["met"], chart["n"])),
                hovertemplate="%{x}: %{y:.0f}% (%{customdata[0]} of %{customdata[1]} patients)<extra></extra>",
            ))
            flagged = chart[chart["signal"] != ""]
            if not flagged.empty:
                fig.add_trace(go.Scatter(
                    x=[months[i] for i in flagged.index], y=flagged["pct"], mode="markers+text", name="Signal", text=flagged["signal"],
                    textposition="top center", marker=dict(color="#eb6834", size=13, line=dict(color="white", width=2)),
                    hovertemplate="%{x}: %{y:.0f}%<br>%{text}<extra></extra>",
                ))
            fig.update_xaxes(type="category", tickangle=-45)
            fig.update_yaxes(range=[0, 100], title="Patients meeting the element (%)")
            fig = style(fig, 400, top=44)
            fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            show(fig)
            if flagged.empty:
                st.caption("No signals: the month-to-month changes are what chance alone would produce.")
            else:
                st.caption("Signals: " + "; ".join(f"{m} ({s})" for m, s in zip(flagged['month'], flagged['signal'])) + f". Months with fewer than {bundle.MIN_MONTH_N} patients are left out.")

    with st.expander("How the elements are defined"):
        st.markdown(
            "- **Lactate in 1 hour / antibiotics in 1 hour:** within 1 hour of ER registration, for all patients (antibiotics in 3 hours is the looser reference).\n"
            "- **IV fluid in 3 hours:** only for patients with the hypotension flag.\n"
            "- **Lactate measured again:** only for patients whose first lactate value is above 2.\n"
            "- **Complete bundle:** the patient meets every element that applies to them.\n"
            "- The control limits are the overall rate plus or minus 3 standard errors; a run of 8 months on one side of the average is also flagged."
        )
        st.download_button("Download the per-patient results (CSV)", bt.to_csv(index=False), "care_bundle.csv", "text/csv")
