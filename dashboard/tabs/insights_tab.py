"""Dashboard tab: Smart Insights (transparent rule-based findings and operational suggestions)."""
import pandas as pd
import streamlit as st

import insights
import ui

SEVERITY_ICON = {"High": "🔴", "Medium": "🟠", "Low": "🟡", "Info": "🔵"}


@st.cache_data(show_spinner="Looking for bottlenecks and workflow issues...")
def _cached_insights(cache_key: tuple, _ctx):
    metrics = insights.collect_from_context(_ctx)
    return metrics, insights.generate(metrics)


def render(ctx) -> None:
    ui.callout(
        "<b>What deserves attention first?</b> The rules below look at the selected patients and flag findings that cross a visible threshold. "
        "Each insight shows <b>the rule that fired</b>, <b>why it matters</b> and <b>what to review</b>. These are suggestions about the process, "
        "<b>not clinical advice</b>."
    )
    key = (tuple(sorted(ctx.cases["case:concept:name"])), len(ctx.cases_all))
    metrics, found = _cached_insights(key, ctx)

    counts = pd.Series([i.severity for i in found]).value_counts()
    columns = st.columns(4)
    ui.kpi(columns[0], "High priority", f"{counts.get('High', 0)}", "findings", "alert")
    ui.kpi(columns[1], "Medium priority", f"{counts.get('Medium', 0)}", "findings", "list")
    ui.kpi(columns[2], "Low priority or notes", f"{counts.get('Low', 0) + counts.get('Info', 0)}", "findings", "check")
    ui.kpi(columns[3], "Patients analysed", f"{metrics['n_patients']:,}", ctx.filters_text if len(ctx.filters_text) < 38 else "filters applied", "users")

    ui.spacer(1.0)
    severities = st.multiselect("Show priorities", list(insights.SEVERITY_ORDER), default=list(insights.SEVERITY_ORDER), key="insight_filter")
    shown = [i for i in found if i.severity in severities]
    if not shown:
        st.info("No insights match this filter. Either nothing crosses a threshold for these patients, or the priority filter hides them.")
    for item in shown:
        with st.container(key=f"card_insight_{item.id}"):
            st.markdown(f"#### {SEVERITY_ICON[item.severity]} {item.title}")
            st.markdown(f"**{item.severity} priority.** {item.finding}")
            left, right = st.columns(2)
            left.markdown(f"**Why this rule fired**  \n{item.rule}")
            right.markdown(f"**What to review**  \n{item.action}")
            with st.expander("Why it matters, and the numbers"):
                st.markdown(item.explanation)
                st.json({k: (round(v, 2) if isinstance(v, float) else v) for k, v in item.evidence.items()}, expanded=False)
        ui.spacer(0.5)

    with st.expander("The thresholds behind these rules"):
        st.dataframe(pd.DataFrame({"Setting": list(insights.THRESHOLDS), "Value": list(insights.THRESHOLDS.values())}), hide_index=True, width="stretch")
        st.caption("These are working defaults for this project, set in src/insights.py. They are not clinical standards.")
