"""Dashboard tab: departments (moved unchanged from app.py)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import handover
import ui
from bottlenecks import format_hours
from load_data import CASE_ID
from dashboard.common import BLUE, show, style


def render(ctx) -> None:
    cases = ctx.cases
    clean_log = ctx.clean_log
    ui.callout(
        "<b>Which departments handle patients, and who hands over to whom?</b> The log records the department for each step. The codes are "
        "anonymised, so each department's <b>role</b> is inferred from the steps it records (a department that only records blood tests is shown "
        "as Laboratory). Roles are an interpretation, not official names."
    )
    log_d = clean_log[clean_log[CASE_ID].isin(cases[CASE_ID])]
    dept = handover.department_table(log_d)
    hov = handover.handover_events(log_d)
    transfers = handover.ward_transfers(log_d)
    transitions = int(log_d.groupby(CASE_ID).size().sub(1).sum())
    changed_share = transfers["changed_department"].mean() * 100 if len(transfers) else None

    columns = st.columns(4)
    ui.kpi(columns[0], "Departments", f"{len(dept):,}", f"{int((dept['role'] == 'Ward').sum())} of them ward units", "users")
    ui.kpi(columns[1], "Handovers", f"{len(hov):,}", f"{len(hov) / max(transitions, 1) * 100:.0f}% of step-to-step moves", "repeat")
    ui.kpi(columns[2], "Busiest department", dept.iloc[0]["department"], f"{dept.iloc[0]['role']}, {dept.iloc[0]['share_of_events_pct']:.0f}% of events", "list")
    ui.kpi(columns[3], "Ward records that move",
           f"{changed_share:.0f}%" if changed_share is not None else "n/a", f"of {len(transfers):,} repeated ward records change unit", "route")

    ui.spacer(1.0)
    with st.container(key="card_dept_network"):
        ui.section("Handover network", "Arrows show handovers between departments; the number is how many times. Wider arrows mean more handovers.")
        controls = st.columns([1.2, 1.6])
        level = controls[0].radio("Level of detail", ["By role (simple)", "By department (12 busiest)"], horizontal=True)
        min_handovers = controls[1].slider("Smallest handover count shown", 5, 300, 20, step=5)
        if hov.empty:
            st.info("No handovers in this selection.")
        else:
            if level.startswith("By role"):
                pairs = handover.handover_pairs(hov, roles=dept.set_index("department")["role"])
                nodes = handover.role_nodes(dept)
            else:
                nodes = handover.department_nodes(dept, top=12)
                pairs = handover.handover_pairs(hov)
                pairs = pairs[pairs["from_department"].isin(nodes["node"]) & pairs["to_department"].isin(nodes["node"])]
            dot = handover.network_dot(pairs, nodes, min_handovers=min_handovers)
            st.graphviz_chart(dot, width="stretch")
            st.caption("Colours show the inferred role: blue ER team, light blue triage, green laboratory, orange ward, grey discharge and return. "
                       "The loop on Ward is patients moving from one ward unit to another.")

    ui.spacer(1.0)
    left, right = st.columns(2)
    with left, st.container(key="card_dept_workload"):
        ui.section("Events per department", "The ten busiest departments.")
        top = dept.head(10).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=top["events"], y=top["label"], orientation="h", marker_color=BLUE, customdata=list(zip(top["patients"], top["share_of_events_pct"])),
            hovertemplate="%{y}<br>%{x:,} events (%{customdata[1]:.1f}%), %{customdata[0]} patients<extra></extra>",
        ))
        fig.update_xaxes(title="Events")
        fig.update_yaxes(automargin=True)
        show(style(fig, 360))
    with right, st.container(key="card_dept_transfers"):
        ui.section("Moves between ward units", "Repeated ward records that are recorded by a different unit than the previous one.")
        moves = transfers[transfers["changed_department"]]
        if moves.empty:
            st.info("No repeated ward records that change unit in this selection.")
        else:
            top_moves = (moves.groupby(["previous_department", "department"]).agg(transfers=("gap_hours", "size"), median_gap=("gap_hours", "median"))
                         .reset_index().sort_values("transfers", ascending=False).head(8))
            top_moves["route"] = top_moves["previous_department"] + " → " + top_moves["department"]
            top_moves = top_moves.iloc[::-1]
            fig = go.Figure(go.Bar(
                x=top_moves["transfers"], y=top_moves["route"], orientation="h", marker_color=BLUE, customdata=top_moves["median_gap"],
                hovertemplate="%{y}: %{x} moves, median %{customdata:.0f} h between the records<extra></extra>",
            ))
            fig.update_xaxes(title="Moves")
            fig.update_yaxes(automargin=True)
            show(style(fig, 360))
            st.caption(f"{len(moves):,} of {len(transfers):,} repeated ward records ({changed_share:.0f}%) move to a different unit, a median of "
                       f"{moves['gap_hours'].median():.0f} hours after the previous record. They look like transfers, not duplicate entries.")

    ui.spacer(1.0)
    with st.container(key="card_dept_slow"):
        ui.section("Slowest handovers")
        slow = handover.handover_pairs(hov)
        slow = slow[slow["handovers"] >= 30].sort_values("median_wait_hours", ascending=False).head(8)
        if slow.empty:
            st.info("No pair of departments has enough handovers in this selection.")
        else:
            labels_by_department = dept.set_index("department")["label"]
            view = pd.DataFrame({
                "From": slow["from_department"].map(labels_by_department), "To": slow["to_department"].map(labels_by_department),
                "Handovers": slow["handovers"], "Median wait": slow["median_wait_hours"].map(format_hours),
                "90th percentile": slow["p90_wait_hours"].map(format_hours),
            })
            st.dataframe(view, hide_index=True, width="stretch")
            st.caption("Pairs of departments with at least 30 handovers, by median wait between the two events. "
                       "Waits involving the laboratory mostly reflect testing rhythm.")
