"""Department handover network: who hands patients to whom, and what the ward "rework" really is.

The log records the department (org:group) that performed each step. The codes are anonymised (A, B,
C, ...), so a department's ROLE is inferred from the steps it records (for example, a department that
only records blood tests is shown as "Laboratory"). Roles are therefore an interpretation, not
official names.

A handover is two consecutive events of one patient recorded by different departments. The handover
wait is the time between those two events.

Usage:
    python src/handover.py

Outputs:
    reports/department_workload.csv   events and patients per department, with the inferred role
    reports/handover_pairs.csv        handovers between departments, with counts and waits
    reports/ward_transfers.csv        where repeated ward admission records move from and to
"""
from pathlib import Path

import numpy as np
import pandas as pd

from conformance import load_log
from load_data import ACTIVITY, CASE_ID, TIMESTAMP
from variants import LAB_ACTIVITIES

ROOT = Path(__file__).resolve().parents[1]
WORKLOAD_PATH = ROOT / "reports" / "department_workload.csv"
PAIRS_PATH = ROOT / "reports" / "handover_pairs.csv"
WARD_TRANSFERS_PATH = ROOT / "reports" / "ward_transfers.csv"

GROUP = "org:group"
WARD = "Admission NC"
ROLE_MIN_SHARE = 0.5  # a department gets a role when one kind of step is at least this share of its events

ROLE_OF_ACTIVITY = {
    "ER Registration": "ER team", "ER Sepsis Triage": "ER team", "IV Liquid": "ER team", "IV Antibiotics": "ER team",
    "ER Triage": "Triage",
    "Admission NC": "Ward", "Admission IC": "Ward",
    "Return ER": "Return (unit not recorded)",
}
ROLE_OF_ACTIVITY.update({name: "Laboratory" for name in LAB_ACTIVITIES})
ROLE_COLORS = {  # fill colour and text colour
    "ER team": ("#2A78D6", "white"), "Triage": ("#7FB2EC", "#1D2939"), "Laboratory": ("#1BAF7A", "white"),
    "Ward": ("#EB6834", "white"), "Discharge": ("#667085", "white"),
    "Return (unit not recorded)": ("#98A2B3", "#1D2939"), "Mixed": ("#D0D5DD", "#1D2939"),
}


def activity_role(name: str) -> str:
    return "Discharge" if name.startswith("Release") else ROLE_OF_ACTIVITY.get(name, "Mixed")


def department_table(log: pd.DataFrame) -> pd.DataFrame:
    """Per department: inferred role, events, patients and the step it records most."""
    events = log.assign(role=log[ACTIVITY].map(activity_role))
    rows = []
    for group, part in events.groupby(GROUP):
        shares = part["role"].value_counts(normalize=True)
        role = shares.index[0] if shares.iloc[0] >= ROLE_MIN_SHARE else "Mixed"
        rows.append({
            "department": group, "role": role, "label": f"{group}: {role}", "events": len(part),
            "patients": part[CASE_ID].nunique(), "top_step": part[ACTIVITY].value_counts().index[0],
        })
    table = pd.DataFrame(rows).sort_values("events", ascending=False).reset_index(drop=True)
    table["share_of_events_pct"] = table["events"] / table["events"].sum() * 100
    return table


def handover_events(log: pd.DataFrame) -> pd.DataFrame:
    """Every handover: consecutive events of one patient recorded by different departments."""
    ordered = log.sort_values([CASE_ID, TIMESTAMP], kind="stable")
    by_case = ordered.groupby(CASE_ID, sort=False)
    ordered = ordered.assign(
        to_department=by_case[GROUP].shift(-1), to_activity=by_case[ACTIVITY].shift(-1),
        wait_hours=(by_case[TIMESTAMP].shift(-1) - ordered[TIMESTAMP]).dt.total_seconds() / 3600,
    )
    changed = ordered[ordered["to_department"].notna() & (ordered[GROUP] != ordered["to_department"])]
    return changed.rename(columns={GROUP: "from_department", ACTIVITY: "from_activity"})[
        [CASE_ID, "from_department", "to_department", "from_activity", "to_activity", "wait_hours"]
    ].reset_index(drop=True)


def handover_pairs(handovers: pd.DataFrame, roles: pd.Series | None = None) -> pd.DataFrame:
    """Handovers per pair of departments (or of roles when `roles` maps department to role)."""
    h = handovers.copy()
    if roles is not None:
        h["from_department"] = h["from_department"].map(roles)
        h["to_department"] = h["to_department"].map(roles)
    pairs = (
        h.groupby(["from_department", "to_department"])["wait_hours"]
        .agg(handovers="size", median_wait_hours="median", p90_wait_hours=lambda s: s.quantile(0.9))
        .reset_index()
        .sort_values("handovers", ascending=False)
        .reset_index(drop=True)
    )
    pairs["share_of_handovers_pct"] = pairs["handovers"] / pairs["handovers"].sum() * 100
    return pairs


def ward_transfers(log: pd.DataFrame) -> pd.DataFrame:
    """Repeated ward admission records: which department recorded the previous one and which this one."""
    ward = log[log[ACTIVITY] == WARD].sort_values([CASE_ID, TIMESTAMP], kind="stable")
    by_case = ward.groupby(CASE_ID)
    repeats = ward.assign(
        previous_department=by_case[GROUP].shift(1),
        gap_hours=by_case[TIMESTAMP].diff().dt.total_seconds() / 3600,
    )
    repeats = repeats[repeats["previous_department"].notna()].rename(columns={GROUP: "department"})
    repeats["changed_department"] = repeats["department"] != repeats["previous_department"]
    return repeats[[CASE_ID, "previous_department", "department", "gap_hours", "changed_department"]].reset_index(drop=True)


def first_ward_department(log: pd.DataFrame) -> pd.Series:
    """The department that recorded each patient's first ward admission."""
    ward = log[log[ACTIVITY] == WARD].sort_values([CASE_ID, TIMESTAMP], kind="stable")
    return ward.groupby(CASE_ID)[GROUP].first()


def network_dot(pairs: pd.DataFrame, nodes: pd.DataFrame, min_handovers: int = 1, rankdir: str = "LR") -> str:
    """Graphviz DOT text of the handover network.

    `nodes` needs columns node, label, role and events; `pairs` needs from_department, to_department and handovers
    (both holding node names). Edges with fewer than `min_handovers` handovers are left out.
    """
    edges = pairs[pairs["handovers"] >= min_handovers]
    used = set(edges["from_department"]) | set(edges["to_department"])
    shown = nodes[nodes["node"].isin(used)]
    top = max(edges["handovers"].max(), 1) if len(edges) else 1
    lines = [
        f'digraph G {{ rankdir={rankdir}; bgcolor="transparent"; nodesep=0.45; ranksep=0.9;',
        'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=13 margin="0.18,0.1" penwidth=0];',
        'edge [fontname="Helvetica" fontsize=11 color="#667085" fontcolor="#475467" arrowsize=0.8];',
    ]
    for _, node in shown.iterrows():
        fill, font = ROLE_COLORS.get(node["role"], ROLE_COLORS["Mixed"])
        label = f'{node["label"]}\\n{int(node["events"]):,} events'
        lines.append(f'"{node["node"]}" [label="{label}" fillcolor="{fill}" fontcolor="{font}"];')
    for _, edge in edges.iterrows():
        width = 0.8 + 5.5 * edge["handovers"] / top
        lines.append(f'"{edge["from_department"]}" -> "{edge["to_department"]}" [label="{int(edge["handovers"]):,}" penwidth={width:.2f}];')
    lines.append("}")
    return "\n".join(lines)


def role_nodes(departments: pd.DataFrame) -> pd.DataFrame:
    """One node per role, for the simple network."""
    nodes = departments.groupby("role").agg(events=("events", "sum"), departments=("department", "nunique")).reset_index()
    nodes["node"] = nodes["role"]
    nodes["label"] = [f"{r} ({d} departments)" if d > 1 else r for r, d in zip(nodes["role"], nodes["departments"])]
    return nodes


def department_nodes(departments: pd.DataFrame, top: int = 12) -> pd.DataFrame:
    """One node per department for the detailed network (the busiest `top`)."""
    nodes = departments.head(top).copy()
    nodes["node"] = nodes["department"]
    return nodes


def save_network_png(dot: str, path: Path) -> None:
    import graphviz

    from discovery import ensure_graphviz

    ensure_graphviz()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(graphviz.Source(dot).pipe(format="png"))


def summary(log: pd.DataFrame) -> dict:
    departments = department_table(log)
    handovers = handover_events(log)
    transfers = ward_transfers(log)
    transitions = int(log.groupby(CASE_ID).size().sub(1).sum())
    return {
        "departments": len(departments),
        "events": len(log),
        "handovers": len(handovers),
        "handover_share_of_transitions_pct": len(handovers) / transitions * 100,
        "busiest_department": departments.iloc[0]["label"],
        "busiest_share_pct": departments.iloc[0]["share_of_events_pct"],
        "repeated_ward_records": len(transfers),
        "ward_records_changing_department": int(transfers["changed_department"].sum()),
        "ward_records_changing_department_pct": transfers["changed_department"].mean() * 100 if len(transfers) else np.nan,
    }


def main():
    log = load_log()
    departments = department_table(log)
    handovers = handover_events(log)
    pairs = handover_pairs(handovers)
    transfers = ward_transfers(log)
    top_transfers = (
        transfers[transfers["changed_department"]].groupby(["previous_department", "department"])
        .agg(transfers=("gap_hours", "size"), median_gap_hours=("gap_hours", "median")).reset_index().sort_values("transfers", ascending=False)
    )

    WORKLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    departments.round(2).to_csv(WORKLOAD_PATH, index=False)
    pairs.round(2).to_csv(PAIRS_PATH, index=False)
    top_transfers.round(2).to_csv(WARD_TRANSFERS_PATH, index=False)

    s = summary(log)
    print(f"Departments            : {s['departments']}")
    print(f"Busiest department     : {s['busiest_department']} ({s['busiest_share_pct']:.0f}% of events)")
    print(f"Handovers              : {s['handovers']:,} ({s['handover_share_of_transitions_pct']:.0f}% of step-to-step transitions)")
    print(f"Repeated ward records  : {s['repeated_ward_records']}; {s['ward_records_changing_department']} "
          f"({s['ward_records_changing_department_pct']:.0f}%) are recorded by a different department than the previous one")
    print("Roles:", departments.groupby("role")["department"].apply(lambda s: ", ".join(s)).to_dict())
    print(f"Saved: {WORKLOAD_PATH}\n       {PAIRS_PATH}\n       {WARD_TRANSFERS_PATH}")
    return departments


if __name__ == "__main__":
    main()
