"""Process discovery: draw the patient pathways as process models.

Models built with pm4py:
    DFG          directly-follows graph, shows how often one step follows another
    Heuristics   heuristics net, keeps only strong, frequent relations
    Inductive    Petri net from the inductive miner (can filter noise)

Two logs are used. The milestone log leaves out the repeated lab tests
(Leucocytes, CRP, LacticAcid); with them every map turns into a tangle.

Needs the Graphviz program for drawing (`winget install Graphviz.Graphviz`).

Usage:
    python src/discovery.py

Outputs:
    reports/figures/process_*.png
    reports/model_evaluation.csv
"""
import os
import shutil
from pathlib import Path

import pandas as pd
import pm4py

from load_data import ACTIVITY
from prepare_log import load_events
from variants import LAB_ACTIVITIES

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "reports" / "figures"
EVALUATION_PATH = ROOT / "reports" / "model_evaluation.csv"

# Share of the most frequent paths kept in a DFG drawing.
DFG_PATH_SHARE = 0.5
DFG_PATH_SHARE_ALL_ACTIVITIES = 0.2
# Heuristics net: relations must be strong (0-1) and seen at least this many times.
HEURISTICS_DEPENDENCY = 0.8
HEURISTICS_MIN_OCCURRENCES = 20
# Inductive miner: share of infrequent behaviour to ignore.
INDUCTIVE_NOISE = 0.2


def ensure_graphviz() -> None:
    """Make sure the Graphviz `dot` program can be found (it is not always on PATH on Windows)."""
    if shutil.which("dot"):
        return
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
        if base and (Path(base) / "Graphviz" / "bin" / "dot.exe").exists():
            os.environ["PATH"] += os.pathsep + str(Path(base) / "Graphviz" / "bin")
            return
    raise RuntimeError("Graphviz not found. Install it with: winget install Graphviz.Graphviz")


def milestone_log(events: pd.DataFrame) -> pd.DataFrame:
    """Event log without the repeated lab tests."""
    return events[~events[ACTIVITY].isin(LAB_ACTIVITIES)].reset_index(drop=True)


def save_dfg(log: pd.DataFrame, path: Path, path_share: float, title: str) -> None:
    dfg, start, end = pm4py.discover_dfg(log)
    dfg, start, end = pm4py.filter_dfg_paths_percentage(dfg, start, end, percentage=path_share)
    pm4py.save_vis_dfg(dfg, start, end, str(path), graph_title=title)


def save_heuristics_net(log: pd.DataFrame, path: Path) -> None:
    net = pm4py.discover_heuristics_net(
        log, dependency_threshold=HEURISTICS_DEPENDENCY, min_dfg_occurrences=HEURISTICS_MIN_OCCURRENCES
    )
    pm4py.save_vis_heuristics_net(net, str(path))


def discover_inductive(log: pd.DataFrame, noise: float):
    return pm4py.discover_petri_net_inductive(log, noise_threshold=noise)


def discover_heuristics_petri(log: pd.DataFrame):
    return pm4py.discover_petri_net_heuristics(log, dependency_threshold=HEURISTICS_DEPENDENCY)


def evaluate(log: pd.DataFrame, model) -> dict:
    """Quality of a Petri net against the log (token-based replay)."""
    net, initial, final = model
    fit = pm4py.fitness_token_based_replay(log, net, initial, final)
    precision = pm4py.precision_token_based_replay(log, net, initial, final)
    fitness = fit["average_trace_fitness"]
    return {
        "fitness": fitness,
        "perfectly_fitting_cases_pct": fit["perc_fit_traces"],
        "precision": precision,
        "f_score": 2 * fitness * precision / (fitness + precision),
        "places": len(net.places),
        "transitions": len(net.transitions),
        "arcs": len(net.arcs),
    }


def evaluate_models(events: pd.DataFrame, milestone: pd.DataFrame) -> pd.DataFrame:
    runs = [
        ("Inductive miner, no noise filter", "milestone", milestone, lambda log: discover_inductive(log, 0.0)),
        (f"Inductive miner, noise {INDUCTIVE_NOISE:.0%}", "milestone", milestone, lambda log: discover_inductive(log, INDUCTIVE_NOISE)),
        (f"Heuristics miner, dependency {HEURISTICS_DEPENDENCY}", "milestone", milestone, discover_heuristics_petri),
        (f"Inductive miner, noise {INDUCTIVE_NOISE:.0%}", "all activities", events, lambda log: discover_inductive(log, INDUCTIVE_NOISE)),
    ]
    rows = []
    for name, log_name, log, discover in runs:
        rows.append({"model": name, "log": log_name, **evaluate(log, discover(log))})
    table = pd.DataFrame(rows)
    return table.round({"fitness": 3, "perfectly_fitting_cases_pct": 1, "precision": 3, "f_score": 3})


def main():
    ensure_graphviz()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    events = load_events()
    milestone = milestone_log(events)

    save_dfg(events, FIGURES_DIR / "process_dfg_all_activities.png", DFG_PATH_SHARE_ALL_ACTIVITIES,
             "Directly-follows graph with lab tests (most frequent paths)")
    save_dfg(milestone, FIGURES_DIR / "process_dfg_milestone.png", DFG_PATH_SHARE,
             "Directly-follows graph, clinical steps (most frequent paths)")
    save_heuristics_net(milestone, FIGURES_DIR / "process_heuristics_milestone.png")
    net, initial, final = discover_inductive(milestone, INDUCTIVE_NOISE)
    pm4py.save_vis_petri_net(net, initial, final, str(FIGURES_DIR / "process_petri_inductive.png"))

    evaluation = evaluate_models(events, milestone)
    evaluation.to_csv(EVALUATION_PATH, index=False)
    print(evaluation.to_string(index=False))
    print(f"Figures: {FIGURES_DIR}\nSaved  : {EVALUATION_PATH}")
    return evaluation


if __name__ == "__main__":
    main()
