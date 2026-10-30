"""Smart insights: transparent, rule-based findings and operational suggestions.

Every insight is produced by a named rule with a visible threshold. It carries:
    finding      what the data shows, with the numbers
    rule         the exact condition that fired
    explanation  why this is worth attention
    action       a suggestion about what to REVIEW in the process (never a clinical instruction)
    evidence     the numbers behind it

Nothing here is a clinical guideline. The thresholds are working defaults chosen for this project, kept in one place
(`THRESHOLDS`) so they are easy to see and change.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import bundle as bundle_module
import conformance as conformance_module
import handover as handover_module
import rework as rework_module
import simulator
from bottlenecks import clinical_steps, er_waits, format_hours, transition_stats
from load_data import ACTIVITY, CASE_ID

THRESHOLDS = {
    "antibiotics_within_1h_low_pct": 50,  # below this: flagged; below 35: high priority
    "antibiotics_within_1h_very_low_pct": 35,
    "no_antibiotics_pct": 15,
    "missing_discharge_pct": 15,
    "missing_discharge_high_pct": 25,
    "conformance_low_pct": 50,
    "ward_repeat_pct": 20,
    "ward_transfer_share_pct": 70,
    "return_30d_pct": 10,
    "bundle_low_pct": 30,
    "repeat_lactate_low_pct": 50,
    "lab_to_discharge_hours": 24,
    "er_wait_hours": 2.0,
    "min_patients": 30,
    "min_step_patients": 30,
}
SEVERITY_ORDER = {"High": 0, "Medium": 1, "Low": 2, "Info": 3}


@dataclass
class Insight:
    id: str
    severity: str
    title: str
    finding: str
    rule: str
    explanation: str
    action: str
    evidence: dict = field(default_factory=dict)


def pct(part, whole) -> float:
    return float(part) / float(whole) * 100 if whole else float("nan")


def collect_metrics(cases: pd.DataFrame, hours: pd.DataFrame, events: pd.DataFrame, clean_log: pd.DataFrame, conformance_df: pd.DataFrame,
                    rework_df: pd.DataFrame, bundle_df: pd.DataFrame) -> dict:
    """Numbers used by the rules, for the selected patients. All frames must already be filtered to the same patients."""
    m = {"n_patients": len(cases)}
    antibiotic_hours = hours["IV Antibiotics"].dropna() if "IV Antibiotics" in hours else pd.Series(dtype=float)
    m["abx_treated"] = len(antibiotic_hours)
    m["abx_untreated"] = len(cases) - len(antibiotic_hours)
    m["abx_untreated_pct"] = pct(m["abx_untreated"], len(cases))
    m["abx_median_hours"] = float(antibiotic_hours.median()) if len(antibiotic_hours) else float("nan")
    m["abx_within_1h"] = int((antibiotic_hours <= 1).sum())
    m["abx_within_1h_pct"] = pct(m["abx_within_1h"], len(antibiotic_hours))

    # which segment of the delay is longer among late patients
    if len(antibiotic_hours) >= 10 and "ER Sepsis Triage" in hours:
        parts = simulator.segments(hours)
        late = parts[parts["total"] > 1]
        m["late_patients"] = len(late)
        m["late_segment1_median_h"] = float(late["segment1"].median()) if len(late) else float("nan")
        m["late_segment2_median_h"] = float(late["segment2"].median()) if len(late) else float("nan")
    m["missing_discharge_pct"] = pct((cases["outcome"] == "No release recorded").sum(), len(cases))
    m["conformance_pct"] = float(conformance_df["conforms"].mean() * 100) if len(conformance_df) else float("nan")
    m["avg_fitness"] = float(conformance_df["fitness"].mean()) if len(conformance_df) else float("nan")
    patterns = conformance_module.deviation_patterns(conformance_df, top=1) if len(conformance_df) else pd.DataFrame()
    m["top_deviation"] = (patterns.iloc[0]["description"], int(patterns.iloc[0]["cases"])) if len(patterns) else None

    ward = rework_df["repeated_steps"].str.contains(rework_module.WARD)
    m["repeated_ward_patients"] = int(ward.sum())
    m["repeated_ward_pct"] = pct(ward.sum(), len(rework_df))
    transfers = handover_module.ward_transfers(clean_log)
    m["repeated_ward_records"] = len(transfers)
    m["ward_transfer_share_pct"] = float(transfers["changed_department"].mean() * 100) if len(transfers) else float("nan")

    returns = rework_module.return_intervals(clean_log)
    released = clean_log[clean_log[ACTIVITY].str.startswith("Release")][CASE_ID].nunique()
    m["released"] = released
    m["returned"] = len(returns)
    m["return_30d"] = int((returns["days_to_return"] <= 30).sum()) if len(returns) else 0
    m["return_30d_pct"] = pct(m["return_30d"], released)

    card = bundle_module.scorecard(bundle_df).set_index("element") if len(bundle_df) else pd.DataFrame()
    for element in ("complete_bundle", "lactate_1h", "antibiotics_1h", "repeat_lactate", "fluid_3h"):
        m[f"bundle_{element}_pct"] = float(card.loc[element, "met_pct"]) if element in card.index else float("nan")
        m[f"bundle_{element}_n"] = int(card.loc[element, "applicable"]) if element in card.index else 0
    if len(bundle_df):
        chart = bundle_module.control_chart(bundle_df, "antibiotics_1h")
        m["control_signals"] = [(r["month"], r["signal"], float(r["pct"])) for _, r in chart[chart["signal"] != ""].iterrows()] if len(chart) else []
        m["control_months"] = len(chart)
    else:
        m["control_signals"], m["control_months"] = [], 0

    steps = clinical_steps(events)
    transitions = transition_stats(steps)
    waits = er_waits(transitions, THRESHOLDS["min_step_patients"])
    m["top_wait"] = (waits.iloc[0]["from_step"], waits.iloc[0]["to_step"], float(waits.iloc[0]["median_hours"]), int(waits.iloc[0]["cases"])) if len(waits) else None

    def median_for(a, b):
        row = transitions[(transitions["from_step"] == a) & (transitions["to_step"] == b)]
        return (float(row.iloc[0]["median_hours"]), int(row.iloc[0]["cases"])) if len(row) else None

    m["direct_to_antibiotics"] = median_for("ER Sepsis Triage", "IV Antibiotics")
    m["triage_to_fluid"] = median_for("ER Sepsis Triage", "IV Liquid")

    if len(clean_log):
        departments = handover_module.department_table(clean_log)
        roles = departments.set_index("department")["role"]
        pairs = handover_module.handover_pairs(handover_module.handover_events(clean_log), roles)
        row = pairs[(pairs["from_department"] == "Laboratory") & (pairs["to_department"] == "Discharge")]
        m["lab_to_discharge"] = (float(row.iloc[0]["median_wait_hours"]), int(row.iloc[0]["handovers"])) if len(row) else None
    else:
        m["lab_to_discharge"] = None
    return m


def _insight_antibiotics(m: dict) -> Insight | None:
    t = THRESHOLDS
    if m["abx_treated"] < t["min_patients"] or np.isnan(m["abx_within_1h_pct"]) or m["abx_within_1h_pct"] >= t["antibiotics_within_1h_low_pct"]:
        return None
    severity = "High" if m["abx_within_1h_pct"] < t["antibiotics_within_1h_very_low_pct"] else "Medium"
    detail = ""
    if "late_segment1_median_h" in m:
        longer = "from sepsis triage to antibiotics" if m["late_segment2_median_h"] >= m["late_segment1_median_h"] else "from registration to sepsis triage"
        detail = (f" Among patients treated after more than 1 hour, the longer part of the wait is {longer} "
                  f"(median {format_hours(max(m['late_segment1_median_h'], m['late_segment2_median_h']))} against {format_hours(min(m['late_segment1_median_h'], m['late_segment2_median_h']))}).")
    return Insight(
        "slow_antibiotics", severity, "Antibiotics are often given later than 1 hour after registration",
        f"Only {m['abx_within_1h_pct']:.0f}% of treated patients ({m['abx_within_1h']:,} of {m['abx_treated']:,}) receive antibiotics within 1 hour; "
        f"the median is {format_hours(m['abx_median_hours'])}.{detail}",
        f"Share treated within 1 hour is below {t['antibiotics_within_1h_low_pct']}% (high priority below {t['antibiotics_within_1h_very_low_pct']}%).",
        "Early antibiotics are a widely used sepsis quality target. The clock here starts at registration, so the true share is probably higher, but a large gap is worth understanding.",
        "Review where the time goes between registration and the antibiotic order, starting with the longer segment shown in the Bottleneck Analysis and What-if Simulator tabs.",
        {"within_1h_pct": m["abx_within_1h_pct"], "median_hours": m["abx_median_hours"], "treated": m["abx_treated"]},
    )


def _insight_untreated(m: dict) -> Insight | None:
    if m["n_patients"] < THRESHOLDS["min_patients"] or m["abx_untreated_pct"] <= THRESHOLDS["no_antibiotics_pct"]:
        return None
    return Insight(
        "no_antibiotics", "Medium", "Many patients have no antibiotics recorded",
        f"{m['abx_untreated']:,} patients ({m['abx_untreated_pct']:.0f}%) have no antibiotic event in the log.",
        f"Patients with no antibiotic record exceed {THRESHOLDS['no_antibiotics_pct']}%.",
        "Either treatment was not given or it was not recorded. The data cannot tell which, but both matter for monitoring.",
        "Check with the data owner whether these records are incomplete, and whether these patients differ in how they entered the ER.",
        {"untreated": m["abx_untreated"], "untreated_pct": m["abx_untreated_pct"]},
    )


def _insight_discharge(m: dict) -> Insight | None:
    t = THRESHOLDS
    if m["n_patients"] < t["min_patients"] or m["missing_discharge_pct"] <= t["missing_discharge_pct"]:
        return None
    severity = "High" if m["missing_discharge_pct"] > t["missing_discharge_high_pct"] else "Medium"
    return Insight(
        "missing_discharge", severity, "A large share of pathways has no discharge event",
        f"{m['missing_discharge_pct']:.0f}% of the selected patients have no discharge recorded, so their pathway ends early in the data.",
        f"Patients without a discharge event exceed {t['missing_discharge_pct']}% (high priority above {t['missing_discharge_high_pct']}%).",
        "Without an ending, durations and outcomes for these patients cannot be measured, which weakens every downstream analysis.",
        "Find out why these pathways stop (transfer, data capture, or a different record system) and whether recording of the final step can be checked.",
        {"missing_discharge_pct": m["missing_discharge_pct"]},
    )


def _insight_conformance(m: dict) -> Insight | None:
    if m["n_patients"] < THRESHOLDS["min_patients"] or np.isnan(m["conformance_pct"]) or m["conformance_pct"] >= THRESHOLDS["conformance_low_pct"]:
        return None
    top = f" The most common deviation is: {m['top_deviation'][0]} ({m['top_deviation'][1]} patients)." if m["top_deviation"] else ""
    return Insight(
        "low_conformance", "Medium", "Few patients follow the expected pathway exactly",
        f"{m['conformance_pct']:.0f}% of patients follow the assumed expected pathway exactly (average fitness {m['avg_fitness']:.2f}).{top}",
        f"Share following the expected pathway is below {THRESHOLDS['conformance_low_pct']}%.",
        "The expected pathway is an assumption, so a low figure may reflect a strict definition as much as real variation.",
        "Confirm the expected pathway with clinical staff, then look at the most common deviation patterns in Quality & Conformance.",
        {"conformance_pct": m["conformance_pct"], "average_fitness": m["avg_fitness"]},
    )


def _insight_ward(m: dict) -> Insight | None:
    t = THRESHOLDS
    if m["n_patients"] < t["min_patients"] or m["repeated_ward_pct"] <= t["ward_repeat_pct"]:
        return None
    transfers = m["ward_transfer_share_pct"]
    mostly_transfers = not np.isnan(transfers) and transfers >= t["ward_transfer_share_pct"]
    return Insight(
        "ward_records", "Low", "Ward admission is often recorded more than once",
        f"{m['repeated_ward_patients']:,} patients ({m['repeated_ward_pct']:.0f}%) have more than one ward admission record" +
        (f", and {transfers:.0f}% of the repeated records are entered by a different unit than the previous one, which looks like transfers." if mostly_transfers else "."),
        f"More than {t['ward_repeat_pct']}% of patients repeat the ward admission record.",
        "Repeated records that change unit usually reflect patients moving between ward units, which inflates event counts and can look like rework.",
        "Consider recording transfers as their own event type, so a move between units is not mistaken for a repeated admission.",
        {"patients": m["repeated_ward_patients"], "transfer_share_pct": transfers},
    )


def _insight_returns(m: dict) -> Insight | None:
    if m["released"] < THRESHOLDS["min_patients"] or m["return_30d_pct"] <= THRESHOLDS["return_30d_pct"]:
        return None
    return Insight(
        "returns", "Medium", "A notable share of discharged patients return within 30 days",
        f"{m['return_30d']:,} of {m['released']:,} discharged patients ({m['return_30d_pct']:.0f}%) return to the ER within 30 days.",
        f"The 30-day return share exceeds {THRESHOLDS['return_30d_pct']}%.",
        "Early returns can signal an incomplete recovery or a gap after discharge, although the data does not show the reason for a return.",
        "Review a sample of early returns to see what brought the patients back and whether anything in discharge could be checked.",
        {"returns_30d": m["return_30d"], "return_30d_pct": m["return_30d_pct"]},
    )


def _insight_bundle(m: dict) -> list:
    t = THRESHOLDS
    out = []
    if m["n_patients"] >= t["min_patients"] and not np.isnan(m["bundle_complete_bundle_pct"]) and m["bundle_complete_bundle_pct"] < t["bundle_low_pct"]:
        elements = {"lactate_1h": "lactate within 1 hour", "antibiotics_1h": "antibiotics within 1 hour"}
        weakest = min(elements, key=lambda e: m[f"bundle_{e}_pct"])
        out.append(Insight(
            "low_bundle", "High", "Few patients receive every early care element on time",
            f"{m['bundle_complete_bundle_pct']:.0f}% of patients meet every applicable element of the adapted care bundle; the weakest element is "
            f"{elements[weakest]} ({m[f'bundle_{weakest}_pct']:.0f}%).",
            f"Complete-bundle share is below {t['bundle_low_pct']}%.",
            "The scorecard is adapted from the hour-1 sepsis bundle using only what the log records, so it is an indicator, not an official measure.",
            f"Start with the weakest element ({elements[weakest]}) and check how it is recorded before drawing conclusions.",
            {"complete_bundle_pct": m["bundle_complete_bundle_pct"], "weakest_element": weakest},
        ))
    if m["bundle_repeat_lactate_n"] >= t["min_patients"] and m["bundle_repeat_lactate_pct"] < t["repeat_lactate_low_pct"]:
        out.append(Insight(
            "repeat_lactate", "Medium", "Raised lactate is often not measured again",
            f"Of {m['bundle_repeat_lactate_n']:,} patients whose first lactate was above 2, only {m['bundle_repeat_lactate_pct']:.0f}% have a repeat measurement.",
            f"Repeat-lactate share is below {t['repeat_lactate_low_pct']}% among patients with a first lactate above 2.",
            "Repeat measurement after a high value is a common part of sepsis bundles. It may also be recorded elsewhere than in this log.",
            "Check whether repeat lactate results are captured in this event log before treating this as missed care.",
            {"repeat_lactate_pct": m["bundle_repeat_lactate_pct"]},
        ))
    return out


def _insight_control(m: dict) -> Insight | None:
    if m["control_months"] < 6:
        return None
    if m["control_signals"]:
        months = ", ".join(f"{month} ({pct_value:.0f}%)" for month, _, pct_value in m["control_signals"])
        return Insight(
            "control_signal", "Low", "The monthly antibiotic rate changed beyond chance in some months",
            f"On the control chart for antibiotics within 1 hour, these months fall outside the limits or in a long run: {months}.",
            "A month is outside 3 standard errors of the overall rate, or 8 months in a row are on one side of the average.",
            "Control-chart signals separate real changes from normal month-to-month variation.",
            "Look at what was different in those months (staffing, volume, recording practice) in Quality & Conformance.",
            {"signals": m["control_signals"]},
        )
    return Insight(
        "control_stable", "Info", "Monthly antibiotic timing is stable, not improving",
        f"No month differs from the overall antibiotic rate by more than chance across {m['control_months']} months.",
        "No control-chart signals in the monthly antibiotics-within-1-hour rate.",
        "Stable performance at a low level means month-to-month differences are noise, and improvement needs a change in the process rather than waiting for the rate to move.",
        "Use the What-if Simulator to explore which segment of the wait is worth working on.",
        {"months": m["control_months"]},
    )


def _insight_lab_discharge(m: dict) -> Insight | None:
    row = m.get("lab_to_discharge")
    if not row or row[1] < THRESHOLDS["min_step_patients"] or row[0] <= THRESHOLDS["lab_to_discharge_hours"]:
        return None
    return Insight(
        "lab_to_discharge", "Medium", "Long gap between the last laboratory record and discharge",
        f"For {row[1]:,} patients the median time from a laboratory record to the discharge record is {format_hours(row[0])}.",
        f"Median wait from Laboratory to Discharge exceeds {THRESHOLDS['lab_to_discharge_hours']} hours.",
        "The data cannot show whether this gap is clinical (waiting for results or recovery) or administrative.",
        "Review whether discharge timing could be checked against the last laboratory result for a sample of patients.",
        {"median_hours": row[0], "patients": row[1]},
    )


def _insight_er_wait(m: dict) -> Insight | None:
    top = m.get("top_wait")
    if not top or top[2] <= THRESHOLDS["er_wait_hours"]:
        return None
    return Insight(
        "slow_er_step", "Medium", "One ER step has a long typical wait",
        f"The longest typical wait between two ER steps is {top[0]} to {top[1]}: median {format_hours(top[2])} ({top[3]:,} patients).",
        f"The median wait of the slowest frequent ER step exceeds {THRESHOLDS['er_wait_hours']:.0f} hours.",
        "A long typical wait at one step marks a candidate bottleneck.",
        "Look at what happens between these two steps (who is responsible, what is awaited) in Bottleneck Analysis.",
        {"from": top[0], "to": top[1], "median_hours": top[2]},
    )


def _insight_fluid(m: dict) -> Insight | None:
    direct, fluid = m.get("direct_to_antibiotics"), m.get("triage_to_fluid")
    if not direct or not fluid or direct[1] < THRESHOLDS["min_step_patients"] or fluid[1] < THRESHOLDS["min_step_patients"]:
        return None
    if direct[0] <= 2 * fluid[0] or direct[0] - fluid[0] < 0.5:
        return None
    return Insight(
        "direct_antibiotics", "Low", "Patients who go straight to antibiotics wait longer than those who get IV fluid first",
        f"After sepsis triage the median wait is {format_hours(direct[0])} to antibiotics ({direct[1]:,} patients) against {format_hours(fluid[0])} to IV fluid ({fluid[1]:,} patients).",
        "Median wait from sepsis triage to antibiotics is more than twice, and at least 30 minutes longer than, the wait to IV fluid.",
        "The two groups may differ in severity or in how the steps are recorded, so this is a pattern to understand, not a cause.",
        "Check whether the two routes reflect different patient groups or different ways of recording the same care.",
        {"direct_hours": direct[0], "fluid_hours": fluid[0]},
    )


def generate(metrics: dict) -> list:
    """Apply every rule. Returns insights sorted by severity (most important first)."""
    found = [_insight_antibiotics(metrics), _insight_untreated(metrics), _insight_discharge(metrics), _insight_conformance(metrics),
             _insight_ward(metrics), _insight_returns(metrics), *_insight_bundle(metrics), _insight_control(metrics),
             _insight_lab_discharge(metrics), _insight_er_wait(metrics), _insight_fluid(metrics)]
    insights = [i for i in found if i is not None]
    if metrics["n_patients"] < THRESHOLDS["min_patients"]:
        insights.append(Insight(
            "small_sample", "Info", "Very few patients are selected",
            f"Only {metrics['n_patients']} patients match the filters, so most rules are switched off.",
            f"Fewer than {THRESHOLDS['min_patients']} selected patients.",
            "Percentages from very small groups can move a lot with one patient.", "Widen the filters to see more reliable insights.",
            {"patients": metrics["n_patients"]},
        ))
    return sorted(insights, key=lambda i: SEVERITY_ORDER[i.severity])


def collect_from_context(ctx) -> dict:
    """Gather the metrics from a dashboard Context (duck-typed, so tests can pass any object with the same fields)."""
    ids = set(ctx.cases[CASE_ID])
    conf = ctx.conformance_all[ctx.conformance_all[CASE_ID].isin(ids)]
    rw = ctx.rework_all[ctx.rework_all[CASE_ID].isin(ids)]
    bt = ctx.bundle_all[ctx.bundle_all[CASE_ID].isin(ids)]
    log = ctx.clean_log[ctx.clean_log[CASE_ID].isin(ids)]
    return collect_metrics(ctx.cases, ctx.hours, ctx.events, log, conf, rw, bt)
