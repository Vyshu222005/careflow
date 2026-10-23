"""Variant analysis: which pathways do patients follow, and how common is each?

A variant is the exact sequence of activities of a case. Two levels are used:

    exact      every event, including repeated lab tests
    milestone  lab tests removed and consecutive repeats collapsed, so only the
               clinical steps remain (registration, triage, IV treatment,
               admission, release, return)

Usage:
    python src/variants.py

Output:
    reports/top_variants.csv
"""
from pathlib import Path

import pandas as pd

from load_data import ACTIVITY, CASE_ID
from prepare_log import load_cases, load_events

ROOT = Path(__file__).resolve().parents[1]
TOP_VARIANTS_PATH = ROOT / "reports" / "top_variants.csv"

LAB_ACTIVITIES = {"Leucocytes", "CRP", "LacticAcid"}
SEPARATOR = " > "

# Short names for chart labels.
SHORT_NAMES = {
    "ER Registration": "Registration",
    "ER Triage": "Triage",
    "ER Sepsis Triage": "Sepsis triage",
    "IV Liquid": "IV fluid",
    "IV Antibiotics": "Antibiotics",
    "Admission NC": "Ward",
    "Admission IC": "ICU",
    "Return ER": "Return",
    "Leucocytes": "Leuco",
    "LacticAcid": "Lactic",
}


def milestone_sequence(activities: pd.Series) -> str:
    """Drop lab tests and collapse consecutive repeats of the same step."""
    steps = []
    for name in activities:
        if name in LAB_ACTIVITIES or (steps and steps[-1] == name):
            continue
        steps.append(name)
    return SEPARATOR.join(steps)


def add_milestone_variant(events: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    milestone = events.groupby(CASE_ID, sort=False)[ACTIVITY].agg(milestone_sequence)
    return cases.assign(milestone_variant=cases[CASE_ID].map(milestone))


def shorten(variant: str, drop_prefix: int = 0) -> str:
    """Readable label for a chart: short activity names joined with arrows."""
    steps = variant.split(SEPARATOR)[drop_prefix:]
    return " → ".join(SHORT_NAMES.get(s, s) for s in steps)


def variant_table(cases: pd.DataFrame, column: str) -> pd.DataFrame:
    """One row per variant, most common first, with share, cumulative share and median duration."""
    table = (
        cases.groupby(column)
        .agg(cases=(CASE_ID, "size"), median_duration_days=("duration_days", "median"))
        .reset_index()
        .rename(columns={column: "variant"})
        .sort_values(["cases", "variant"], ascending=[False, True])
        .reset_index(drop=True)
    )
    table.insert(0, "rank", table.index + 1)
    table["steps"] = table["variant"].str.count(SEPARATOR) + 1
    table["share_pct"] = table["cases"] / len(cases) * 100
    table["cumulative_pct"] = table["share_pct"].cumsum()
    return table


def coverage(table: pd.DataFrame, top_n=(1, 3, 5, 10, 20, 50)) -> pd.Series:
    """Share of all cases covered by the `n` most common variants."""
    return pd.Series({n: table["cumulative_pct"].iloc[min(n, len(table)) - 1] for n in top_n})


def summary(table: pd.DataFrame) -> dict:
    once = table["cases"] == 1
    return {
        "variants": len(table),
        "seen_once": int(once.sum()),
        "cases_in_unique_variants_pct": table.loc[once, "share_pct"].sum(),
        "top10_coverage_pct": table["cumulative_pct"].iloc[min(10, len(table)) - 1],
    }


def main():
    events, cases = load_events(), load_cases()
    cases = add_milestone_variant(events, cases)
    exact = variant_table(cases, "variant")
    milestone = variant_table(cases, "milestone_variant")

    top = pd.concat(
        [exact.head(20).assign(level="exact"), milestone.head(20).assign(level="milestone")],
        ignore_index=True,
    )
    top = top[["level", "rank", "variant", "cases", "share_pct", "cumulative_pct", "median_duration_days", "steps"]]
    TOP_VARIANTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    top.round(2).to_csv(TOP_VARIANTS_PATH, index=False)

    for name, table in (("Exact", exact), ("Milestone", milestone)):
        s = summary(table)
        print(
            f"{name:>9}: {s['variants']} variants, {s['seen_once']} seen once "
            f"({s['cases_in_unique_variants_pct']:.0f}% of cases), top 10 cover {s['top10_coverage_pct']:.1f}%"
        )
    print(f"Saved: {TOP_VARIANTS_PATH}")
    return exact, milestone


if __name__ == "__main__":
    main()
