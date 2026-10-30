"""What-if simulator for the time from ER registration to first antibiotics.

This is a transparent arithmetic scenario tool, NOT a forecast and NOT evidence of what would happen in a real
hospital. It answers one question: "if the recorded waits were shorter by these amounts, how would the delay
metrics in this dataset change?"

Rules (all visible to the user)
    1. For every treated patient the time to antibiotics is split into two segments:
         segment 1  registration -> sepsis triage
         segment 2  sepsis triage -> antibiotics
       Patients whose antibiotics were recorded before sepsis triage keep their whole time in segment 1 and have no
       segment 2.
    2. Each segment can be shortened by a percentage; segment 2 can also be shortened by a fixed number of minutes.
    3. A segment is never made longer, and segment 2 is never reduced below a floor (default 5 minutes) unless it
       was already shorter than the floor.
    4. The change applies to a share of patients ("adoption"). Which patients is fixed by a seeded shuffle, so the
       same settings always give the same answer.
    5. Everything else (lab tests, IV fluid, ward stay) is left exactly as recorded.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRIAGE = "ER Sepsis Triage"
ANTIBIOTICS = "IV Antibiotics"
TARGETS_HOURS = (1.0, 3.0)
SEED = 42
DEFAULT_FLOOR_MINUTES = 5.0


@dataclass(frozen=True)
class Scenario:
    cut_segment1_pct: float = 0.0  # registration -> sepsis triage
    cut_segment2_pct: float = 0.0  # sepsis triage -> antibiotics
    cut_segment2_minutes: float = 0.0
    adoption_pct: float = 100.0  # share of patients the change applies to
    floor_minutes: float = DEFAULT_FLOOR_MINUTES

    def validate(self) -> None:
        for name in ("cut_segment1_pct", "cut_segment2_pct", "adoption_pct"):
            if not 0 <= getattr(self, name) <= 100:
                raise ValueError(f"{name} must be between 0 and 100.")
        if self.cut_segment2_minutes < 0 or self.floor_minutes < 0:
            raise ValueError("Minutes must not be negative.")

    @property
    def is_baseline(self) -> bool:
        return self.cut_segment1_pct == 0 and self.cut_segment2_pct == 0 and self.cut_segment2_minutes == 0


@dataclass(frozen=True)
class Result:
    baseline_hours: pd.Series  # hours from registration to antibiotics, one value per treated patient
    simulated_hours: pd.Series
    scenario: Scenario

    def share_within(self, hours: float, simulated: bool = True) -> float:
        series = self.simulated_hours if simulated else self.baseline_hours
        return float((series <= hours).mean() * 100) if len(series) else float("nan")

    def summary(self) -> pd.DataFrame:
        """Baseline and simulated delay metrics side by side."""
        rows = []
        for label, fn in (
            ("Treated within 1 hour (%)", lambda s: (s <= 1).mean() * 100),
            ("Treated within 3 hours (%)", lambda s: (s <= 3).mean() * 100),
            ("Median hours to antibiotics", lambda s: s.median()),
            ("90th percentile hours", lambda s: s.quantile(0.9)),
            ("Mean hours to antibiotics", lambda s: s.mean()),
        ):
            base, sim = float(fn(self.baseline_hours)), float(fn(self.simulated_hours))
            rows.append({"metric": label, "baseline": base, "simulated": sim, "change": sim - base})
        return pd.DataFrame(rows)

    @property
    def newly_within_one_hour(self) -> int:
        return int(((self.simulated_hours <= 1) & (self.baseline_hours > 1)).sum())

    @property
    def patients_changed(self) -> int:
        return int((self.simulated_hours < self.baseline_hours - 1e-9).sum())


def segments(hours: pd.DataFrame) -> pd.DataFrame:
    """Split each treated patient's time to antibiotics into segment 1 and segment 2 (hours).

    `hours` has one row per patient with columns 'ER Sepsis Triage' and 'IV Antibiotics' (hours since registration).
    """
    for column in (TRIAGE, ANTIBIOTICS):
        if column not in hours.columns:
            raise ValueError(f"The table needs a '{column}' column.")
    treated = hours[hours[ANTIBIOTICS].notna()].copy()
    total = treated[ANTIBIOTICS]
    triage = treated[TRIAGE]
    after_triage = triage.notna() & (total >= triage)
    out = pd.DataFrame(index=treated.index)
    out["segment1"] = np.where(after_triage, triage, total)
    out["segment2"] = np.where(after_triage, total - triage, 0.0)
    out["total"] = total
    return out


def adopted_mask(index: pd.Index, adoption_pct: float) -> pd.Series:
    """Which patients the change applies to: a fixed pseudo-random subset of the requested size."""
    n = len(index)
    chosen = np.zeros(n, dtype=bool)
    count = int(round(n * adoption_pct / 100))
    chosen[np.random.default_rng(SEED).permutation(n)[:count]] = True
    return pd.Series(chosen, index=index)


def simulate(hours: pd.DataFrame, scenario: Scenario) -> Result:
    """Apply a scenario to the treated patients and return baseline and simulated times."""
    scenario.validate()
    parts = segments(hours)
    adopted = adopted_mask(parts.index, scenario.adoption_pct)

    new1 = parts["segment1"] * (1 - scenario.cut_segment1_pct / 100)
    reduced2 = parts["segment2"] * (1 - scenario.cut_segment2_pct / 100) - scenario.cut_segment2_minutes / 60
    floor = scenario.floor_minutes / 60
    new2 = np.maximum(reduced2, np.minimum(parts["segment2"], floor))  # never below the floor, never above the original
    new2 = np.minimum(new2, parts["segment2"])
    simulated = np.where(adopted, new1 + new2, parts["total"])
    return Result(parts["total"], pd.Series(simulated, index=parts.index), scenario)


def complete_bundle_share(bundle: pd.DataFrame, result: Result) -> tuple:
    """(baseline %, simulated %) of ALL patients meeting every applicable bundle element if only antibiotic timing changes."""
    sim_abx = bundle["case:concept:name"].map(result.simulated_hours.rename_axis("case:concept:name")).le(1)
    others = bundle["lactate_1h"] & (~bundle["hypotension"] | bundle["fluid_3h"]) & (~bundle["high_lactate"] | bundle["repeat_lactate"])
    return float(bundle["complete_bundle"].mean() * 100), float((others & sim_abx.fillna(False)).mean() * 100)


def sensitivity(hours: pd.DataFrame, minutes: list, adoption_pct: float = 100.0, cut_segment1_pct: float = 0.0) -> pd.DataFrame:
    """Share treated within 1 hour as the fixed reduction of segment 2 grows."""
    rows = []
    for m in minutes:
        result = simulate(hours, Scenario(cut_segment1_pct=cut_segment1_pct, cut_segment2_minutes=m, adoption_pct=adoption_pct))
        rows.append({"minutes_saved": m, "within_1h_pct": result.share_within(1.0), "median_hours": float(result.simulated_hours.median())})
    return pd.DataFrame(rows)


PRESETS = {
    "Baseline (no change)": Scenario(),
    "Faster sepsis triage (-30% wait)": Scenario(cut_segment1_pct=30),
    "Faster antibiotic decision (-30 minutes)": Scenario(cut_segment2_minutes=30),
    "Both improvements": Scenario(cut_segment1_pct=30, cut_segment2_minutes=30),
}
