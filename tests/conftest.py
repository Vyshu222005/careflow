"""Shared test fixtures: a small synthetic event log and helpers.

The synthetic log mimics the structure of the real cleaned log (one ER Registration first in every case, flags on the
registration event, lab values on lab events), so the feature, model and report code can be tested anywhere,
including in CI where the real dataset is not available.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
C, A, T = "case:concept:name", "concept:name", "time:timestamp"
FLAGS = ["InfectionSuspected", "SIRSCritTachypnea", "SIRSCritHeartRate", "SIRSCritTemperature", "SIRSCritLeucos", "SIRSCriteria2OrMore",
         "DisfuncOrg", "Hypotensie", "Hypoxie", "Oligurie"]


def make_log(n: int = 400, seed: int = 0, signal: bool = True, untreated_share: float = 0.1) -> pd.DataFrame:
    """Synthetic cleaned log. With signal=True, older patients are treated later (a planted, learnable effect)."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2014-01-01", tz="UTC")
    rows = []
    for i in range(n):
        case = f"P{i:04d}"
        reg = start + pd.Timedelta(minutes=int(i * 97 + rng.integers(0, 60)))
        age = int(rng.integers(20, 90))
        flags = {f: bool(rng.random() < 0.3) for f in FLAGS}
        triage = reg + pd.Timedelta(minutes=float(rng.uniform(2, 20)))
        sepsis = triage + pd.Timedelta(minutes=float(rng.uniform(1, 30)))
        slope = (age - 20) / 70 if signal else 0.5
        delay_minutes = float(rng.normal(30 + 70 * slope, 12))
        abx = sepsis + pd.Timedelta(minutes=max(delay_minutes - 15, 3))
        events = [("ER Registration", reg, None), ("ER Triage", triage, None), ("ER Sepsis Triage", sepsis, None)]
        events.append(("LacticAcid", reg + pd.Timedelta(minutes=1), float(rng.uniform(0.5, 4))))
        if rng.random() >= untreated_share:
            events.append(("IV Antibiotics", abx, None))
        events.append(("Admission NC", abx + pd.Timedelta(hours=2), None))
        events.append(("Release A", abx + pd.Timedelta(days=3), None))
        for name, ts, lab in events:
            row = {C: case, A: name, T: ts, "org:group": "L" if (name == "ER Registration" and i % 15 == 0) else "A",
                   "Age": age, "Diagnose": "A", "LacticAcid": np.nan, "CRP": np.nan, "Leucocytes": np.nan}
            row.update({f: v for f, v in flags.items()})
            if lab is not None:
                row[name] = lab
            rows.append(row)
    log = pd.DataFrame(rows)
    return log.sort_values([C, T], kind="stable").reset_index(drop=True)


@pytest.fixture(scope="session")
def synthetic_log() -> pd.DataFrame:
    return make_log()


@pytest.fixture(scope="session")
def noise_log() -> pd.DataFrame:
    """A log where nothing predicts delay."""
    return make_log(seed=1, signal=False)


@pytest.fixture(scope="session")
def real_paths():
    paths = [ROOT / "data" / "processed" / name for name in ("sepsis_clean.csv", "sepsis_events.csv", "sepsis_cases.csv")]
    if not all(p.exists() for p in paths):
        pytest.skip("processed data not built: run python src/clean_data.py and python src/prepare_log.py")
    return paths
