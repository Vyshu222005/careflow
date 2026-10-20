"""Load the Sepsis event log into a pandas DataFrame.

Usage:
    python src/load_data.py
"""
from pathlib import Path

import pandas as pd
import pm4py

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "sepsis.xes"

CASE_ID = "case:concept:name"
ACTIVITY = "concept:name"
TIMESTAMP = "time:timestamp"
RESOURCE = "org:group"


def load_event_log(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read an XES file and return the events as a DataFrame (one row per event)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/download_data.py` first."
        )
    df = pm4py.read_xes(str(path), return_legacy_log_object=False)
    return pd.DataFrame(df)


def summarize(df: pd.DataFrame) -> dict:
    """Headline numbers for an event log."""
    return {
        "events": len(df),
        "cases": df[CASE_ID].nunique(),
        "activities": df[ACTIVITY].nunique(),
        "columns": df.shape[1],
        "start": df[TIMESTAMP].min(),
        "end": df[TIMESTAMP].max(),
    }


if __name__ == "__main__":
    log = load_event_log()
    print(f"Shape: {log.shape}")
    for key, value in summarize(log).items():
        print(f"{key:>10}: {value}")
    print("\nFirst rows:")
    print(log[[CASE_ID, ACTIVITY, TIMESTAMP, RESOURCE]].head(10).to_string(index=False))
