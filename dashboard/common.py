"""Shared constants and helpers for every dashboard tab."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from load_data import ACTIVITY, CASE_ID, TIMESTAMP

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#1d2939"
INK_SOFT = "#475467"
GRID = "rgba(102,112,133,0.18)"
AGE_ORDER = ["under 45", "45-64", "65-79", "80+"]
RETURN_OPTIONS = ["All", "Returned", "Did not return"]


def style(fig: go.Figure, height: int = 340, top: int = 8) -> go.Figure:
    """Shared chart look: transparent background, light grid, soft text."""
    fig.update_layout(
        height=height, margin=dict(l=0, r=12, t=top, b=0), showlegend=False, bargap=0.35,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, Roboto, sans-serif", size=12, color=INK_SOFT),
        hoverlabel=dict(bgcolor="white", font_size=12, font_color=INK, bordercolor="#d0d5dd"),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, automargin=True)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def median_return_days(events: pd.DataFrame):
    """Median days from the first discharge to a return to the ER."""
    release = events[events[ACTIVITY].str.startswith("Release")].groupby(CASE_ID)[TIMESTAMP].min()
    back = events[events[ACTIVITY] == "Return ER"].groupby(CASE_ID)[TIMESTAMP].min()
    gap = (back - release.reindex(back.index)).dropna().dt.total_seconds() / 86400
    return gap.median() if len(gap) else None
