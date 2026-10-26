"""Small HTML helpers for the dashboard (cards, header, callouts).

The look itself lives in assets/style.css. All HTML is built on one line, because
Streamlit's markdown treats blank lines and indented lines as the end of an HTML block.
"""
import html
from pathlib import Path

import streamlit as st

STYLE_PATH = Path(__file__).resolve().parents[1] / "assets" / "style.css"

# Simple line icons (24x24 grid, drawn with strokes).
ICONS = {
    "pulse": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "users": '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'
             '<path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "list": '<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/>'
            '<line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "heart": '<path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>',
    "drop": '<path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/>',
    "repeat": '<polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/>'
              '<polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/>',
    "alert": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>',
    "route": '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/>'
             '<line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
}


def icon(name: str) -> str:
    return f'<svg viewBox="0 0 24 24">{ICONS[name]}</svg>'


def inject_css(path: Path = STYLE_PATH) -> None:
    css = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def hero(subtitle: str, chips: list) -> None:
    chip_html = "".join(f"<span>{html.escape(c)}</span>" for c in chips)
    ecg = ('<svg class="hero-ecg" viewBox="0 0 360 60"><polyline points="0,30 60,30 76,30 88,8 100,52 112,18 122,30 190,30 '
           '206,30 218,6 230,54 242,20 252,30 360,30"/></svg>')
    st.markdown(
        f'<div class="hero">{ecg}<div class="hero-top"><div class="hero-logo">{icon("pulse")}</div>'
        f'<div><div class="hero-kicker">Clinical pathway process mining</div><div class="hero-title">CareFlow</div></div></div>'
        f'<div class="hero-sub">{html.escape(subtitle)}</div><div class="chips">{chip_html}</div></div>',
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    st.sidebar.markdown(
        f'<div class="side-brand"><div class="side-logo">{icon("pulse")}</div>'
        f'<div><div class="side-name">CareFlow</div><div class="side-tag">Sepsis pathway analytics</div></div></div>',
        unsafe_allow_html=True,
    )


def sidebar_title(text: str) -> None:
    st.sidebar.markdown(f'<div class="side-title">{html.escape(text)}</div>', unsafe_allow_html=True)


def selected_box(selected: int, total: int) -> None:
    width = selected / total * 100 if total else 0
    st.sidebar.markdown(
        f'<div class="selected"><div class="selected-label">Patients selected</div>'
        f'<div class="selected-value">{selected:,} <span>of {total:,}</span></div>'
        f'<div class="selected-bar"><div style="width:{width:.1f}%"></div></div></div>',
        unsafe_allow_html=True,
    )


def kpi(column, label: str, value: str, sub: str = "", icon_name: str = "pulse") -> None:
    column.markdown(
        f'<div class="kpi"><div class="kpi-icon">{icon(icon_name)}</div><div>'
        f'<div class="kpi-label">{html.escape(label)}</div><div class="kpi-value">{html.escape(value)}</div>'
        f'<div class="kpi-sub">{html.escape(sub)}</div></div></div>',
        unsafe_allow_html=True,
    )


def insight(column, tag: str, value: str, note: str, icon_name: str = "pulse") -> None:
    column.markdown(
        f'<div class="insight"><div class="insight-head">{icon(icon_name)}<span class="insight-tag">{html.escape(tag)}</span></div>'
        f'<div class="insight-value">{html.escape(value)}</div><div class="insight-note">{html.escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


def section(title: str, note: str = "") -> None:
    st.markdown(f'<div class="section"><span class="section-title">{html.escape(title)}</span></div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="section-note">{html.escape(note)}</div>', unsafe_allow_html=True)


def callout(text_html: str) -> None:
    """`text_html` is trusted, hand-written HTML (it may contain <b>)."""
    st.markdown(f'<div class="callout">{text_html}</div>', unsafe_allow_html=True)


def spacer(height_rem: float = 0.8) -> None:
    st.markdown(f'<div style="height:{height_rem}rem"></div>', unsafe_allow_html=True)


def footer(text: str) -> None:
    st.markdown(f'<div class="footer">{html.escape(text)}</div>', unsafe_allow_html=True)
