"""Navigation: a permanent left sidebar menu with four groups and ten pages.

    MONITOR            Overview, Patient Pathways, Bottleneck Analysis
    QUALITY & FLOW     Quality & Conformance, Department Flow
    INTELLIGENCE       Predictive Intelligence, What-if Simulator, Smart Insights
    OUTPUT             Reports, About

Pages that combine earlier views show them as sub-tabs (Patient Pathways: common pathways, process map, patient journey;
Quality & Conformance: conformance, care bundle; Department Flow: rework and returns, departments and handovers), so no
analysis was removed. The menu itself is Streamlit's native sidebar navigation, styled in assets/style.css.
"""
import traceback
from dataclasses import dataclass
from typing import Callable

import streamlit as st

import ui
from dashboard.tabs import (
    about, bottleneck_trends, care_bundle, conformance_tab, departments, insights_tab, journey, overview, pathways, predictive,
    process_map, reports_tab, rework_tab, simulator, waiting_times,
)

SMALL_SAMPLE = 30


def safe(render, ctx) -> None:
    """Run one page. If it fails, show a clear message instead of breaking the whole dashboard."""
    try:
        render(ctx)
    except Exception as error:  # noqa: BLE001 - the dashboard must keep working when one page fails
        st.error(f"This section could not be drawn: {error}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())


def _group(names: list, renders: list) -> Callable:
    """A page that shows several views as sub-tabs."""
    def render(ctx) -> None:
        for sub, view in zip(st.tabs(names), renders):
            with sub:
                safe(view, ctx)

    return render


def bottlenecks(ctx) -> None:
    """Waiting times plus the monthly delay trends, as one page."""
    safe(waiting_times.render, ctx)
    safe(bottleneck_trends.render, ctx)


@dataclass(frozen=True)
class PageSpec:
    key: str  # url path
    title: str
    icon: str
    section: str
    subtitle: str
    render: Callable


PAGES = [
    PageSpec("overview", "Overview", ":material/dashboard:", "Monitor", "The whole picture in one page.", overview.render),
    PageSpec("patient-pathways", "Patient Pathways", ":material/route:", "Monitor", "Which routes patients take, how they flow, and one patient's journey.",
             _group(["Common pathways", "Process map", "Patient journey"], [pathways.render, process_map.render, journey.render])),
    PageSpec("bottleneck-analysis", "Bottleneck Analysis", ":material/hourglass_top:", "Monitor", "Where patients wait, and how that changes over time.", bottlenecks),
    PageSpec("quality-conformance", "Quality & Conformance", ":material/verified:", "Quality & Flow", "Expected against actual pathways, and early sepsis care on time.",
             _group(["Conformance", "Care bundle"], [conformance_tab.render, care_bundle.render])),
    PageSpec("department-flow", "Department Flow", ":material/swap_horiz:", "Quality & Flow", "Repeated steps, return visits and department handovers.",
             _group(["Rework and returns", "Departments and handovers"], [rework_tab.render, departments.render])),
    PageSpec("predictive-intelligence", "Predictive Intelligence", ":material/psychology:", "Intelligence", "Can antibiotic delay be predicted, and why?", predictive.render),
    PageSpec("what-if-simulator", "What-if Simulator", ":material/tune:", "Intelligence", "Hypothetical shorter waits (not a forecast).", simulator.render),
    PageSpec("smart-insights", "Smart Insights", ":material/lightbulb:", "Intelligence", "What deserves attention first, and why.", insights_tab.render),
    PageSpec("reports", "Reports", ":material/picture_as_pdf:", "Output", "Download the findings as a PDF.", reports_tab.render),
    PageSpec("about", "About", ":material/info:", "Output", "Data, method, technologies and limitations.", about.render),
]
SECTIONS = ["Monitor", "Quality & Flow", "Intelligence", "Output"]


def _page(ctx, spec: PageSpec, hero_subtitle: str, chips: list):
    def run() -> None:
        if spec.key == "overview":
            ui.hero(hero_subtitle, chips)
        else:
            ui.page_header(spec.title, spec.subtitle, ctx.filters_text)
        if len(ctx.cases) < SMALL_SAMPLE:
            st.info(f"Only {len(ctx.cases)} patients are selected, so the numbers below can vary a lot.")
        safe(spec.render, ctx)

    return st.Page(run, title=spec.title, icon=spec.icon, url_path=spec.key, default=spec.key == "overview")


def render_navigation(ctx, hero_subtitle: str, chips: list) -> None:
    """Build the sidebar menu and run the page the user selected."""
    menu = {section: [_page(ctx, p, hero_subtitle, chips) for p in PAGES if p.section == section] for section in SECTIONS}
    st.navigation(menu, position="sidebar", expanded=True).run()
