"""Navigation: the left sidebar menu (four groups, ten pages), the filters, and a render check of every page."""
from pathlib import Path

import pytest

import dashboard  # noqa: F401
from dashboard.navigation import PAGES, SECTIONS

HARNESS = Path(__file__).resolve().parent / "page_harness.py"
APP = Path(__file__).resolve().parents[1] / "app.py"

EXPECTED_MENU = {
    "Monitor": ["Overview", "Patient Pathways", "Bottleneck Analysis"],
    "Quality & Flow": ["Quality & Conformance", "Department Flow"],
    "Intelligence": ["Predictive Intelligence", "What-if Simulator", "Smart Insights"],
    "Output": ["Reports", "About"],
}


def test_menu_has_the_requested_groups_and_pages_in_order():
    assert SECTIONS == list(EXPECTED_MENU)
    for section, titles in EXPECTED_MENU.items():
        assert [p.title for p in PAGES if p.section == section] == titles
    assert len(PAGES) == 10


def test_every_page_has_a_unique_url_title_icon_and_description():
    keys = [p.key for p in PAGES]
    assert len(set(keys)) == len(keys)
    assert all(p.title and p.subtitle and p.icon.startswith(":material/") for p in PAGES)


def test_overview_is_the_default_first_page():
    assert PAGES[0].key == "overview"


def test_every_earlier_view_is_still_reachable_as_a_page_or_a_sub_tab():
    """The pages that combine earlier views must show them as sub-tabs, so nothing was dropped."""
    import inspect

    import dashboard.navigation as navigation

    source = inspect.getsource(navigation)
    for subtab in ("Common pathways", "Process map", "Patient journey", "Conformance", "Care bundle", "Rework and returns", "Departments and handovers"):
        assert f'"{subtab}"' in source


@pytest.mark.integration
@pytest.mark.parametrize("key", [p.key for p in PAGES])
def test_each_page_renders_without_errors_on_real_data(real_paths, key, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CAREFLOW_TEST_PAGE", key)
    at = AppTest.from_file(str(HARNESS), default_timeout=600).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert any("PAGE-RENDERED" in m.value for m in at.markdown)


@pytest.mark.integration
def test_app_opens_on_the_overview_with_no_mode_switch_and_a_filters_section(real_paths):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=600).run()
    assert not at.exception and not at.error
    assert any("hero-title" in m.value for m in at.markdown)  # the Overview shows the full banner
    assert not [r for r in at.sidebar.radio if r.label == "View"]  # the Executive / Analyst switch is gone
    assert [e.label for e in at.sidebar.expander if "filters" in e.label.lower()]
    labels = {w.label for w in [*at.sidebar.multiselect, *at.sidebar.radio, *at.sidebar.date_input]}
    assert {"How the case ended", "Age group", "Returned to the ER", "Registered between"} <= labels


@pytest.mark.integration
@pytest.mark.parametrize("filter_name, setter, expected", [
    ("Returned to the ER", lambda w: w.set_value("Returned"), 278),
    ("Age group", lambda w: w.set_value(["80+"]), 413),
    ("How the case ended", lambda w: w.set_value(["Release B"]), 50),
])
def test_each_filter_changes_the_selection(real_paths, filter_name, setter, expected):
    import re

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=600).run()
    setter([w for w in [*at.sidebar.multiselect, *at.sidebar.radio] if w.label == filter_name][0])
    at.run()
    assert not at.exception and not at.error
    counter = " ".join(m.value for m in at.sidebar.markdown)
    found = re.search(r'selected-value">([\d,]+) <span>of ([\d,]+)', counter)
    assert found and int(found.group(1).replace(",", "")) == expected and found.group(2) == "995"
