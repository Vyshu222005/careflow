"""Test harness: renders ONE dashboard page against the real data.

Streamlit's AppTest cannot click through st.navigation menus built from functions, so the tests run this script once per
page (the page is chosen with the CAREFLOW_TEST_PAGE environment variable). It uses the same data context and the same
error handling as the real app, so an error inside a page shows up as st.error and fails the test.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

import dashboard  # noqa: E402,F401
import ui  # noqa: E402
from dashboard.data import build_context, load_all  # noqa: E402
from dashboard.navigation import PAGES, safe  # noqa: E402

ui.inject_css()
loaded = load_all()
cases = loaded[1]
if os.environ.get("CAREFLOW_TEST_AGE"):
    cases = cases[cases["age_group"] == os.environ["CAREFLOW_TEST_AGE"]]
ctx = build_context(loaded, cases, "test selection")
spec = next(p for p in PAGES if p.key == os.environ["CAREFLOW_TEST_PAGE"])
ui.page_header(spec.title, spec.subtitle, ctx.filters_text)
safe(spec.render, ctx)
st.write("PAGE-RENDERED")
