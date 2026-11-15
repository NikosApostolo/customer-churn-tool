"""Responsive presentation shared by the dashboard sections."""
from pathlib import Path

import streamlit as st

SECTIONS = ["Customers", "Models", "Individual risk", "Retention plan"]


def apply_layout() -> None:
    """Load local styles; media queries adapt without collecting device details."""
    css = Path(__file__).with_name("styles.css").read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def select_section() -> str:
    """A labeled native-style selector remains usable on narrow touch screens."""
    return st.selectbox("Explore the dashboard", SECTIONS, key="dashboard_section")
