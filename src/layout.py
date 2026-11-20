"""Responsive presentation shared by the dashboard sections."""
from pathlib import Path

import streamlit as st

SECTIONS = ["Customers", "Models", "Individual risk", "Retention plan", "Score active customers"]


def apply_layout() -> None:
    """Load local styles; media queries adapt without collecting device details."""
    css = Path(__file__).with_name("styles.css").read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def select_section() -> str:
    """A labeled native-style selector remains usable on narrow touch screens."""
    return st.selectbox("Explore the dashboard", SECTIONS, key="dashboard_section")


def has_customer_data() -> bool:
    """Choose a compact dashboard header once an input is available."""
    if st.session_state.get("input_source") == "Upload CSV":
        return st.session_state.get("customer_upload") is not None
    return st.session_state.get("sample_content") is not None


def landing_intro() -> None:
    """A short, semantic introduction with an honest project contribution."""
    st.markdown('''<div class="landing-hero">
<div class="landing-eyebrow"><span class="landing-dot"></span>CUSTOMER CHURN</div>
<h1 class="landing-title">Understand<br><span>who may leave.</span></h1>
<p class="landing-lead">Customer risk, clear explanations, and retention costs in one workflow.</p>
</div>''', unsafe_allow_html=True)


def landing_features() -> None:
    """Static SVG icons illustrate the existing workflow, without invented data."""
    st.markdown('''<div class="landing-features" aria-label="What you can explore">
<article class="landing-feature"><div class="feature-top"><span class="feature-icon feature-teal" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 20h16M7 16v-5m5 5V5m5 11V8" stroke-linecap="round"/></svg></span><span class="feature-step">01 / PREDICT</span></div><h2>See who is at risk</h2><p>Rank customers by estimated churn risk.</p></article>
<article class="landing-feature"><div class="feature-top"><span class="feature-icon feature-blue" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M11 4v16M11 7h8M5 12h6m0 5h6" stroke-linecap="round"/><circle cx="19" cy="7" r="1"/><circle cx="5" cy="12" r="1"/><circle cx="17" cy="17" r="1"/></svg></span><span class="feature-step">02 / EXPLAIN</span></div><h2>Understand the score</h2><p>See what influences each prediction.</p></article>
<article class="landing-feature"><div class="feature-top"><span class="feature-icon feature-coral" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 19h16M6 15l5-5 4 3 4-7m-5 0h5v5" stroke-linecap="round" stroke-linejoin="round"/></svg></span><span class="feature-step">03 / EXPLORE VALUE</span></div><h2>Weigh the cost of acting</h2><p>Compare offer costs with potential savings.</p></article>
</div><p class="landing-footnote">Established models. Savings depend on assumptions.</p>''', unsafe_allow_html=True)

