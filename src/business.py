"""Transparent scenario calculations; these are not causal impact estimates."""
import math

import numpy as np
import pandas as pd


def estimate_savings(ranked: pd.DataFrame, contact_percent: float, offer_cost: float,
                     customer_value: float, success_rate: float) -> dict[str, float | int]:
    """Assume success_rate of otherwise-churning contacted customers are retained."""
    inputs = [contact_percent, offer_cost, customer_value, success_rate]
    if not all(math.isfinite(float(value)) for value in inputs):
        raise ValueError("Enter finite numbers for the scenario.")
    if not 0 <= contact_percent <= 100 or not 0 <= success_rate <= 1:
        raise ValueError("Contact share must be 0–100%; success rate must be 0–1.")
    if offer_cost < 0 or customer_value < 0:
        raise ValueError("Offer cost and customer value cannot be negative.")
    if "risk_score" not in ranked or ranked.empty:
        raise ValueError("Train a model before calculating savings.")
    scores = ranked["risk_score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Risk scores must be between zero and one.")
    count = math.ceil(len(ranked) * contact_percent / 100)
    selected = ranked.sort_values("risk_score", ascending=False).head(count)
    expected_churners = float(selected["risk_score"].sum())
    saved = expected_churners * success_rate
    cost = count * offer_cost
    benefit = saved * customer_value
    denominator = expected_churners * customer_value
    return {
        "contacted": count, "expected_churners": expected_churners,
        "expected_retained": saved, "offer_spend": cost,
        "avoided_loss": benefit, "net_savings": benefit - cost,
        "break_even_success_rate": cost / denominator if denominator else float("inf"),
    }
