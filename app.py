"""Visual customer churn dashboard."""
import logging
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src import charts
from src.business import estimate_savings
from src.data import clean_data, load_sample, read_csv_bytes
from src.explain import explain_customer
from src.model import rank_customers, train_models
from src.layout import apply_layout, select_section, has_customer_data, landing_intro, landing_features

REPO = "https://github.com/NikosApostolo/customer-churn-tool"
st.set_page_config(page_title="Customer Churn | Retention dashboard", page_icon="◉", layout="wide")


@st.cache_data(ttl=3600, max_entries=2, show_spinner=False)
def cached_sample() -> tuple[pd.DataFrame, str]:
    return load_sample()


@st.cache_data(ttl=1800, max_entries=3, show_spinner=False)
def cached_training(content: bytes) -> dict[str, Any]:
    X, y, ids, notes = clean_data(read_csv_bytes(content))
    result = train_models(X, y)
    scores = {name: model.predict_proba(X.iloc[result["test_idx"]])[:, 1] for name, model in result["models"].items()}
    return {"X": X, "y": y, "ids": ids, "notes": notes, "result": result, "scores": scores}


@st.cache_data(ttl=1800, max_entries=100, show_spinner=False)
def cached_explanation(content: bytes, row_index: int) -> pd.DataFrame:
    bundle = cached_training(content)
    result, X = bundle["result"], bundle["X"]
    return explain_customer(result["models"][result["winner"]], X.iloc[result["train_idx"]], X.iloc[[row_index]])


def draw(chart: Any) -> None:
    st.altair_chart(chart, use_container_width=True)


def caption(text: str) -> None:
    st.caption(text)


def select_sample() -> None:
    """Load before rerendering so the welcome screen becomes the dashboard."""
    frame, label = cached_sample()
    st.session_state["sample_content"] = frame.to_csv(index=False).encode("utf-8")
    st.session_state["sample_label"] = label


def load_input() -> tuple[bytes | None, str]:
    loaded = has_customer_data()
    panel = st.expander("Change customer data", expanded=False) if loaded else st.container(border=True, key="data_start")
    with panel:
        if not loaded:
            st.subheader("Get started")
        source = st.radio("Choose data", ["Sample data", "Upload CSV"], key="input_source", horizontal=True)
        if source == "Sample data":
            if not loaded:
                st.caption("IBM Telco sample · 7,043 historical customers · No file needed")
            st.button("Use sample data", type="primary", use_container_width=True, on_click=select_sample)
            content = st.session_state.get("sample_content")
            label = st.session_state.get("sample_label", "IBM Telco sample")
        else:
            upload = st.file_uploader("Historical customer CSV", type=["csv"], key="customer_upload")
            content = upload.getvalue() if upload is not None else None
            label = "Uploaded customer data"
            if not loaded:
                st.caption("Include a Churn column with Yes/No or 1/0 historical outcomes. Maximum 10 MB.")
        caption("Use public or fictional data only.")
        st.markdown(f"[CSV requirements and privacy]({REPO}/blob/main/docs/METHODOLOGY.md#input-data)")
    return content, label


def customers_view(bundle: dict[str, Any]) -> None:
    X, y, result = bundle["X"], bundle["y"], bundle["result"]
    st.subheader("Your customers")
    draw(charts.outcomes_chart(y))
    caption(f"{int(y.sum()):,} customers left ({y.mean():.1%}). These are historical outcomes.")
    st.markdown("#### Churn by customer group")
    features = list(X.columns)
    feature = st.selectbox("Compare by", features, index=features.index("Contract") if "Contract" in features else 0, key="group_feature", format_func=charts.feature_name)
    summary = charts.feature_summary(X, y, feature)
    draw(charts.feature_chart(summary))
    caption("Bars show the share who left in each group. A difference is an association, not a cause.")
    with st.expander("Group counts"):
        values = summary[["Group", "Customers", "Churn rate"]].copy()
        values["Churn rate"] = values["Churn rate"].map(lambda v: f"{v:.1%}")
        st.dataframe(values, hide_index=True, use_container_width=True)
    with st.expander("Data checks"):
        missing = int(X.isna().sum().sum())
        st.write(f"{missing:,} feature values need filling, using training data only.")
        if missing:
            draw(charts.missing_chart(X))
        for note in bundle["notes"]:
            st.write(note)
        st.dataframe(X.head(10), use_container_width=True)
        caption("Split: 60% training, 20% validation, 20% test. Test customers do not train or select the model.")
        st.markdown(f"[Data and training details]({REPO}/blob/main/docs/METHODOLOGY.md)")


def models_view(bundle: dict[str, Any]) -> None:
    result = bundle["result"]
    st.subheader("Model comparison")
    caption(f"Selected: {result['winner']} · chosen on validation data. Results below use unseen test customers.")
    draw(charts.comparison_chart(result["metrics"]))
    st.write("**Precision:** how many flagged customers actually left.\n\n**Recall:** how many actual churners were found.")
    names = list(result["models"])
    name = st.selectbox("Confusion matrix for", names, index=names.index(result["winner"]), key="matrix_model")
    draw(charts.confusion_chart(result["matrices"][name]))
    matrix = result["matrices"][name]
    caption(f"{matrix[1, 1]:,} churners found · {matrix[1, 0]:,} missed · {matrix[0, 1]:,} false alarms. Rows = actual; columns = predicted.")
    with st.expander("Detailed evaluation"):
        draw(charts.pr_chart(bundle["y"].iloc[result["test_idx"]], bundle["scores"]))
        caption("Higher precision and recall are better. The dashed line is the test churn rate. This chart varies the cutoff for display only.")
        st.dataframe(result["metrics"].round(4), hide_index=True, use_container_width=True)
        st.write("Both models flag churn at 50% risk. Validation average precision (AP) selects the model; test outcomes do not. Brier score measures probability error, with lower values better.")
        st.markdown(f"[How evaluation works]({REPO}/blob/main/docs/METHODOLOGY.md#evaluation)")


def risk_view(bundle: dict[str, Any], content: bytes, ranked: pd.DataFrame) -> None:
    st.subheader("Customer risk")
    caption(f"{len(ranked):,} unseen test customers, highest risk first. Scores are estimates, not certainties.")
    visible = ranked[["customerID", "risk_score"]].copy()
    visible["risk_percent"] = (visible.pop("risk_score").astype(float) * 100).round(1)
    st.dataframe(visible, column_config={
        "customerID": st.column_config.TextColumn("Customer", width="small"),
        "risk_percent": st.column_config.ProgressColumn("Risk", width="small", format="%.1f%%", min_value=0, max_value=100),
    }, hide_index=True, height=250, use_container_width=True)
    st.download_button("Download rankings", visible.to_csv(index=False, float_format="%.1f"), "churn_rankings.csv", "text/csv", use_container_width=True)
    ids = ranked.customerID.tolist()
    previous = st.session_state.get("selected_customer")
    if st.session_state.get("customer") not in ids:
        st.session_state["customer"] = previous if previous in ids else ids[0]
    selected = st.selectbox("Customer", ids, key="customer")
    st.session_state["selected_customer"] = selected
    record = ranked.loc[ranked.customerID == selected].iloc[0]
    st.metric("Estimated churn risk", f"{record.risk_score:.1%}")
    try:
        reasons = cached_explanation(content, int(record.row_index))
        draw(charts.shap_chart(reasons))
        caption("Coral pushes the model score up; teal pushes it down. Bar length shows influence, not a probability change or proven cause.")
        top = reasons.iloc[0]
        if abs(float(top.Contribution)) > 1e-8:
            direction = "up" if top.Contribution > 0 else "down"
            st.write(f"Strongest influence: **{charts.feature_name(top.Feature)} = {top.Value}**, pushing the score {direction}.")
        with st.expander("Customer details and all influences"):
            st.dataframe(reasons.round({"Contribution": 3}), hide_index=True, use_container_width=True)
            st.dataframe(bundle["X"].iloc[[int(record.row_index)]], hide_index=True, use_container_width=True)
            st.markdown(f"[How SHAP explanations work]({REPO}/blob/main/docs/READING_THE_DASHBOARD.md#individual-risk)")
    except Exception:
        logging.exception("SHAP explanation failed")
        st.warning("This explanation is unavailable. Rankings and campaign calculations still work.")
    with st.expander("Check the risk ranking"):
        draw(charts.distribution_chart(ranked))
        caption("Customers in each 10-point risk range. Probability calibration has not been established.")
        draw(charts.capture_chart(charts.capture_summary(ranked, bundle["y"])))
        count = int(np.ceil(len(ranked) * 0.2))
        truth = bundle["y"]
        total = int(truth.iloc[ranked.row_index].sum())
        found = int(truth.iloc[ranked.head(count).row_index].sum())
        if total:
            st.write(f"The top 20% contain {found / total:.1%} of actual test churners. Random selection would find about 20% on average.")
        caption("This measures historical ranking quality, not whether offers prevent churn.")


def campaign_view(ranked: pd.DataFrame) -> None:
    st.subheader("Retention scenario")
    caption("Set costs and an assumed offer effect. Use the same currency for both amounts.")
    settings = st.session_state.setdefault("campaign_settings", {"cost": 20.0, "value": 500.0, "share": 20, "success": 15})
    for widget_key, setting in [("offer_cost", "cost"), ("customer_value", "value"), ("contact_share", "share"), ("offer_success", "success")]:
        if widget_key not in st.session_state:
            st.session_state[widget_key] = settings[setting]
    a, b = st.columns(2)
    cost = a.number_input("Offer cost per contacted customer", min_value=0.0, value=None, step=1.0, key="offer_cost", help="Assume this cost is paid for everyone contacted.")
    value = b.number_input("Value of a lost customer", min_value=0.0, value=None, step=10.0, key="customer_value", help="Use lost profit or contribution over one defined period, in the same currency as the offer cost.")
    share = st.slider("Contact the top X% by risk", 0, 100, key="contact_share")
    success = st.slider("Assumed share of would-be churners saved by the offer (%)", 0, 100, key="offer_success")
    if cost is None or value is None:
        st.info("Enter an offer cost and customer value to calculate the scenario.")
        return
    settings.update(cost=cost, value=value, share=share, success=success)
    caption("Offer success means the share of would-be churners the offer prevents from leaving. It is an assumption.")
    savings = estimate_savings(ranked, share, cost, value, success / 100)
    a, b, c = st.columns(3)
    a.metric("Customers contacted", f"{savings['contacted']:,}")
    b.metric("Expected customers retained", f"{savings['expected_retained']:.1f}")
    c.metric("Scenario net savings", f"{savings['net_savings']:,.2f}")
    draw(charts.savings_curve_chart(charts.campaign_curve(ranked, cost, value, success / 100), share))
    caption("The dot marks your contact share. Above zero = estimated gain; below zero = estimated loss.")
    rate = savings["break_even_success_rate"]
    if savings["contacted"] == 0:
        st.info("No customers selected. Costs and savings are zero.")
    elif not np.isfinite(rate) and savings["offer_spend"] == 0:
        st.info("No modeled benefit or offer spend: net savings are zero.")
    elif not np.isfinite(rate) or rate > 1:
        st.info("These costs cannot be recovered even if every expected churner is retained.")
    else:
        st.write(f"**Break-even offer success: {rate:.1%}.** Your assumption: {success}%.")
    caption("Scenario only: offer effectiveness is unmeasured and risk probabilities are not fully calibrated.")
    with st.expander("Cost breakdown and calculation"):
        draw(charts.money_chart(savings))
        st.write(f"Avoided loss {savings['avoided_loss']:,.2f} − offer spend {savings['offer_spend']:,.2f} = net savings {savings['net_savings']:,.2f}.")
        st.write("Expected retained customers = sum of contacted risk scores × assumed offer success. Costs apply to every contacted customer.")
        st.markdown(f"[Assumptions and savings formula]({REPO}/blob/main/docs/METHODOLOGY.md#retention-scenario)")


def main() -> None:
    apply_layout()
    if has_customer_data():
        st.title("Customer churn")
        content, label = load_input()
    else:
        with st.container(key="landing_start"):
            intro, start = st.columns([1.15, 1])
            with intro:
                landing_intro()
            with start:
                content, label = load_input()
        landing_features()
    if content is None:
        st.markdown(f"[Read how the app works]({REPO}/blob/main/docs/READING_THE_DASHBOARD.md) · [Project and methodology]({REPO}#readme)")
        return
    with st.spinner("Preparing the data and comparing models. The first run takes a little longer…"):
        bundle = cached_training(content)
    if "SYNTHETIC" in label:
        st.warning("The public download was unavailable. You are viewing fictional offline data, not the Telco dataset.")
    caption(f"{label} · {len(bundle['X']):,} customers · {bundle['y'].mean():.1%} left · {len(bundle['result']['test_idx']):,} held out for testing")
    section = select_section()
    ranked = rank_customers(bundle["result"], bundle["X"], bundle["ids"])
    if section == "Customers":
        customers_view(bundle)
    elif section == "Models":
        models_view(bundle)
    elif section == "Individual risk":
        risk_view(bundle, content, ranked)
    else:
        campaign_view(ranked)
    st.divider()
    st.markdown(f"[Guide and methodology]({REPO}/blob/main/docs/READING_THE_DASHBOARD.md) · [Source code]({REPO})")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        st.error(str(exc))
    except Exception:
        logging.exception("Application failed")
        st.error("We could not finish this run. Try the sample data or a smaller Telco-style CSV. If the problem continues, check the server logs.")
