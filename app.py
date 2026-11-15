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
            st.caption("GET STARTED")
            st.subheader("Choose your data")
            st.write("Try the built-in example or bring a customer CSV.")
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
        caption("Use public or fictional data. Uploads are processed on the server and cached in memory for up to 30 minutes.")
    return content, label


def customers_view(bundle: dict[str, Any]) -> None:
    X, y, result = bundle["X"], bundle["y"], bundle["result"]
    st.subheader("Start with the customers")
    st.write("Churn means a customer left the service. These are recorded historical outcomes, not model predictions.")
    a, b = st.columns([1.1, 1])
    with a:
        st.markdown("#### How many stayed or left?")
        draw(charts.outcomes_chart(y))
        st.write(f"**{int(y.sum()):,} of {len(y):,} customers left ({y.mean():.1%}).** A model can miss churners even if it correctly predicts that most customers stay.")
    with b:
        st.markdown("#### Is anything missing?")
        draw(charts.missing_chart(X))
        missing = int(X.isna().sum().sum())
        st.write(f"**{missing:,} feature values need filling.** Replacement values are learned from training customers only.")
    st.markdown("#### Where does observed churn differ?")
    features = list(X.columns)
    default = features.index("Contract") if "Contract" in features else 0
    feature = st.selectbox("Compare customer groups by", features, index=default, key="group_feature", format_func=charts.feature_name)
    summary = charts.feature_summary(X, y, feature)
    draw(charts.feature_chart(summary))
    caption("Each bar is the share who left within that group. Tap or hover for details, or open the values below. Small groups are less reliable. Numeric inputs are divided into six equal-width ranges. These are associations across the uploaded data, not proof of causes.")
    with st.expander("Group rates and customer counts"):
        values = summary[["Group", "Customers", "Churn rate"]].copy()
        values["Churn rate"] = values["Churn rate"].map(lambda rate: f"{rate:.1%}")
        st.dataframe(values, hide_index=True, use_container_width=True)
    with st.expander("Cleaning notes and supported CSV format"):
        for note in bundle["notes"]:
            st.write(note)
        st.write("Provide 100–15,000 rows, a Churn column containing Yes/No or 1/0, at least 20 examples in each class, and supported Telco features. customerID is optional and must be unique when supplied. Maximum file size: 10 MB.")
        st.markdown(f"[Full input column list and cleaning rules]({REPO}/blob/main/docs/METHODOLOGY.md#input-data)")
        st.dataframe(X.head(10), use_container_width=True)
    with st.expander("Which customers are used at each stage?"):
        sizes = pd.DataFrame({"Stage": ["Training: learn patterns", "Validation: choose model", "Test: check results"], "Customers": [len(result[k]) for k in ["train_idx", "val_idx", "test_idx"]]})
        st.bar_chart(sizes.set_index("Stage"), color=charts.TEAL)
        st.write("The split is approximately 60% / 20% / 20%. Test customers do not teach or select the model. Rankings and campaign scenarios use this unseen test group.")


def models_view(bundle: dict[str, Any]) -> None:
    result, y = bundle["result"], bundle["y"]
    truth = y.iloc[result["test_idx"]]
    selected = result["metrics"].set_index("Model").loc[result["winner"]]
    st.subheader("How useful are the predictions?")
    st.write("Two models learn the same training customers. Logistic regression is the simpler reference; XGBoost combines small decision trees to learn more complex patterns.")
    st.success(f"Selected model: **{result['winner']}** — chosen using ranking quality on validation customers, before looking at test outcomes.")
    a, b, c = st.columns(3)
    a.metric("Correct churn flags · precision", f"{selected['Test precision']:.1%}")
    b.metric("Actual churners found · recall", f"{selected['Test recall']:.1%}")
    c.metric("Test ranking score · AP", f"{selected['Test AP']:.3f}")
    a, b = st.columns(2)
    with a:
        st.markdown("#### Correct flags vs churners found")
        draw(charts.comparison_chart(result["metrics"]))
        st.write("**Precision:** of the customers flagged as leaving, how many really left? **Recall:** of all customers who left, how many did we flag?")
        caption("Both models flag churn at a score of 50% or more. A higher bar is better for that measure; the two measures can trade off.")
    with b:
        st.markdown("#### The tradeoff across decision thresholds")
        draw(charts.pr_chart(truth, bundle["scores"]))
        st.write("Each line shows the precision–recall tradeoff as the cutoff changes. Closer to the top-right is better. The dashed line is the test churn rate, a no-skill reference.")
        caption("AP (average precision) summarizes this ranking performance. It is not accuracy. This chart does not change the fixed 50% cutoff or choose the model.")
    st.markdown("#### Where are the mistakes?")
    names = list(result["models"])
    name = st.selectbox("Confusion matrix for", names, index=names.index(result["winner"]), key="matrix_model")
    a, b = st.columns([1, 1])
    matrix = result["matrices"][name]
    with a:
        draw(charts.confusion_chart(matrix))
    with b:
        st.markdown(f"**{matrix[1, 1]:,} churners found**\n\nCustomers who left and were correctly flagged.")
        st.markdown(f"**{matrix[1, 0]:,} churners missed**\n\nCustomers who left but were predicted to stay.")
        st.markdown(f"**{matrix[0, 1]:,} false alarms**\n\nCustomers flagged as leaving who actually stayed.")
        caption(f"The remaining {matrix[0, 0]:,} customers were correctly predicted to stay. Rows show actual outcomes; columns show predictions.")
    with st.expander("Exact scores and selection details"):
        st.dataframe(result["metrics"].round(4), hide_index=True, use_container_width=True)
        st.write("Validation AP chooses the model. Test AP reports its ranking quality on unseen customers. Brier score measures squared probability error; lower is better. It does not alone prove that the probabilities match real-world frequencies.")
        st.markdown(f"[Evaluation method and limitations]({REPO}/blob/main/docs/METHODOLOGY.md#evaluation)")


def risk_view(bundle: dict[str, Any], content: bytes, ranked: pd.DataFrame) -> None:
    st.subheader("Explore risk, then inspect a customer")
    st.write("These scores belong to customers excluded from training and model selection. They estimate risk; they do not guarantee who will leave.")
    a, b = st.columns(2)
    with a:
        st.markdown("#### How is risk distributed?")
        draw(charts.distribution_chart(ranked))
        caption("Each bar counts customers in a 10-point risk range. A score of 70% is an estimate, not a certainty; probability calibration has not been established.")
    with b:
        st.markdown("#### Does ranking help find churners?")
        capture = charts.capture_summary(ranked, bundle["y"])
        draw(charts.capture_chart(capture))
        count = int(np.ceil(len(ranked) * 0.2))
        total = int(bundle["y"].iloc[ranked.row_index].sum())
        found = int(bundle["y"].iloc[ranked.head(count).row_index].sum())
        st.write(f"The top **20%** of test customers contain **{found / total:.1%}** of the actual churners." if total else "There are no actual churners in this test group.")
        caption("The dashed line represents random selection in expectation. This is a historical ranking check, not evidence that an offer prevents churn.")
    st.markdown("#### Customer risk list")
    visible = ranked[["customerID", "risk_score"]].copy()
    visible["risk_percent"] = (visible.pop("risk_score").astype(float) * 100).round(1)
    st.dataframe(visible, column_config={
        "customerID": st.column_config.TextColumn("Customer", width="small"),
        "risk_percent": st.column_config.ProgressColumn("Risk", width="small", format="%.1f%%", min_value=0, max_value=100),
    }, hide_index=True, height=250, use_container_width=True)
    st.download_button("Download test-customer rankings", visible.to_csv(index=False, float_format="%.1f"), "churn_rankings.csv", "text/csv", use_container_width=True)
    st.markdown("#### What influenced an individual prediction?")
    ids = ranked.customerID.tolist()
    previous = st.session_state.get("selected_customer")
    if st.session_state.get("customer") not in ids:
        st.session_state["customer"] = previous if previous in ids else ids[0]
    selected = st.selectbox("Customer", ids, key="customer")
    st.session_state["selected_customer"] = selected
    record = ranked.loc[ranked.customerID == selected].iloc[0]
    st.metric("This customer's estimated churn risk", f"{record.risk_score:.1%}")
    try:
        reasons = cached_explanation(content, int(record.row_index))
        draw(charts.shap_chart(reasons))
        up = reasons.loc[reasons.Contribution > 1e-8]
        down = reasons.loc[reasons.Contribution < -1e-8]
        a, b = st.columns(2)
        with a:
            st.markdown("**Strongest upward influence**")
            st.write(f"{charts.feature_name(up.iloc[0].Feature)} = {up.iloc[0].Value}" if not up.empty else "No feature pushes the score up relative to this reference.")
        with b:
            st.markdown("**Strongest downward influence**")
            st.write(f"{charts.feature_name(down.iloc[0].Feature)} = {down.iloc[0].Value}" if not down.empty else "No feature pushes the score down relative to this reference.")
        st.write("Read from the zero line: coral bars push the model's risk estimate up; teal bars push it down. Longer bars have more influence on this prediction.")
        caption("SHAP shares the model score among inputs relative to a reference. Bar lengths use log-odds, the model's internal score scale, not percentage-point changes in probability. These are model patterns, not proven causes.")
        with st.expander("All contributions and customer inputs"):
            st.dataframe(reasons.round({"Contribution": 3}), hide_index=True, use_container_width=True)
            st.dataframe(bundle["X"].iloc[[int(record.row_index)]], hide_index=True, use_container_width=True)
            st.markdown(f"[How to interpret SHAP]({REPO}/blob/main/docs/READING_THE_DASHBOARD.md#individual-risk)")
    except Exception:
        logging.exception("SHAP explanation failed")
        st.warning("We could not calculate this explanation. The rankings and retention scenario remain available.")


def campaign_view(ranked: pd.DataFrame) -> None:
    st.subheader("Would a retention offer be worth its cost?")
    st.write("Choose a campaign size and assumptions. The app estimates what could happen; the data does not tell us how effective an offer would be.")
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
    caption("Example: a 15% success rate assumes the offer prevents churn for 15 out of 100 contacted customers who would otherwise leave. This rate is assumed, not learned.")
    savings = estimate_savings(ranked, share, cost, value, success / 100)
    a, b, c = st.columns(3)
    a.metric("Customers contacted", f"{savings['contacted']:,}")
    b.metric("Expected customers retained", f"{savings['expected_retained']:.1f}")
    c.metric("Scenario net savings", f"{savings['net_savings']:,.2f}")
    a, b = st.columns(2)
    with a:
        st.markdown("#### Where the money comes from")
        draw(charts.money_chart(savings))
        st.write(f"Avoided loss of **{savings['avoided_loss']:,.2f}**, minus offer spend of **{savings['offer_spend']:,.2f}**, gives net savings of **{savings['net_savings']:,.2f}** in your chosen currency.")
    with b:
        st.markdown("#### What if you contacted more customers?")
        draw(charts.savings_curve_chart(charts.campaign_curve(ranked, cost, value, success / 100), share))
        st.write("The dot marks your chosen contact share. Above the dashed zero line means positive scenario savings; below it means offer costs exceed the estimated benefit.")
        caption("All points use the same cost, value, and assumed success rate. The curve explores assumptions; its highest point is not a validated campaign recommendation.")
    st.warning("Estimated savings depend on both the model probabilities and your success assumption. A randomized campaign is needed to measure whether the offer actually reduces churn.")
    with st.expander("Calculation and assumptions"):
        st.write("Expected retained customers = sum of contacted customers' risk scores × assumed success rate.\n\nNet savings = expected retained customers × customer value − contacted customers × offer cost.")
        st.write("A fractional retained-customer count is an average expected across similar campaigns, not part of a person. Customer value and offer cost must use one currency and a consistent period.")
        st.markdown(f"[Full savings method]({REPO}/blob/main/docs/METHODOLOGY.md#retention-scenario)")


def main() -> None:
    apply_layout()
    if has_customer_data():
        st.caption("CUSTOMER RETENTION / PREDICTION & PLANNING")
        st.title("Understand customer churn")
        st.write("See who may leave, understand the model's reasons, and explore the cost of acting.")
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
    caption(label + " · Historical evaluation · No offer effects have been measured")
    with st.container(key="summary_metrics"):
        a, b, c, d = st.columns(4)
        a.metric("Historical customers", f"{len(bundle['X']):,}")
        b.metric("Observed churn", f"{bundle['y'].mean():.1%}")
        c.metric("Test customers", f"{len(bundle['result']['test_idx']):,}")
        d.metric("Selected model", bundle["result"]["winner"])
    section = select_section()
    caption("Choose a section above. Charts and controls adapt to your screen; detailed explanations are linked below.")
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
    caption("Predictions describe patterns, not causes. Changing an input does not prove a customer will stay.")
    st.markdown(f"[Read the dashboard guide]({REPO}/blob/main/docs/READING_THE_DASHBOARD.md) · [Methodology]({REPO}/blob/main/docs/METHODOLOGY.md) · [Source code]({REPO})")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        st.error(str(exc))
    except Exception:
        logging.exception("Application failed")
        st.error("We could not finish this run. Try the sample data or a smaller Telco-style CSV. If the problem continues, check the server logs.")
