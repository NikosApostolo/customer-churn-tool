"""Customer Churn Prediction Tool: Streamlit entry point."""
import logging

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.business import estimate_savings
from src.data import clean_data, load_sample, read_csv_bytes
from src.explain import explain_customer, explanation_sentence
from src.model import rank_customers, train_models

st.set_page_config(page_title="Customer Churn Prediction Tool", page_icon="📊", layout="wide")


@st.cache_data(ttl=3600, max_entries=2, show_spinner=False)
def cached_sample() -> tuple[pd.DataFrame, str]:
    return load_sample()


@st.cache_data(ttl=1800, max_entries=3, show_spinner=False)
def cached_training(content: bytes) -> dict:
    X, y, ids, notes = clean_data(read_csv_bytes(content))
    result = train_models(X, y)
    return {"X": X, "y": y, "ids": ids, "notes": notes, "result": result}


@st.cache_data(ttl=1800, max_entries=100, show_spinner=False)
def cached_explanation(content: bytes, row_index: int) -> pd.DataFrame:
    bundle = cached_training(content)
    result, X = bundle["result"], bundle["X"]
    return explain_customer(
        result["models"][result["winner"]], X.iloc[result["train_idx"]],
        X.iloc[[row_index]],
    )


def main() -> None:
    st.title("Customer Churn Prediction Tool")
    st.write("Compare two models, inspect customer risk, and explore retention scenarios.")
    st.info(
        "Upload historical Telco-style data with a Churn column (Yes/No or 1/0). "
        "The rankings use only the unseen test customers, so this is an honest historical demo."
    )
    st.caption(
        "Use public or fictional data here. Uploads are processed on the server and "
        "cached in server memory for up to 30 minutes; they are not written to a database."
    )
    with st.expander("Supported CSV columns"):
        st.write(
            "Supported features: tenure, MonthlyCharges, TotalCharges, SeniorCitizen, "
            "gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, "
            "OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, "
            "StreamingMovies, Contract, PaperlessBilling, PaymentMethod. "
            "Include at least one feature and Churn. customerID is optional. "
            "Only include information that would be available before the churn outcome. "
            "Other columns are ignored. Maximum: 10 MB and 15,000 rows."
        )
    source = st.radio("Choose data", ["Sample data", "Upload CSV"], horizontal=True)
    if source == "Sample data":
        if st.button("Use sample data", type="primary"):
            with st.spinner("Loading sample data…"):
                frame, label = cached_sample()
            st.session_state["sample_content"] = frame.to_csv(index=False).encode("utf-8")
            st.session_state["sample_label"] = label
        if "sample_content" not in st.session_state:
            st.write("Click Use sample data to begin.")
            return
        content = st.session_state["sample_content"]
        label = st.session_state["sample_label"]
        st.caption(label)
        if "SYNTHETIC" in label:
            st.warning("The public download failed. Results below use fictional offline data.")
    else:
        upload = st.file_uploader("Upload a UTF-8 customer CSV", type=["csv"])
        if upload is None:
            return
        content = upload.getvalue()
    with st.spinner("Cleaning data and training both models…"):
        bundle = cached_training(content)
    X, y, ids, result = (bundle[key] for key in ["X", "y", "ids", "result"])
    with st.expander("Data checks and cleaning"):
        for note in bundle["notes"]:
            st.write("• " + note)
        st.dataframe(X.head(), use_container_width=True)
    a, b, c = st.columns(3)
    a.metric("Historical customers", f"{len(X):,}")
    b.metric("Observed churn", f"{y.mean():.1%}")
    c.metric("Test customers", f"{len(result['test_idx']):,}")
    st.subheader("1. Honest model comparison")
    st.write(
        "60% of customers teach the models, 20% select the model, and the final 20% "
        "measure performance. Missing-value filling, scaling and category encoding "
        "learn from the training group only. Both models use a fixed 0.50 decision threshold."
    )
    st.dataframe(result["metrics"].round(3), hide_index=True, use_container_width=True)
    st.success(f"Selected using validation average precision: {result['winner']}")
    st.write("Precision: of the customers flagged as leaving, the fraction who actually left.")
    st.write("Recall: of all customers who actually left, the fraction the model flagged.")
    st.write("Average precision (AP): summarizes how well the model ranks churners ahead of others across thresholds; higher is better.")
    st.write("Brier score: the average squared error of probability predictions; lower is better.")
    st.caption("AP should be interpreted against the churn rate. Neither model is assumed to win in advance.")
    chosen = st.selectbox("Confusion matrix for", list(result["models"]))
    st.write("Confusion matrix: counts correct and incorrect decisions separately for customers who stayed and left.")
    st.dataframe(pd.DataFrame(
        result["matrices"][chosen], index=["Actually stayed", "Actually left"],
        columns=["Predicted stay", "Predicted leave"],
    ))
    st.subheader("2. Customers ranked by risk")
    ranked = rank_customers(result, X, ids)
    visible = ranked[["customerID", "risk_score"]].copy()
    visible["risk_percent"] = (visible.pop("risk_score") * 100).round(1)
    st.dataframe(visible, hide_index=True, use_container_width=True)
    st.caption("Scores are model estimates, not guaranteed or fully calibrated probabilities. Training customers are excluded.")
    st.download_button("Download test-customer rankings", visible.to_csv(index=False), "churn_rankings.csv", "text/csv")
    st.subheader("3. Why did the model assign this risk?")
    selected = st.selectbox("Customer", ranked["customerID"].tolist())
    row_index = int(ranked.loc[ranked["customerID"] == selected, "row_index"].iloc[0])
    try:
        with st.spinner("Calculating SHAP explanation…"):
            reasons = cached_explanation(content, row_index)
        top = reasons.head(7).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(top["Feature"], top["Contribution"], color=["#df6358" if v > 0 else "#3988bd" for v in top["Contribution"]])
        ax.axvline(0, color="gray", linewidth=1)
        ax.set_xlabel("SHAP contribution to log-odds (not percentage points)")
        ax.set_title(f"Main model influences for {selected}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.write(explanation_sentence(reasons))
        st.caption(
            "SHAP assigns a share of the prediction to each feature. Red pushes predicted "
            "risk up; blue pushes it down. Log-odds are the model's internal score scale. "
            "Contributions are grouped back to original columns. Only the seven largest are plotted. "
            "Explanations depend on the model and its reference data."
        )
        st.dataframe(reasons, hide_index=True)
    except Exception:
        logging.exception("SHAP explanation failed")
        st.warning("This explanation could not be calculated. Rankings and evaluation remain available.")
    st.subheader("4. Retention business scenario")
    st.write(
        "Explore a hypothetical campaign on the ranked test group. Choose one currency "
        "and use it consistently. Customer value should mean lost contribution or profit "
        "over a defined time horizon, rather than lifetime revenue."
    )
    a, b = st.columns(2)
    cost = a.number_input("Offer cost per contacted customer", min_value=0.0, value=20.0, step=1.0)
    value = b.number_input("Value of a lost customer", min_value=0.0, value=500.0, step=10.0)
    share = st.slider("Contact the top X% by risk", 0, 100, 20)
    success = st.slider("Assumed share of would-be churners saved by the offer (%)", 0, 100, 15)
    savings = estimate_savings(ranked, share, cost, value, success / 100)
    a, b, c = st.columns(3)
    a.metric("Customers contacted", savings["contacted"])
    b.metric("Expected customers retained", f"{savings['expected_retained']:.1f}")
    c.metric("Scenario net savings", f"{savings['net_savings']:,.2f}")
    st.write(f"Offer spend: {savings['offer_spend']:,.2f}; avoided loss: {savings['avoided_loss']:,.2f}.")
    st.code("Net savings = sum(contacted risk scores) × assumed offer success × customer value − contacted customers × offer cost")
    st.warning(
        "This is a scenario, not measured financial impact. The dataset contains no offer "
        "experiment, so the success rate is your assumption. Costs apply to everyone contacted. "
        "A model pattern does not prove that changing a feature will prevent churn. "
        "Validate probability quality and run a randomized campaign before relying on savings."
    )


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        st.error(str(exc))
    except Exception:
        logging.exception("Application failed")
        st.error("We could not complete this run. Try the sample data or a smaller Telco-style CSV. If it continues, check the server logs.")
