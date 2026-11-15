"""Interactive charts and the small summaries that feed them."""
import altair as alt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

TEAL = "#087F8C"
BLUE = "#5265B8"
CORAL = "#D96954"
GRAY = "#A5B1BE"
FEATURE_NAMES = {
    "tenure": "Months as a customer", "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges", "SeniorCitizen": "Senior citizen indicator",
    "gender": "Gender", "Partner": "Partner", "Dependents": "Dependents",
    "PhoneService": "Phone service", "MultipleLines": "Multiple phone lines",
    "InternetService": "Internet service", "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
    "TechSupport": "Technical support", "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies", "Contract": "Contract",
    "PaperlessBilling": "Paperless billing", "PaymentMethod": "Payment method",
}


def feature_name(column: str) -> str:
    return FEATURE_NAMES.get(column, column)


MODEL_COLORS = alt.Scale(domain=["Logistic regression", "XGBoost"], range=[BLUE, TEAL])


def finish(chart: alt.Chart, height: int = 270) -> alt.Chart:
    """Apply a consistent readable style without changing global settings."""
    return chart.properties(height=height, padding={"left": 0, "right": 6, "top": 8, "bottom": 0}).configure_view(stroke=None).configure_axis(
        labelFontSize=12, titleFontSize=12, titlePadding=10, tickCount=4, gridColor="#EEF1F5",
        domain=False, tickColor="#D5DCE5", labelColor="#526174", titleColor="#34445A",
    ).configure_legend(title=None, labelFontSize=12, orient="bottom")


def outcomes_chart(y: pd.Series) -> alt.Chart:
    counts = pd.DataFrame({"Outcome": ["Stayed", "Left"], "Customers": [int((y == 0).sum()), int(y.sum())]})
    counts["Share"] = counts.Customers / len(y)
    base = alt.Chart(counts).encode(
        y=alt.Y("Outcome:N", sort=["Stayed", "Left"], title=None),
        x=alt.X("Customers:Q", title="Customers", scale=alt.Scale(domain=[0, len(y) * 1.15])),
        tooltip=["Outcome", alt.Tooltip("Customers:Q", format=","), alt.Tooltip("Share:Q", format=".1%")],
    )
    bars = base.mark_bar(cornerRadiusEnd=5, size=42).encode(
        color=alt.Color("Outcome:N", scale=alt.Scale(domain=["Stayed", "Left"], range=[TEAL, CORAL]), legend=None),
    )
    labels = base.mark_text(align="left", dx=7, fontSize=13).encode(text=alt.Text("Customers:Q", format=","))
    return finish(bars + labels, 180)


def missing_chart(X: pd.DataFrame) -> alt.Chart:
    values = X.isna().sum().rename_axis("Feature").reset_index(name="Missing")
    values = values[values.Missing > 0]
    if values.empty:
        values = pd.DataFrame({"Feature": ["No missing feature values"], "Missing": [0]})
    chart = alt.Chart(values).mark_bar(color=CORAL, cornerRadiusEnd=4).encode(
        y=alt.Y("Feature:N", sort="-x", title=None), x=alt.X("Missing:Q", title=["Values to fill", "using training data"]),
        tooltip=["Feature", "Missing"],
    )
    return finish(chart, max(130, min(330, len(values) * 30)))


def feature_summary(X: pd.DataFrame, y: pd.Series, feature: str) -> pd.DataFrame:
    values = X[feature]
    if pd.api.types.is_numeric_dtype(values) and values.nunique() > 8:
        bins = pd.cut(values, bins=6, duplicates="drop")
        ordered = [str(interval) for interval in bins.cat.categories]
        group = bins.astype("string")
    else:
        group = values.astype("string")
        ordered = sorted(group.dropna().unique().tolist())
    frame = pd.DataFrame({"Group": group.fillna("Missing"), "Churn": y})
    result = frame.groupby("Group", observed=True, sort=True).Churn.agg(["mean", "size"]).reset_index()
    result = result.rename(columns={"mean": "Churn rate", "size": "Customers"})
    ordering = {label: i for i, label in enumerate(ordered)}
    result["Order"] = result.Group.map(ordering).fillna(len(ordered)).astype(int)
    return result.sort_values("Order").reset_index(drop=True)


def feature_chart(summary: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(summary).encode(
        y=alt.Y("Group:N", title=None, sort=alt.SortField(field="Order"), axis=alt.Axis(labelLimit=125)),
        x=alt.X("Churn rate:Q", title="Customers who left", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        tooltip=["Group", alt.Tooltip("Churn rate:Q", format=".1%"), "Customers"],
    )
    bars = base.mark_bar(color=TEAL, cornerRadiusEnd=4)
    inside = base.transform_filter(alt.datum["Churn rate"] >= 0.18).mark_text(align="right", dx=-6, color="white", fontSize=12).encode(text=alt.Text("Churn rate:Q", format=".1%"))
    outside = base.transform_filter(alt.datum["Churn rate"] < 0.18).mark_text(align="left", dx=6, color="#34445A", fontSize=12).encode(text=alt.Text("Churn rate:Q", format=".1%"))
    return finish(bars + inside + outside, max(200, min(400, len(summary) * 35)))


def comparison_chart(metrics: pd.DataFrame) -> alt.Chart:
    frame = metrics.melt(id_vars="Model", value_vars=["Test precision", "Test recall"], var_name="Metric", value_name="Score")
    frame["Metric"] = frame.Metric.str.replace("Test ", "", regex=False).str.capitalize()
    base = alt.Chart(frame).encode(
        x=alt.X("Metric:N", title=None, axis=alt.Axis(labelAngle=0)),
        xOffset="Model:N", y=alt.Y("Score:Q", title="Share of customers", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Model:N", scale=MODEL_COLORS),
        tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".1%")],
    )
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    labels = base.mark_text(dy=-8, fontSize=12, color="#34445A").encode(text=alt.Text("Score:Q", format=".0%"))
    return finish(bars + labels)


def pr_chart(truth: pd.Series, scores: dict[str, np.ndarray]) -> alt.Chart:
    frames = []
    for model, values in scores.items():
        precision, recall, _ = precision_recall_curve(truth, values)
        frames.append(pd.DataFrame({"Recall": recall, "Precision": precision, "Model": model, "Order": np.arange(len(recall))}))
    frame = pd.concat(frames, ignore_index=True)
    # Keep this bounded for larger uploads while preserving endpoints.
    if len(frame) > 4000:
        frame = pd.concat([g.iloc[np.unique(np.linspace(0, len(g) - 1, 2000).astype(int))] for _, g in frame.groupby("Model")])
    lines = alt.Chart(frame).mark_line(strokeWidth=2.5).encode(
        x=alt.X("Recall:Q", title=["Actual churners found", "(recall)"], axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Precision:Q", title=["Correct churn flags", "(precision)"], axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Model:N", scale=MODEL_COLORS), order="Order:Q",
        tooltip=["Model", alt.Tooltip("Recall:Q", format=".1%"), alt.Tooltip("Precision:Q", format=".1%")],
    )
    baseline = alt.Chart(pd.DataFrame({"Rate": [float(truth.mean())]})).mark_rule(color=GRAY, strokeDash=[5, 4]).encode(y="Rate:Q")
    return finish(lines + baseline)


def confusion_chart(matrix: np.ndarray) -> alt.Chart:
    labels = ["Stayed", "Left"]
    rows = [{"Actual": labels[i], "Predicted": labels[j], "Customers": int(matrix[i, j])} for i in range(2) for j in range(2)]
    base = alt.Chart(pd.DataFrame(rows)).encode(
        x=alt.X("Predicted:N", sort=labels, title="Model predicted", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Actual:N", sort=labels, title="Actually"), tooltip=["Actual", "Predicted", "Customers"],
    )
    squares = base.mark_rect(cornerRadius=5).encode(color=alt.Color("Customers:Q", scale=alt.Scale(range=["#EAF4F5", TEAL]), legend=None))
    text = base.mark_text(fontSize=24, fontWeight=600).encode(
        text=alt.Text("Customers:Q", format=","), color=alt.condition(alt.datum.Customers > matrix.max() * 0.55, alt.value("white"), alt.value("#23354A")),
    )
    return finish(squares + text, 250)


def distribution_chart(ranked: pd.DataFrame) -> alt.Chart:
    frame = ranked[["risk_score"]].assign(Risk=lambda d: d.risk_score.astype(float) * 100)
    chart = alt.Chart(frame).mark_bar(color=TEAL, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("Risk:Q", bin=alt.Bin(step=10, extent=[0, 100]), title="Estimated churn risk (%)", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("count():Q", title="Test customers"), tooltip=[alt.Tooltip("Risk:Q", bin=True), alt.Tooltip("count():Q", title="Customers")],
    )
    return finish(chart)


def capture_summary(ranked: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    truth = y.iloc[ranked.row_index.to_numpy()].to_numpy()
    if truth.sum() == 0:
        return pd.DataFrame({"Contact share": [0.0, 1.0], "Churners found": [0.0, 0.0]})
    return pd.DataFrame({"Contact share": np.arange(len(truth) + 1) / len(truth), "Churners found": np.r_[0, np.cumsum(truth) / truth.sum()]})


def capture_chart(summary: pd.DataFrame) -> alt.Chart:
    line = alt.Chart(summary).mark_line(color=TEAL, strokeWidth=3).encode(
        x=alt.X("Contact share:Q", title=["Highest-risk share", "of test customers"], axis=alt.Axis(format="%")),
        y=alt.Y("Churners found:Q", title="Actual churners included", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("Contact share:Q", format=".1%"), alt.Tooltip("Churners found:Q", format=".1%")],
    )
    random_line = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=GRAY, strokeDash=[5, 4]).encode(x="x:Q", y="y:Q")
    return finish(line + random_line)


def shap_chart(reasons: pd.DataFrame) -> alt.Chart:
    frame = reasons.head(5).copy()
    frame["Input"] = frame.Feature.map(feature_name)
    frame["Direction"] = np.where(frame.Contribution >= 0, "Pushes risk up", "Pushes risk down")
    chart = alt.Chart(frame).mark_bar(cornerRadius=4).encode(
        y=alt.Y("Input:N", sort=frame.Input.tolist(), title=None, axis=alt.Axis(labelLimit=125, labelFontSize=11)),
        x=alt.X("Contribution:Q", title="Influence on model score"),
        color=alt.Color("Direction:N", scale=alt.Scale(domain=["Pushes risk up", "Pushes risk down"], range=[CORAL, TEAL])),
        tooltip=["Feature", "Value", "Direction", alt.Tooltip("Contribution:Q", format=".3f")],
    )
    zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color=GRAY).encode(x="zero:Q")
    return finish(chart + zero, 320)


def campaign_curve(ranked: pd.DataFrame, cost: float, value: float, success: float) -> pd.DataFrame:
    ordered = np.sort(ranked.risk_score.to_numpy(dtype=float))[::-1]
    prefix = np.r_[0, np.cumsum(ordered)]
    shares = np.arange(101)
    counts = np.ceil(len(ordered) * shares / 100).astype(int)
    return pd.DataFrame({"Contact share": shares, "Net savings": prefix[counts] * success * value - counts * cost})


def savings_curve_chart(curve: pd.DataFrame, share: int) -> alt.Chart:
    base = alt.Chart(curve).encode(
        x=alt.X("Contact share:Q", title=["Highest-risk customers", "contacted (%)"], scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Net savings:Q", title=["Scenario net savings", "(chosen currency)"]),
        tooltip=[alt.Tooltip("Contact share:Q", title="Contacted (%)"), alt.Tooltip("Net savings:Q", format=",.2f")],
    )
    line = base.mark_line(color=TEAL, strokeWidth=3)
    marker = base.transform_filter(alt.datum["Contact share"] == share).mark_point(color=CORAL, size=100, filled=True)
    zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color=GRAY, strokeDash=[5, 4]).encode(y="zero:Q")
    return finish(line + marker + zero)


def money_chart(savings: dict) -> alt.Chart:
    frame = pd.DataFrame({"Item": ["Avoided loss", "Offer spend", "Net savings"], "Amount": [savings["avoided_loss"], -savings["offer_spend"], savings["net_savings"]]})
    chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("Item:N", sort=frame.Item.tolist(), title=None), x=alt.X("Amount:Q", title=["Amount in your", "chosen currency"]),
        color=alt.condition(alt.datum.Amount >= 0, alt.value(TEAL), alt.value(CORAL)),
        tooltip=["Item", alt.Tooltip("Amount:Q", format=",.2f")],
    )
    return finish(chart, 220)
