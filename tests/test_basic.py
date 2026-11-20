"""Run with python -m pytest -q from the project root."""
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from src.business import estimate_savings
from src.data import clean_data, load_sample, read_csv_bytes, synthetic_sample
from src.explain import explain_customer
from src.model import rank_customers, train_models


@pytest.fixture(scope="module")
def trained() -> tuple:
    X, y, ids, _ = clean_data(synthetic_sample())
    return X, y, ids, train_models(X, y)


def test_missing_label() -> None:
    with pytest.raises(ValueError, match="Missing Churn"):
        clean_data(synthetic_sample().drop(columns="Churn"))


def test_bad_csv() -> None:
    with pytest.raises(ValueError, match="at least two columns"):
        read_csv_bytes(b"a\n1\n")


def test_inferred_inputs_from_semicolon_csv() -> None:
    frame = synthetic_sample().rename(
        columns={"Churn": "churned", "tenure": "Months enrolled", "Contract": "Plan family"}
    )[["churned", "Months enrolled", "Plan family"]]
    frame["churn_reason"] = "outcome proxy"
    raw = frame.to_csv(index=False, sep=";").encode()
    features, labels, _, _ = clean_data(read_csv_bytes(raw), infer=True)
    assert list(features) == ["Months enrolled", "Plan family"]
    assert pd.api.types.is_numeric_dtype(features["Months enrolled"])
    np.testing.assert_array_equal(labels, frame.churned.eq("Yes"))
    result = train_models(features, labels)
    assert result["metrics"]["Test AP"].notna().all()


def test_cleaning() -> None:
    raw = synthetic_sample()
    raw["TotalCharges"] = raw["TotalCharges"].astype(object)
    raw.loc[0, "TotalCharges"] = "bad"
    X, _, _, notes = clean_data(raw)
    assert pd.isna(X.loc[0, "TotalCharges"])
    assert any("invalid" in note for note in notes)


def test_offline_sample() -> None:
    with patch("src.data.urlopen", side_effect=OSError("offline")):
        frame, label = load_sample()
    assert "SYNTHETIC" in label
    clean_data(frame)


def test_split_and_rank(trained: tuple) -> None:
    X, _, ids, result = trained
    groups = [set(result[key]) for key in ["train_idx", "val_idx", "test_idx"]]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert len(set.union(*groups)) == len(X)
    ranked = rank_customers(result, X, ids)
    assert set(ranked.row_index) == groups[2]
    assert ranked.risk_score.between(0, 1).all()
    assert ranked.risk_score.is_monotonic_decreasing
    assert all(matrix.sum() == len(groups[2]) for matrix in result["matrices"].values())


def test_training_only_imputation(trained: tuple) -> None:
    X, _, _, result = trained
    prepare = result["models"]["Logistic regression"].named_steps["prepare"]
    actual = prepare.named_transformers_["numeric"].named_steps["fill"].statistics_
    expected = X.iloc[result["train_idx"]][["tenure", "MonthlyCharges", "TotalCharges"]].median().to_numpy()
    np.testing.assert_allclose(actual, expected)


def test_shap_both_models(trained: tuple) -> None:
    X, _, _, result = trained
    for model in result["models"].values():
        pair = X.iloc[result["test_idx"][:2]]
        background = X.iloc[result["train_idx"]]
        a = explain_customer(model, background, pair.iloc[[0]])
        b = explain_customer(model, background, pair.iloc[[1]])
        assert set(a.Feature) == set(X.columns)
        assert np.isfinite(a.Contribution).all()
        probability = model.predict_proba(pair)[:, 1]
        raw = np.log(probability / (1 - probability))
        np.testing.assert_allclose(a.Contribution.sum() - b.Contribution.sum(), raw[0] - raw[1], atol=1e-4)


def test_savings() -> None:
    ranked = pd.DataFrame({"risk_score": [0.8, 0.6, 0.2, 0.1]})
    result = estimate_savings(ranked, 50, 10, 100, 0.5)
    assert result["contacted"] == 2
    assert result["net_savings"] == pytest.approx(50)
    assert estimate_savings(ranked, 0, 10, 100, 0.5)["net_savings"] == 0
    with pytest.raises(ValueError):
        estimate_savings(ranked, 50, -1, 100, 0.5)


def test_streamlit_sample() -> None:
    from streamlit.testing.v1 import AppTest
    with patch("src.data.urlopen", side_effect=OSError("offline")):
        app = AppTest.from_file("app.py", default_timeout=90).run()
        app.button[0].click().run()
    assert not app.exception
    assert not app.error
    next(control for control in app.selectbox if control.label == "Explore the dashboard").set_value("Retention plan").run()
    assert not app.exception and not app.error
    assert any(metric.label == "Customers contacted" for metric in app.metric)


@pytest.mark.parametrize("change, message", [
    ("duplicate_ids", "unique"),
    ("bad_labels", "Every Churn"),
    ("one_class", "both churn classes"),
    ("no_features", "No supported"),
])
def test_invalid_uploads(change: str, message: str) -> None:
    raw = synthetic_sample()
    if change == "duplicate_ids":
        raw.loc[1, "customerID"] = raw.loc[0, "customerID"]
    elif change == "bad_labels":
        raw.loc[0, "Churn"] = "unknown"
    elif change == "one_class":
        raw["Churn"] = "No"
    elif change == "no_features":
        raw = raw[["customerID", "Churn"]]
    with pytest.raises(ValueError, match=message):
        clean_data(raw)


def test_unseen_categories_and_all_missing_numeric(trained: tuple) -> None:
    X, _, _, result = trained
    customer = X.iloc[[0]].copy()
    customer["Contract"] = "previously unseen contract"
    for model in result["models"].values():
        assert np.isfinite(model.predict_proba(customer)).all()
        reasons = explain_customer(model, X.iloc[result["train_idx"]], customer)
        assert np.isfinite(reasons.Contribution).all()
    raw = synthetic_sample()
    raw["MonthlyCharges"] = np.nan
    X, y, ids, _ = clean_data(raw)
    result = train_models(X, y)
    assert rank_customers(result, X, ids).risk_score.between(0, 1).all()


def test_upload_interface_and_cached_controls() -> None:
    from io import BytesIO
    from streamlit.testing.v1 import AppTest
    raw = synthetic_sample()
    raw.loc[0, "MonthlyCharges"] = 77.12345
    content = raw.to_csv(index=False).encode()
    app = AppTest.from_file("app.py", default_timeout=90).run()
    with patch("streamlit.file_uploader", return_value=BytesIO(content)):
        with patch("src.model.train_models", wraps=train_models) as training:
            app.radio[0].set_value("Upload CSV").run()
            assert not app.exception and not app.error
            assert training.call_count == 1
            next(control for control in app.selectbox if control.label == "Explore the dashboard").set_value("Retention plan").run()
            app.slider[0].set_value(0).run()
            assert not app.exception and not app.error
            assert training.call_count == 1
            assert any(m.label == "Scenario net savings" and m.value == "0.00" for m in app.metric)
            for cost in [25.0, 30.0]:
                app.number_input[0].set_value(cost).run()
                assert app.number_input[0].value == cost
                assert not app.error and not app.exception
            for share in [30, 40]:
                app.slider[0].set_value(share).run()
                assert app.slider[0].value == share
                assert not app.error and not app.exception
            next(control for control in app.selectbox if control.label == "Explore the dashboard").set_value("Models").run()
            assert not app.exception and not app.error
            next(control for control in app.selectbox if control.label == "Explore the dashboard").set_value("Individual risk").run()
            customer = next(control for control in app.selectbox if control.label == "Customer")
            customer.set_value(customer.options[-1]).run()
            assert not app.exception and not app.error
            assert training.call_count == 1
            next(control for control in app.selectbox if control.label == "Explore the dashboard").set_value("Retention plan").run()
            assert app.slider[0].value == 40
            assert app.number_input[0].value == 30.0
            app.slider[0].set_value(20).run()
            assert app.slider[0].value == 20
            assert training.call_count == 1


def test_invalid_upload_interface() -> None:
    from io import BytesIO
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file("app.py", default_timeout=90).run()
    content = synthetic_sample().drop(columns="Churn").to_csv(index=False).encode()
    with patch("streamlit.file_uploader", return_value=BytesIO(content)):
        app.radio[0].set_value("Upload CSV").run()
    assert not app.exception
    assert any(w.label == "Which column records whether the customer left?" for w in app.selectbox)


def test_zero_contribution_sentence() -> None:
    from src.explain import explanation_sentence
    reasons = pd.DataFrame({"Feature": ["tenure"], "Value": ["1"], "Contribution": [0.0]})
    assert "No input feature changes" in explanation_sentence(reasons)


def test_campaign_curve_matches_calculator() -> None:
    from src.charts import campaign_curve
    ranked = pd.DataFrame({"risk_score": [0.08, 0.91, 0.53, 0.67, 0.22, 0.31, 0.48]})
    curve = campaign_curve(ranked, 17, 430, 0.18)
    for share in [0, 1, 20, 35, 99, 100]:
        expected = estimate_savings(ranked, share, 17, 430, 0.18)
        assert curve.loc[curve["Contact share"] == share, "Net savings"].iloc[0] == pytest.approx(expected["net_savings"])


def test_capture_curve_uses_ranked_test_outcomes() -> None:
    from src.charts import capture_summary
    ranked = pd.DataFrame({"row_index": [3, 1, 4], "risk_score": [0.9, 0.6, 0.2]})
    truth = pd.Series([1, 0, 1, 1, 1])
    curve = capture_summary(ranked, truth)
    np.testing.assert_allclose(curve["Churners found"], [0, 0.5, 0.5, 1])
    assert curve["Contact share"].iloc[-1] == 1


def test_group_churn_rates_and_numeric_order() -> None:
    from src.charts import feature_summary
    X = pd.DataFrame({"tenure": np.arange(100)})
    y = pd.Series((np.arange(100) >= 50).astype(int))
    summary = feature_summary(X, y, "tenure")
    assert summary.Customers.sum() == len(X)
    assert (summary["Churn rate"] * summary.Customers).sum() == pytest.approx(y.sum())
    assert summary.Order.is_monotonic_increasing


def test_chart_specs_on_real_pipeline(trained: tuple) -> None:
    from src import charts
    X, y, ids, result = trained
    ranked = rank_customers(result, X, ids)
    scores = {name: model.predict_proba(X.iloc[result["test_idx"]])[:, 1] for name, model in result["models"].items()}
    reasons = explain_customer(result["models"][result["winner"]], X.iloc[result["train_idx"]], X.iloc[[int(ranked.iloc[0].row_index)]])
    savings = estimate_savings(ranked, 20, 20, 500, 0.15)
    figures = [
        charts.outcomes_chart(y), charts.missing_chart(X),
        charts.feature_chart(charts.feature_summary(X, y, "Contract")),
        charts.comparison_chart(result["metrics"]), charts.pr_chart(y.iloc[result["test_idx"]], scores),
        charts.confusion_chart(result["matrices"][result["winner"]]), charts.distribution_chart(ranked),
        charts.capture_chart(charts.capture_summary(ranked, y)), charts.shap_chart(reasons),
        charts.savings_curve_chart(charts.campaign_curve(ranked, 20, 500, 0.15), 20), charts.money_chart(savings),
    ]
    for figure in figures:
        figure.to_json()

