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
    with pytest.raises(ValueError, match="comma-separated"):
        read_csv_bytes(b"a;b\n1;2\n")


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
            app.slider[0].set_value(0).run()
            assert not app.exception and not app.error
            assert training.call_count == 1
            assert any(m.label == "Scenario net savings" and m.value == "0.00" for m in app.metric)
            app.selectbox[1].set_value(app.selectbox[1].options[-1]).run()
            assert not app.exception and not app.error
            assert training.call_count == 1


def test_invalid_upload_interface() -> None:
    from io import BytesIO
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file("app.py", default_timeout=90).run()
    content = synthetic_sample().drop(columns="Churn").to_csv(index=False).encode()
    with patch("streamlit.file_uploader", return_value=BytesIO(content)):
        app.radio[0].set_value("Upload CSV").run()
    assert not app.exception
    assert any("Missing Churn" in error.value for error in app.error)


def test_zero_contribution_sentence() -> None:
    from src.explain import explanation_sentence
    reasons = pd.DataFrame({"Feature": ["tenure"], "Value": ["1"], "Contribution": [0.0]})
    assert "No input feature changes" in explanation_sentence(reasons)
