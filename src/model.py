"""Train two candidates, select on validation data, evaluate on untouched tests."""
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data import NUMERIC

SEED = 42


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Fill missing data, scale numbers and encode categories inside the pipeline."""
    numeric = [c for c in X if c in NUMERIC]
    categorical = [c for c in X if c not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("fill", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("category", Pipeline([
            ("fill", SimpleImputer(strategy="constant", fill_value="missing", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], verbose_feature_names_out=False)


def train_models(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Use 60% training, 20% validation and 20% testing, preserving class ratios."""
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=0.2, stratify=y, random_state=SEED,
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.25, stratify=y.iloc[train_idx], random_state=SEED,
    )
    estimators = {
        "Logistic regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=180, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=2,
            objective="binary:logistic", eval_metric="logloss",
            random_state=SEED, n_jobs=2, tree_method="hist",
        ),
    }
    models, validation, rows, matrices = {}, {}, [], {}
    for name, estimator in estimators.items():
        pipeline = Pipeline([("prepare", make_preprocessor(X)), ("model", estimator)])
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        models[name] = pipeline
        validation[name] = float(average_precision_score(
            y.iloc[val_idx], pipeline.predict_proba(X.iloc[val_idx])[:, 1],
        ))
    # Selection uses validation labels only. The test group never chooses a model.
    winner = max(validation, key=validation.get)
    for name, pipeline in models.items():
        probability = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
        predicted = (probability >= 0.5).astype(int)
        truth = y.iloc[test_idx]
        rows.append({
            "Model": name, "Validation AP": validation[name],
            "Test precision": precision_score(truth, predicted, zero_division=0),
            "Test recall": recall_score(truth, predicted, zero_division=0),
            "Test AP": average_precision_score(truth, probability),
            "Test Brier": brier_score_loss(truth, probability),
        })
        matrices[name] = confusion_matrix(truth, predicted, labels=[0, 1])
    return {
        "models": models, "winner": winner, "metrics": pd.DataFrame(rows),
        "matrices": matrices, "train_idx": train_idx, "val_idx": val_idx,
        "test_idx": test_idx,
    }


def rank_customers(result: dict[str, Any], X: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    """Rank held-out customers so every displayed score is out of sample."""
    idx = result["test_idx"]
    pipeline = result["models"][result["winner"]]
    ranked = pd.DataFrame({
        "row_index": idx, "customerID": ids.iloc[idx].to_numpy(),
        "risk_score": pipeline.predict_proba(X.iloc[idx])[:, 1],
    })
    return ranked.sort_values("risk_score", ascending=False, kind="stable").reset_index(drop=True)
