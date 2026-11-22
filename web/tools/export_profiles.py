"""Fit and export the built-in churn models for browser inference."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path, help="Original customer-churn-tool checkout")
parser.add_argument("telco", type=Path)
parser.add_argument("churn2020", type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))

from src.data import clean_data, read_csv_bytes
from src.model import SEED, make_preprocessor, train_models


def fit_telecom2020(X, y, train):
    """Use the configuration selected on the Churn Prediction 2020 development folds."""
    pipeline = Pipeline([
        ("prepare", make_preprocessor(X)),
        ("model", XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.045,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=4,
            min_child_weight=1, objective="binary:logistic", eval_metric="logloss",
            random_state=SEED, n_jobs=1, tree_method="hist",
        )),
    ])
    return pipeline.fit(X.iloc[train], y.iloc[train])


def export(name, path):
    data = read_csv_bytes(path.read_bytes())
    X, y, ids, _ = clean_data(data)
    result = train_models(X, y)
    winner = result["winner"]
    test = result["test_idx"]
    models = result["models"]
    if name == "telecom2020":
        winner = "XGBoost"
        models[winner] = fit_telecom2020(X, y, result["train_idx"])
    pipeline = models[winner]
    prepare, model = pipeline["prepare"], pipeline["model"]
    numeric = prepare.transformers_[0][2]
    categorical = prepare.transformers_[1][2]
    medians = prepare.named_transformers_["numeric"]["fill"].statistics_
    scale = prepare.named_transformers_["numeric"]["scale"]
    encoder = prepare.named_transformers_["category"]["encode"]
    columns = [
        {"name": c, "median": float(m), "mean": float(a), "scale": float(s)}
        for c, m, a, s in zip(numeric, medians, scale.mean_, scale.scale_)
    ] + [
        {"name": c, "categories": list(categories),
         "reference": str(X.iloc[result["train_idx"]][c].fillna("missing").mode().iloc[0])}
        for c, categories in zip(categorical, encoder.categories_)
    ]
    if winner == "XGBoost":
        booster = model.get_booster()
        config = json.loads(booster.save_config())
        base = float(config["learner"]["learner_model_param"]["base_score"])

        def tree(node):
            if "leaf" in node:
                return node["leaf"]
            children = {child["nodeid"]: child for child in node["children"]}
            return [int(node["split"][1:]), node["split_condition"],
                    tree(children[node["yes"]]), tree(children[node["no"]])]

        estimator = {"kind": "boosted", "base": math.log(base / (1 - base)),
                     "trees": [tree(json.loads(t)) for t in booster.get_dump(dump_format="json")]}
    else:
        estimator = {"kind": "logistic", "base": float(model.intercept_[0]),
                     "weights": model.coef_[0].tolist()}
    metrics = []
    for label, candidate in models.items():
        predicted = candidate.predict_proba(X.iloc[test])[:, 1] >= 0.5
        truth = y.iloc[test]
        metrics.append({
            "name": label,
            "precision": precision_score(truth, predicted, zero_division=0),
            "recall": recall_score(truth, predicted, zero_division=0),
            "matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
        })
    out = {
        "winner": winner,
        "columns": columns,
        "estimator": estimator,
        "test": test.tolist(),
        "metrics": metrics,
        "testScores": pipeline.predict_proba(X.iloc[test])[:, 1].tolist(),
    }
    Path(f"dist/data/{name}.json").write_text(json.dumps(out, separators=(",", ":"), allow_nan=False))
    print(name, len(X), int(np.sum(y)), winner)


export("ibm", args.telco)
export("telecom2020", args.churn2020)
# The same public demo, without columns unrelated to the model or views.
sample = read_csv_bytes(args.telco.read_bytes())
X, y, ids, _ = clean_data(sample)
X.assign(customerID=ids, Churn=y).to_csv("dist/data/ibm.csv", index=False)
