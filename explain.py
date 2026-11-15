"""Explain either selected model using SHAP in log-odds units."""
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def explain_customer(pipeline: Pipeline, background: pd.DataFrame, customer: pd.DataFrame) -> pd.DataFrame:
    """Aggregate encoded SHAP contributions back to the original input columns."""
    prepare = pipeline.named_steps["prepare"]
    model = pipeline.named_steps["model"]
    transformed = prepare.transform(customer)
    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, prepare.transform(background.iloc[:100]))
    else:
        explainer = shap.TreeExplainer(model, model_output="raw")
    values = np.asarray(explainer.shap_values(transformed))[0]
    original_columns = []
    for name, transformer, columns in prepare.transformers_:
        if name == "numeric":
            original_columns.extend(columns)
        elif name == "category":
            encoder = transformer.named_steps["encode"]
            for column, categories in zip(columns, encoder.categories_):
                original_columns.extend([column] * len(categories))
    result = pd.DataFrame({"Feature": original_columns, "Contribution": values})
    result = result.groupby("Feature", sort=False, as_index=False)["Contribution"].sum()
    result["Value"] = result["Feature"].map(lambda col: str(customer.iloc[0][col]))
    return result.iloc[np.argsort(-result["Contribution"].abs().to_numpy())].reset_index(drop=True)


def explanation_sentence(reasons: pd.DataFrame) -> str:
    """Describe the strongest model contribution without claiming causation."""
    top = reasons.iloc[0]
    if np.isclose(reasons["Contribution"].abs().max(), 0):
        return "No input feature changes this prediction relative to the explanation reference."
    direction = "higher" if top["Contribution"] > 0 else "lower"
    return (
        f"{top['Feature']} = {top['Value']} makes the strongest contribution toward "
        f"{direction} predicted risk relative to this explanation's reference. "
        "This describes a model pattern, not a proven cause."
    )
