"""Read CSV files and apply deterministic Telco schema checks."""
from io import BytesIO
from urllib.request import urlopen

import numpy as np
import pandas as pd

SAMPLE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
# Published Customer Churn Prediction 2020 fields. Charges retain their
# original meaning and are not treated as monthly or lifetime billing.
NUMERIC += [
    "account_length", "number_vmail_messages", "number_customer_service_calls",
    *[f"total_{period}_{measure}" for period in ("day", "eve", "night", "intl")
      for measure in ("minutes", "calls", "charge")],
]
CATEGORICAL += ["state", "area_code", "international_plan", "voice_mail_plan"]
COLUMN_NAMES = {name.casefold(): name for name in NUMERIC + CATEGORICAL + ["Churn", "customerID"]}
COLUMN_NAMES["id"] = "customerID"


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Resolve supported headers without changing original field meanings."""
    names = [str(name).strip() for name in frame.columns]
    names = [COLUMN_NAMES.get(name.casefold(), name) for name in names]
    if pd.Index(names).duplicated().any():
        raise ValueError("Column names must be unique after normalizing supported headers.")
    return frame.set_axis(names, axis="columns")


MAX_BYTES = 10 * 1024 * 1024


def read_csv_bytes(content: bytes) -> pd.DataFrame:
    """Read a small UTF-8, comma-separated CSV with friendly errors."""
    if not content or len(content) > MAX_BYTES:
        raise ValueError("Choose a nonempty CSV smaller than 10 MB.")
    try:
        frame = pd.read_csv(BytesIO(content), encoding="utf-8-sig")
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("Cannot read this file. Export a UTF-8, comma-separated CSV.") from exc
    if len(frame.columns) < 2:
        raise ValueError("Expected comma-separated columns. Check your CSV separator.")
    if len(frame) > 15000:
        raise ValueError("Use at most 15,000 rows for this free demo.")
    return frame


def synthetic_sample() -> pd.DataFrame:
    """Generate reproducible fictional data only for an offline demonstration."""
    rng = np.random.default_rng(42)
    n = 800
    tenure = rng.integers(0, 73, n)
    charges = rng.uniform(20, 120, n).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    log_odds = -1.5 + 1.2 * (contract == "Month-to-month") - 0.025 * tenure + 0.012 * charges
    churn = rng.random(n) < 1 / (1 + np.exp(-log_odds))
    return pd.DataFrame({
        "customerID": [f"DEMO-{i:04d}" for i in range(n)],
        "tenure": tenure, "MonthlyCharges": charges,
        "TotalCharges": (charges * tenure).round(2), "Contract": contract,
        "Churn": np.where(churn, "Yes", "No"),
    })


def load_sample() -> tuple[pd.DataFrame, str]:
    """Download IBM Telco data, or return an explicitly synthetic fallback."""
    try:
        with urlopen(SAMPLE_URL, timeout=12) as response:
            content = response.read(MAX_BYTES + 1)
        frame = read_csv_bytes(content)
        clean_data(frame)
        return frame, "IBM Telco Customer Churn (public CSV)"
    except Exception:
        # A remote outage must not prevent visitors from trying the interface.
        return synthetic_sample(), "SYNTHETIC offline demo — not the Telco dataset"


def clean_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Validate labels and formats without learning statistics from the data."""
    frame = normalize_columns(frame)
    if "Churn" not in frame:
        raise ValueError("Missing Churn column. Upload a historical CSV with known churn outcomes.")
    if not 100 <= len(frame) <= 15000:
        raise ValueError("Provide between 100 and 15,000 historical customer rows.")
    label = frame["Churn"].astype("string").str.strip().str.lower()
    y = label.map({"yes": 1, "no": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
    if y.isna().any():
        raise ValueError("Every Churn value must be Yes/No or 1/0; missing labels are not guessed.")
    y = y.astype(int)
    if len(y.value_counts()) != 2 or y.value_counts().min() < 20:
        raise ValueError("Include both churn classes, with at least 20 customers in each.")
    X, ids, notes = clean_features(frame)
    return X, y.reset_index(drop=True), ids, notes


def clean_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Apply the same input rules to historical and scoring rows."""
    frame = normalize_columns(frame)
    if not 1 <= len(frame) <= 15000:
        raise ValueError("Provide between 1 and 15,000 customer rows.")
    if "customerID" in frame:
        ids = frame["customerID"].astype("string").str.strip()
        if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
            raise ValueError("customerID must be present and unique for every customer.")
    else:
        ids = pd.Series([f"ROW-{i + 1}" for i in range(len(frame))], index=frame.index)
    features = [c for c in NUMERIC + CATEGORICAL if c in frame]
    if not features:
        raise ValueError("No supported features found. Use IBM Telco or Customer Churn Prediction 2020 columns.")
    X = frame[features].copy()
    notes = []
    ignored = [c for c in frame if c not in features + ["Churn", "customerID"]]
    if ignored:
        notes.append("Ignored unsupported columns: " + ", ".join(ignored))
    for col in features:
        text = X[col].astype("string").str.strip()
        text = text.mask(text.str.lower().isin(["", "na", "n/a", "null", "none", "nan"]))
        if col in NUMERIC:
            values = pd.to_numeric(text, errors="coerce").astype(float)
            invalid = ~np.isfinite(values) | (values < 0)
            if col == "SeniorCitizen":
                invalid |= ~values.isin([0, 1])
            X[col] = values.mask(invalid)
            count = int((text.notna() & X[col].isna()).sum())
            if count:
                notes.append(f"{col}: {count} invalid values treated as missing.")
        else:
            X[col] = text.str.lower().astype(object).where(text.notna(), np.nan)
            if X[col].nunique() > (51 if col == "state" else 50):
                raise ValueError(f"{col} has too many categories. Check for identifiers or free text.")
    notes.append("Missing feature values are filled using training data only.")
    return X.reset_index(drop=True), ids.reset_index(drop=True), notes

