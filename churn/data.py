"""Shared data contract and preprocessing for training and inference."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Datasets/Final/WA_Fn-UseC_-Telco-Customer-Churn.csv"
ARTIFACT_PATH = ROOT / "artifacts/churn.joblib"
REPORT_PATH = ROOT / "reports/benchmark.json"
YES_NO = ("No", "Yes")
INTERNET_ADDONS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)
CATEGORIES = {
    "gender": ("Female", "Male"),
    "SeniorCitizen": (0, 1),
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ("No", "Yes", "No phone service"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    **{name: ("No", "Yes", "No internet service") for name in INTERNET_ADDONS},
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": YES_NO,
    "PaymentMethod": (
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ),
}
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
FEATURES = [*CATEGORIES, *NUMERIC_FEATURES]


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read the selected IBM sample; blank charges stay missing until model fitting."""
    frame = pd.read_csv(path)
    required = {"customerID", "Churn", *FEATURES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    if frame["customerID"].isna().any() or frame["customerID"].duplicated().any():
        raise ValueError("Dataset must have a unique customerID for every row.")
    if not frame["Churn"].isin(YES_NO).all():
        raise ValueError("Churn labels must be 'No' or 'Yes'.")
    for name, choices in CATEGORIES.items():
        if not frame[name].isin(choices).all():
            raise ValueError(f"Invalid or missing category in {name}.")
    for name in NUMERIC_FEATURES:
        raw = frame[name]
        blank = raw.isna() | raw.astype(str).str.strip().eq("")
        values = pd.to_numeric(raw, errors="coerce")
        if (values.isna() & ~blank).any() or np.isinf(values).any():
            raise ValueError(f"{name} contains invalid numbers.")
        if values.lt(0).any() or (name != "TotalCharges" and values.isna().any()):
            raise ValueError(f"{name} contains missing or negative values.")
        frame[name] = values
    return frame


def make_preprocessor() -> ColumnTransformer:
    """Fit imputation, scaling, and nominal encoding inside each training fold."""
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                list(CATEGORIES),
            ),
        ],
        remainder="drop",
    )


def customer_frame(values: dict) -> pd.DataFrame:
    """Validate a complete demo profile, including service dependencies."""
    missing = [name for name in FEATURES if values.get(name) is None or values.get(name) == ""]
    if missing:
        raise ValueError("Complete every field before running an assessment.")
    clean = {name: values[name] for name in FEATURES}
    for name, choices in CATEGORIES.items():
        if clean[name] not in choices:
            raise ValueError(f"Choose a valid value for {name}.")
    for name in NUMERIC_FEATURES:
        try:
            value = float(clean[name])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name} must be a number.") from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError("Tenure and charges must be finite, non-negative numbers.")
        clean[name] = value
    if not clean["tenure"].is_integer() or clean["tenure"] > 72:
        raise ValueError("Use a whole-number tenure between 0 and 72 months for this demo.")
    if clean["MonthlyCharges"] > 200 or clean["TotalCharges"] > 15000:
        raise ValueError("Use monthly charges up to $200 and total charges up to $15,000.")
    if clean["PhoneService"] == "No":
        clean["MultipleLines"] = "No phone service"
    elif clean["MultipleLines"] == "No phone service":
        raise ValueError("Choose Yes or No for multiple lines when phone service is active.")
    if clean["InternetService"] == "No":
        clean.update({name: "No internet service" for name in INTERNET_ADDONS})
    elif any(clean[name] == "No internet service" for name in INTERNET_ADDONS):
        raise ValueError("Choose Yes or No for add-ons when internet service is active.")
    return pd.DataFrame([clean], columns=FEATURES)
