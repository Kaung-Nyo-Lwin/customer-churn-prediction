"""Compare a value-aware model with binary baselines on one fixed split."""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from churn.data import ARTIFACT_PATH, DATA_PATH, FEATURES, REPORT_PATH, load_data, make_preprocessor
from churn.evaluation import choose_threshold, evaluate
from churn.reporting import markdown_report

SEED = 42
SEGMENTS = ["High_Churn", "High_NoChurn", "Low_Churn", "Low_NoChurn"]


def value_labels(X: pd.DataFrame, y, threshold: float) -> np.ndarray:
    # Missing charges are grouped as low observed value (11 new accounts in this sample).
    tier = np.where(X["TotalCharges"].fillna(0) >= threshold, "High", "Low")
    outcome = np.where(np.asarray(y) == 1, "Churn", "NoChurn")
    return np.char.add(np.char.add(tier, "_"), outcome)


def boosting(seed: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=seed,
    )


class ValueAwareClassifier(ClassifierMixin, BaseEstimator):
    """Learn four classes while exposing binary churn probabilities to sklearn metrics.

    The value cutoff is learned inside fit, including during cross-validation.
    Churn probability sums High_Churn and Low_Churn, so confusing their value
    tiers cannot create a false negative in the churn evaluation.
    """

    def __init__(self, random_state: int = SEED):
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y):
        self.value_threshold_ = float(X["TotalCharges"].median())
        labels = value_labels(X, y, self.value_threshold_)
        self.pipeline_ = Pipeline(
            [
                ("preprocess", make_preprocessor()),
                ("model", boosting(self.random_state)),
            ]
        )
        self.pipeline_.fit(X, labels)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "pipeline_")
        probabilities = self.pipeline_.predict_proba(X)
        mask = np.array([label.endswith("_Churn") for label in self.pipeline_.classes_])
        churn = probabilities[:, mask].sum(axis=1)
        return np.column_stack([1 - churn, churn])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_segments(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "pipeline_")
        return self.pipeline_.predict(X)


def run_training(
    data_path: str | Path = DATA_PATH,
    artifact_path: str | Path = ARTIFACT_PATH,
    report_path: str | Path = REPORT_PATH,
    seed: int = SEED,
) -> dict:
    frame = load_data(data_path)
    X = frame[FEATURES]
    y = frame["Churn"].eq("Yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    candidates = {
        "Dummy (prior)": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(max_iter=2000, random_state=seed),
        "Binary gradient boosting": boosting(seed),
        "Value-aware gradient boosting": ValueAwareClassifier(random_state=seed),
    }
    results, fitted = {}, {}
    for name, classifier in candidates.items():
        print(f"Evaluating {name}...", flush=True)
        estimator = (
            classifier
            if isinstance(classifier, ValueAwareClassifier)
            else Pipeline(
                [
                    ("preprocess", make_preprocessor()),
                    ("model", classifier),
                ]
            )
        )
        oof = cross_val_predict(
            estimator,
            X_train,
            y_train,
            cv=folds,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        threshold = choose_threshold(y_train, oof)
        estimator.fit(X_train, y_train)
        probabilities = estimator.predict_proba(X_test)[:, 1]
        results[name] = evaluate(y_test, probabilities, X_test["TotalCharges"], threshold)
        results[name]["oof_average_precision"] = float(average_precision_score(y_train, oof))
        fitted[name] = estimator

    # Demonstrate the original research question, without selecting on holdout results.
    deployed_name = "Value-aware gradient boosting"
    model = fitted[deployed_name]
    true_segments = value_labels(X_test, y_test, model.value_threshold_)
    predicted_segments = model.predict_segments(X_test)
    report = {
        "schema_version": 1,
        "dataset": {
            "file": Path(data_path).name,
            "sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
            "rows": len(frame),
            "features": len(FEATURES),
            "churn_count": int(y.sum()),
            "churn_rate": float(y.mean()),
            "missing_total_charges": int(frame["TotalCharges"].isna().sum()),
        },
        "split": {
            "seed": seed,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "test_churn_count": int(y_test.sum()),
            "test_fraction": 0.2,
            "cross_validation_folds": 5,
            "train_ids_sha256": hashlib.sha256(
                "\n".join(frame.loc[X_train.index, "customerID"]).encode()
            ).hexdigest(),
            "test_ids_sha256": hashlib.sha256(
                "\n".join(frame.loc[X_test.index, "customerID"]).encode()
            ).hexdigest(),
        },
        "environment": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in [
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "joblib",
                ]
            },
        },
        "demo_model": deployed_name,
        "value_threshold": model.value_threshold_,
        "threshold_policy": "Maximize F2 over 0.10–0.80 on five-fold OOF training scores",
        "models": results,
        "four_class": {
            "labels": SEGMENTS,
            "accuracy": float(accuracy_score(true_segments, predicted_segments)),
            "confusion_matrix": confusion_matrix(
                true_segments,
                predicted_segments,
                labels=SEGMENTS,
            ).tolist(),
        },
    }
    artifact = {
        "schema_version": 1,
        "model": model,
        "threshold": results[deployed_name]["threshold"],
        "features": FEATURES,
        "report": report,
    }
    artifact_path, report_path = Path(artifact_path), Path(report_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path, compress=3)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report_path.with_suffix(".md").write_text(markdown_report(report))
    return report


def load_artifact(path: str | Path = ARTIFACT_PATH) -> dict:
    """Load an artifact produced locally by model_train.py."""
    if not Path(path).is_file():
        raise FileNotFoundError("Model not found. Run `python model_train.py` first.")
    artifact = joblib.load(path)
    if artifact.get("schema_version") != 1 or artifact.get("features") != FEATURES:
        raise ValueError("Model schema has changed. Run `python model_train.py` again.")
    return artifact
