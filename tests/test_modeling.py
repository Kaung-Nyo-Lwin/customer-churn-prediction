from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.base import clone

from churn.data import FEATURES
from churn.evaluation import choose_threshold, evaluate
from churn.modeling import ValueAwareClassifier, load_artifact, value_labels


def test_value_cutoff_is_learned_from_fit_rows_only(data, fitted_model):
    expected = data.sample(300, random_state=42)["TotalCharges"].median()
    assert fitted_model.value_threshold_ == expected
    assert not hasattr(clone(fitted_model), "value_threshold_")


def test_probabilities_sum_to_one_and_roundtrip(data, fitted_model, artifact_path):
    X = data.iloc[:20][FEATURES]
    probabilities = fitted_model.predict_proba(X)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    reloaded = load_artifact(artifact_path)
    np.testing.assert_allclose(probabilities, reloaded["model"].predict_proba(X))


def test_wrong_value_tier_is_not_a_missed_churn():
    model = ValueAwareClassifier()
    model.pipeline_ = SimpleNamespace(
        classes_=np.array(["High_Churn", "High_NoChurn", "Low_Churn", "Low_NoChurn"]),
        predict_proba=lambda X: np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.2, 0.1, 0.6]]),
    )
    # Even if the actual churner is low value, predicting High_Churn catches churn.
    probability = model.predict_proba(None)[:, 1]
    np.testing.assert_allclose(probability, [0.8, 0.2])
    result = evaluate([1, 0], probability, [200, 300], 0.5)
    assert result["false_negatives"] == 0
    assert result["missed_churn_historical_charges"] == 0


def test_false_negatives_and_charge_proxy_are_counted_separately():
    result = evaluate([1, 1, 0, 0], [0.1, 0.9, 0.8, 0.2], [100, 900, 5000, 20], 0.5)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["false_negatives"] == 1
    assert result["missed_churn_historical_charges"] == 100
    assert result["missed_charge_share"] == 0.1


def test_no_recorded_churn_charges_does_not_divide_by_zero():
    result = evaluate([1, 0], [0.1, 0.2], [np.nan, 50], 0.5)
    assert result["missed_charge_share"] == 0


def test_threshold_ties_choose_fewer_alerts():
    threshold = choose_threshold([0, 0, 1, 1], np.array([0.1, 0.2, 0.6, 0.8]))
    assert threshold == pytest.approx(0.6)


def test_value_labels_handle_cutoff_and_missing_charges(data):
    frame = data.iloc[:3][FEATURES].copy()
    frame["TotalCharges"] = [100, 99, np.nan]
    assert value_labels(frame, [1, 0, 1], 100).tolist() == [
        "High_Churn",
        "Low_NoChurn",
        "Low_Churn",
    ]


def test_missing_artifact_has_actionable_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="python model_train.py"):
        load_artifact(tmp_path / "missing.joblib")
