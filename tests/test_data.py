import numpy as np
import pytest

from churn.data import FEATURES, INTERNET_ADDONS, customer_frame, load_data, make_preprocessor
from churn.profiles import PROFILES


def test_selected_dataset_contract(data):
    assert data.shape == (7043, 21)
    assert data["TotalCharges"].isna().sum() == 11
    assert len(FEATURES) == 19
    assert "tenure" in FEATURES
    assert "customerID" not in FEATURES
    assert "Churn" not in FEATURES


def test_preprocessing_does_not_learn_from_held_out_rows(data):
    train = data.iloc[:100][FEATURES].copy()
    test = data.iloc[100:105][FEATURES].copy()
    test["TotalCharges"] = 1e12
    pipeline = make_preprocessor().fit(train)
    median_before = pipeline.named_transformers_["numeric"]["impute"].statistics_.copy()
    pipeline.transform(test)
    np.testing.assert_array_equal(
        median_before,
        pipeline.named_transformers_["numeric"]["impute"].statistics_,
    )
    assert median_before[2] == train["TotalCharges"].median()


def test_preprocessor_handles_missing_numeric_and_unknown_category(data):
    train = data.iloc[:100][FEATURES].copy()
    future = train.iloc[:1].copy()
    future["TotalCharges"] = np.nan
    future["PaymentMethod"] = "New payment service"
    transformed = make_preprocessor().fit(train).transform(future)
    assert np.isfinite(transformed).all()


@pytest.mark.parametrize("profile", PROFILES.values())
def test_examples_match_model_contract(profile):
    assert customer_frame(profile).columns.tolist() == FEATURES


@pytest.mark.parametrize(
    "field,value",
    [
        ("MonthlyCharges", None),
        ("MonthlyCharges", -1),
        ("TotalCharges", np.inf),
        ("tenure", 2.5),
        ("tenure", 100),
        ("TotalCharges", "oops"),
        ("Contract", "unknown"),
    ],
)
def test_invalid_inputs_produce_useful_errors(field, value):
    with pytest.raises(ValueError):
        customer_frame({**PROFILES["flexible"], field: value})


def test_zero_charges_are_valid_for_a_new_account():
    frame = customer_frame({**PROFILES["new"], "tenure": 0, "TotalCharges": 0})
    assert frame["TotalCharges"].iloc[0] == 0


def test_unavailable_services_are_normalized():
    frame = customer_frame({**PROFILES["flexible"], "PhoneService": "No", "InternetService": "No"})
    assert frame["MultipleLines"].iloc[0] == "No phone service"
    assert (frame[list(INTERNET_ADDONS)] == "No internet service").all().all()


def test_invalid_dataset_is_rejected(data, tmp_path):
    frame = data.iloc[:10].copy()
    frame.loc[frame.index[0], "Churn"] = "Unknown"
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Churn labels"):
        load_data(path)
