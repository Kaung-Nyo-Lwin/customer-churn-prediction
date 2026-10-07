import json

import pytest

from churn.app import create_app
from churn.data import FEATURES
from churn.profiles import PROFILES


@pytest.fixture
def app(artifact_path):
    return create_app(artifact_path)


def test_app_serves_layout_assets_and_health(app):
    client = app.server.test_client()
    for route in ["/", "/_dash-layout", "/_dash-dependencies", "/assets/style.css", "/healthz"]:
        with client.get(route) as response:
            assert response.status_code == 200
    assert client.get("/healthz").json == {"status": "ready"}


def post_prediction(app, profile):
    key = next(key for key in app.callback_map if "prediction-result.children" in key)
    return app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": key,
            "outputs": [
                {"id": "prediction-result", "property": "children"},
                {"id": "assessed-profile", "property": "data"},
            ],
            "inputs": [{"id": "predict-button", "property": "n_clicks", "value": 1}],
            "state": [
                {"id": name, "property": "value", "value": profile[name]} for name in FEATURES
            ],
            "changedPropIds": ["predict-button.n_clicks"],
        },
    )


def test_prediction_callback_returns_assessment_and_snapshot(app):
    profile = PROFILES["established"]
    response = post_prediction(app, profile)
    assert response.status_code == 200
    payload = response.json["response"]
    assert payload["assessed-profile"]["data"] == profile
    assert "risk-score" in json.dumps(payload["prediction-result"])


def test_invalid_callback_input_returns_error_instead_of_crashing(app):
    response = post_prediction(app, {**PROFILES["flexible"], "MonthlyCharges": None})
    assert response.status_code == 200
    payload = response.json["response"]
    assert payload["assessed-profile"]["data"] is None
    assert "Complete every field" in json.dumps(payload)


def test_numeric_strings_are_formatted_from_validated_input(app):
    response = post_prediction(app, {**PROFILES["flexible"], "MonthlyCharges": "94.5"})
    assert response.status_code == 200
    assert "$94.50" in json.dumps(response.json)


def test_changed_profile_marks_result_stale(app):
    key = next(key for key in app.callback_map if "stale-notice.children" in key)
    previous = PROFILES["flexible"]
    current = {**previous, "tenure": 9}
    response = app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": key,
            "outputs": [
                {"id": "stale-notice", "property": "children"},
                {"id": "prediction-result", "property": "className"},
            ],
            "inputs": [
                {"id": name, "property": "value", "value": current[name]} for name in FEATURES
            ]
            + [{"id": "assessed-profile", "property": "data", "value": previous}],
            "state": [],
            "changedPropIds": ["tenure.value"],
        },
    )
    assert response.status_code == 200
    assert response.json["response"]["prediction-result"]["className"] == "stale-result"
    assert "Profile changed" in response.json["response"]["stale-notice"]["children"]


def test_form_fields_are_state_and_do_not_trigger_inference(app):
    prediction = next(v for k, v in app.callback_map.items() if "prediction-result.children" in k)
    assert prediction["inputs"] == [{"id": "predict-button", "property": "n_clicks"}]
    assert {item["id"] for item in prediction["state"]} == set(FEATURES)


def test_missing_model_keeps_setup_page_available(tmp_path):
    app = create_app(tmp_path / "missing.joblib")
    client = app.server.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/_dash-layout").status_code == 200
    assert client.get("/healthz").status_code == 503
