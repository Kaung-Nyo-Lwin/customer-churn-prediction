import json

import joblib
import pytest

from churn.data import FEATURES, REPORT_PATH, load_data
from churn.modeling import ValueAwareClassifier


@pytest.fixture(scope="session")
def data():
    return load_data()


@pytest.fixture(scope="session")
def fitted_model(data):
    sample = data.sample(300, random_state=42)
    return ValueAwareClassifier().fit(sample[FEATURES], sample["Churn"].eq("Yes").astype(int))


@pytest.fixture(scope="session")
def artifact_path(tmp_path_factory, fitted_model):
    # The saved benchmark supplies the presentation schema; this fixture model
    # is independently fitted so tests work without a prebuilt local artifact.
    report = json.loads(REPORT_PATH.read_text())
    path = tmp_path_factory.mktemp("model") / "test.joblib"
    joblib.dump(
        {
            "schema_version": 1,
            "features": FEATURES,
            "model": fitted_model,
            "threshold": 0.14,
            "report": report,
        },
        path,
    )
    return path
