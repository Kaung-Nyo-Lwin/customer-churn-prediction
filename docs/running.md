# Running and developing Retain

## Local setup

Use Python 3.10–3.12 from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python model_train.py
python main.py
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. The local
app opens at <http://127.0.0.1:8080>. Training builds the ignored local model
artifact; a fresh clone must run this step before making predictions.

The default example is assessed on startup. Choose another profile or edit the
inputs, then select **Run assessment**. The previous result is marked stale after
an edit. Internet add-ons and multiple-line inputs are excluded automatically
when the corresponding parent service is off.

The demo accepts whole-number tenure from 0–72 months, monthly charges from
$0–$200, and total charges from $0–$15,000. These are demo bounds rather than
business rules. Profiles are illustrative and inputs are not persisted by the app.

## Reproducing a run

```bash
python model_train.py --seed 42
```

Optional `--data`, `--artifact`, and `--report` arguments accept file paths.
For example, keep an experiment separate from the published benchmark:

```bash
python model_train.py --seed 7 \
  --artifact artifacts/experiment.joblib \
  --report /tmp/experiment.json
```

The JSON and Markdown reports are written together. The demo uses the report
embedded in its model artifact, so the displayed metrics match the loaded model.
It continues to use `artifacts/churn.joblib` unless a different path is supplied
to `churn.app.create_app` from Python.

Use the same dependency versions for training and serving. If dependencies or
the input schema change, retrain the artifact. Load only locally generated,
trusted joblib files.

## Development checks

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

The tests fit a small independent model, so they run without a prebuilt artifact.
They cover input validation, preprocessing boundaries, probability aggregation,
the charge diagnostic, serialization, and Dash callbacks. GitHub Actions also
reproduces the full training workflow and checks startup on Python 3.10 and 3.12.
Archived coursework is excluded from current lint checks.

## Serving with Gunicorn

On a Unix host, train the artifact first, then use the included WSGI entry point:

```bash
gunicorn main:server --bind 127.0.0.1:8080 --workers 2 --timeout 120
```

Put the WSGI service behind your hosting provider's HTTPS proxy when publishing.
For a local network demo, the built-in development server supports
`python main.py --host 0.0.0.0 --port 8080`. Debug mode is opt-in with `--debug`.

`GET /healthz` returns HTTP 200 when the model is loaded. With a missing artifact,
the setup page remains available, prediction is disabled, and health returns 503.
