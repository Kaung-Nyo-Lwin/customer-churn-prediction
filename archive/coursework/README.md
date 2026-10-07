# Original coursework

This directory preserves the class project by **Kaung Nyo Lwin** and
**Patsachon Pattakulpong** at the Asian Institute of Technology. The files retain
their original contents and authorship.

- `CP_project.ipynb`: exploratory analysis and initial modeling.
- `CP_project_complete.ipynb`: model comparisons and serialization experiments.
- `test.ipynb`: an experimental notebook.
- `st125066_CP_project_report.pdf`: the submitted report.
- `st125066_CP_project_Telecom_Churn_Presentation.pdf`: the presentation.
- Python scripts: the original preprocessing, training, evaluation, search, and app.
- `customer_churn.model`: the original serialized model, retained for provenance.
- CSVs and extensionless summaries: historical experiment outputs.
- `images/`: original coursework plots.
- `original-readme.md`: the original project description and historical claims.

These are historical artifacts, not supported entry points for the refreshed app.
Their scripts assume the original repository layout, and their notebooks contain
older dependencies and intermediate state. Use the root `model_train.py` and
`main.py` for the current workflow.

The original outputs disagree in places and use a different evaluation protocol.
In particular, the original false-negative calculation also counted some incorrect
value-tier predictions as missed churn. The refreshed benchmark evaluates binary
churn separately and documents the historical-charge proxy. Current results are
in [reports/benchmark.md](../../reports/benchmark.md).
