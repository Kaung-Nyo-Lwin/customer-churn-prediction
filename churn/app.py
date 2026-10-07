"""A small, inspectable retention workbench built with Dash."""

import logging
from pathlib import Path

from dash import Dash, Input, Output, State, dcc, html

from churn.data import ARTIFACT_PATH, CATEGORIES, FEATURES, INTERNET_ADDONS, ROOT, customer_frame
from churn.modeling import load_artifact
from churn.profiles import PROFILES

LOGGER = logging.getLogger(__name__)
REPOSITORY = "https://github.com/Kaung-Nyo-Lwin/customer-churn-prediction"
LABELS = {
    "gender": "Gender",
    "SeniorCitizen": "Senior citizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
    "tenure": "Tenure · months",
    "MonthlyCharges": "Monthly charges · $",
    "TotalCharges": "Total charges · $",
}


def field(name: str):
    default = PROFILES["flexible"][name]
    if name in CATEGORIES:
        choices = CATEGORIES[name]
        if name in INTERNET_ADDONS or name == "MultipleLines":
            choices = ("No", "Yes")
        options = [
            {
                "label": ("Yes" if value else "No") if name == "SeniorCitizen" else value,
                "value": value,
            }
            for value in choices
        ]
        control = dcc.Dropdown(
            id=name,
            options=options,
            value=default,
            clearable=False,
            searchable=False,
            className="profile-select",
        )
    else:
        control = dcc.Input(
            id=name,
            type="number",
            value=default,
            min=0,
            max={"tenure": 72, "MonthlyCharges": 200, "TotalCharges": 15000}[name],
            step=1 if name == "tenure" else 0.01,
            className="number-input",
        )
    return html.Div([html.Label(LABELS[name], htmlFor=name), control], className="field")


def status_panel(title: str, detail: str, error: bool = False):
    return html.Div(
        [html.Span("ASSESSMENT", className="eyebrow"), html.H3(title), html.P(detail)],
        className="result-empty" + (" result-error" if error else ""),
        role="alert" if error else "status",
    )


def assessment(artifact: dict, values: dict):
    frame = customer_frame(values)
    probability = float(artifact["model"].predict_proba(frame)[0, 1])
    threshold = artifact["threshold"]
    higher_value = float(frame["TotalCharges"].iloc[0]) >= artifact["report"]["value_threshold"]
    flagged = probability >= threshold
    priority = (
        "Priority review"
        if higher_value and flagged
        else "Review recommended"
        if flagged
        else "Routine monitoring"
    )
    action = (
        "Start a personal check-in about service quality and plan fit. This account has higher "
        "historical charges and is above the review threshold."
        if flagged and higher_value
        else "Check the service experience and plan fit. This account is above the review "
        "threshold; a brief check-in is a useful next step."
        if flagged
        else "Continue regular service check-ins. The score is below the review threshold; "
        "reassess when the account or service experience changes."
    )
    return html.Div(
        [
            html.Div(
                [
                    html.Span("ACCOUNT ASSESSMENT", className="eyebrow"),
                    html.Span("●  Model ready", className="live-tag"),
                ],
                className="result-topline",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Strong(f"{probability:.0%}", id="risk-score"),
                            html.Span("churn score"),
                        ],
                        className="score-center",
                    )
                ],
                className="score-ring",
                style={"--score": f"{probability * 100:.2f}%"},
            ),
            html.Div(priority, className="priority-pill" + (" flagged" if flagged else "")),
            html.P(
                f"{('Above' if flagged else 'Below')} the {threshold:.0%} review threshold",
                className="threshold-caption",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Observed value"),
                            html.Strong("Higher" if higher_value else "Lower"),
                        ],
                        title=(
                            "Higher observed value starts at "
                            f"${artifact['report']['value_threshold']:,.3f} in historical charges."
                        ),
                    ),
                    html.Div(
                        [
                            html.Span("Monthly charges"),
                            html.Strong(f"${frame['MonthlyCharges'].iloc[0]:,.2f}"),
                        ]
                    ),
                ],
                className="result-facts",
            ),
            html.Div(
                [html.Span("SUGGESTED NEXT STEP", className="eyebrow"), html.P(action)],
                className="next-step",
            ),
            html.P(
                "A model estimate, not a calibrated probability or a guarantee. Suggested "
                "outreach is illustrative, not a measured treatment effect.",
                className="result-disclaimer",
            ),
        ],
        className="assessment-content",
    )


def benchmark_table(report: dict):
    rows = []
    for name, metrics in report["models"].items():
        rows.append(
            html.Tr(
                [
                    html.Td(
                        [name, html.Span("IN DEMO", className="table-tag")]
                        if name == report["demo_model"]
                        else name
                    ),
                    html.Td(f"{metrics['average_precision']:.3f}"),
                    html.Td(f"{metrics['roc_auc']:.3f}"),
                    html.Td(f"{metrics['recall']:.1%}"),
                    html.Td(f"{metrics['precision']:.1%}"),
                    html.Td(f"{metrics['flagged_customers']:,}"),
                ]
            )
        )
    return html.Div(
        html.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th(label, scope="col")
                            for label in [
                                "Model",
                                "Avg. precision",
                                "ROC AUC",
                                "Recall",
                                "Precision",
                                "Review queue",
                            ]
                        ]
                    )
                ),
                html.Tbody(rows),
            ]
        ),
        className="table-scroll",
    )


def page_header():
    return html.Header(
        html.Div(
            [
                html.A(
                    [
                        html.Span("r.", className="brand-mark"),
                        "retain",
                        html.Span("/", className="brand-slash"),
                        html.Span("an ML case study", className="brand-caption"),
                    ],
                    href="#",
                    className="brand",
                ),
                html.Nav(
                    [
                        html.A("Risk explorer", href="#explorer"),
                        html.A("Methodology", href="#methodology"),
                        html.A("View source ↗", href=REPOSITORY, className="source-link"),
                    ],
                    **{"aria-label": "Main navigation"},
                ),
            ],
            className="nav-inner",
        ),
        className="site-header",
    )


def hero():
    return html.Section(
        [
            html.Div(
                [
                    html.Div(
                        [html.Span(className="small-dot"), "CUSTOMER INTELLIGENCE"],
                        className="eyebrow hero-eyebrow",
                    ),
                    html.H1(["See the risk.", html.Br(), html.Span("Keep the connection.")]),
                    html.P(
                        "A value-aware approach to telecom churn. Explore which accounts may "
                        "need attention, and put each prediction in context.",
                        className="hero-copy",
                    ),
                    html.Div(
                        [
                            html.Span("Machine learning"),
                            html.Span("Decision support"),
                            html.Span("Reproducible research"),
                        ],
                        className="hero-tags",
                    ),
                ],
                className="hero-text",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("THE RESEARCH QUESTION", className="eyebrow"),
                            html.Span("01 / 04", className="matrix-index"),
                        ],
                        className="matrix-top",
                    ),
                    html.H2("Who needs attention first?"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("HIGHER VALUE"),
                                    html.Strong("Nurture"),
                                    html.Small("Lower churn score"),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Span("HIGHER VALUE"),
                                    html.Strong("Prioritize ↗"),
                                    html.Small("Higher churn score"),
                                ],
                                className="matrix-highlight",
                            ),
                            html.Div(
                                [
                                    html.Span("LOWER VALUE"),
                                    html.Strong("Monitor"),
                                    html.Small("Lower churn score"),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Span("LOWER VALUE"),
                                    html.Strong("Check in"),
                                    html.Small("Higher churn score"),
                                ]
                            ),
                        ],
                        className="value-matrix",
                    ),
                    html.P("Observed value = historical charges. Actions are illustrative."),
                ],
                className="hero-matrix",
            ),
        ],
        className="hero",
        id="overview",
    )


def statistics(stats):
    return html.Section(
        [
            html.Div([html.Span(label), html.Strong(value), html.Small(note)], className="stat")
            for (value, label, note) in stats
        ],
        className="stats-strip",
        **{"aria-label": "Project at a glance"},
    )


def profile_form(artifact):
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Customer profile"),
                            html.P("Adjust the inputs to explore a scenario."),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Example profile", htmlFor="example-profile", className="sr-only"
                            ),
                            dcc.Dropdown(
                                id="example-profile",
                                options=[
                                    {"label": "Month-to-month account", "value": "flexible"},
                                    {"label": "Established account", "value": "established"},
                                    {"label": "New connection", "value": "new"},
                                ],
                                value="flexible",
                                clearable=False,
                                searchable=False,
                            ),
                        ],
                        className="example-picker",
                    ),
                ],
                className="card-header",
            ),
            html.Div(
                [
                    html.Div(
                        [html.Span("01"), html.H4("Account & billing")],
                        className="form-section-title",
                    ),
                    html.Div(
                        [
                            field(name)
                            for name in [
                                "tenure",
                                "Contract",
                                "MonthlyCharges",
                                "TotalCharges",
                                "PaymentMethod",
                                "PaperlessBilling",
                            ]
                        ],
                        className="field-grid",
                    ),
                    html.Div(
                        [html.Span("02"), html.H4("Connected services")],
                        className="form-section-title",
                    ),
                    html.Div(
                        [
                            field(name)
                            for name in ["InternetService", "PhoneService", "MultipleLines"]
                        ],
                        className="field-grid",
                    ),
                    html.Details(
                        [
                            html.Summary("Service add-ons & customer details"),
                            html.P(
                                "Add-ons are excluded when internet is off; multiple lines are "
                                "excluded when phone service is off.",
                                className="form-hint",
                            ),
                            html.Div(
                                [
                                    field(name)
                                    for name in [
                                        *INTERNET_ADDONS,
                                        "gender",
                                        "SeniorCitizen",
                                        "Partner",
                                        "Dependents",
                                    ]
                                ],
                                className="field-grid",
                            ),
                        ],
                        className="additional-fields",
                    ),
                    html.Div(
                        [
                            html.P("All examples are illustrative. Inputs stay in this session."),
                            html.Button(
                                ["Run assessment", html.Span("↗")],
                                id="predict-button",
                                n_clicks=0,
                                disabled=artifact is None,
                                className="primary-button",
                            ),
                        ],
                        className="form-footer",
                    ),
                ],
                className="card-body",
            ),
        ],
        className="profile-card",
    )


def explorer(artifact, initial):
    return html.Section(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("01 / EXPLORE", className="eyebrow"),
                            html.H2("Every account has a story."),
                        ]
                    ),
                    html.P("Start with an illustrative profile, then make it your own."),
                ],
                className="section-heading",
            ),
            html.Div(
                [
                    profile_form(artifact),
                    html.Aside(
                        [
                            html.Div(initial, id="prediction-result", **{"aria-live": "polite"}),
                            html.Div(id="stale-notice", role="status"),
                        ],
                        className="result-card",
                    ),
                ],
                className="explorer-grid",
            ),
            dcc.Store(id="assessed-profile", data=PROFILES["flexible"]),
        ],
        id="explorer",
        className="explorer-section",
    )


def methodology(report):
    return html.Section(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("02 / VALIDATE", className="eyebrow"),
                            html.H2("Evidence behind the estimate."),
                        ]
                    ),
                    html.A(
                        "Read the model card ↗",
                        href=REPOSITORY + "/blob/master/docs/model-card.md",
                        className="text-link",
                    ),
                ],
                className="section-heading",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("A shared holdout. An honest comparison."),
                            html.P(
                                f"{report['split']['test_rows']:,} unseen accounts · "
                                f"stratified 80/20 split · seed {report['split']['seed']}"
                                if report
                                else "Train the model to generate the benchmark."
                            ),
                        ],
                        className="benchmark-heading",
                    ),
                    benchmark_table(report) if report else html.P("Results appear after training."),
                    html.P(
                        "Each review threshold maximizes F2 on five-fold training predictions. "
                        "Recall favors finding churners; precision shows the cost in false "
                        "alerts. The four-class model is retained to explore the original "
                        "research question, without a claim that it outperforms the binary "
                        "baselines.",
                        className="benchmark-note",
                    ),
                ],
                className="benchmark-card",
            ),
            html.Div(
                [
                    html.Article(
                        [html.Span(number, className="method-number"), html.H3(title), html.P(body)]
                    )
                    for (number, title, body) in [
                        (
                            "01",
                            "Separate before learning",
                            "Imputation, encoding, and the value cutoff are fitted on training "
                            "data within every fold. The holdout stays separate.",
                        ),
                        (
                            "02",
                            "Add value context",
                            "Four classes combine observed charges with churn. The two churn "
                            "class scores are summed for the account assessment.",
                        ),
                        (
                            "03",
                            "Make the tradeoff visible",
                            "A recall-focused threshold catches more churners and creates more "
                            "false alerts. Historical charges are a proxy, not revenue saved.",
                        ),
                    ]
                ],
                className="method-grid",
            ),
        ],
        id="methodology",
        className="methodology-section",
    )


def page_footer():
    return html.Footer(
        [
            html.Div([html.Strong("retain / "), "From coursework to a reproducible case study."]),
            html.Span("IBM Telco sample · Educational use"),
        ],
        className="site-footer",
    )


def layout(artifact: dict | None):
    report = artifact["report"] if artifact else None
    metrics = report["models"][report["demo_model"]] if report else None
    stats = [
        (
            f"{report['dataset']['rows']:,}" if report else "—",
            "Customer records",
            "IBM Telco sample",
        ),
        (
            f"{report['dataset']['churn_rate']:.1%}" if report else "—",
            "Observed churn",
            "Full sample prevalence",
        ),
        (f"{metrics['roc_auc']:.3f}" if metrics else "—", "Holdout ROC AUC", "Value-aware model"),
        ("4", "Customer segments", "Churn risk × observed value"),
    ]
    initial = (
        assessment(artifact, PROFILES["flexible"])
        if artifact
        else status_panel(
            "The demo is not initialized", "Follow the training step in the repository quick start."
        )
    )
    return html.Div(
        [
            html.A("Skip to risk explorer", href="#explorer", className="skip-link"),
            page_header(),
            html.Main(
                [
                    hero(),
                    statistics(stats),
                    explorer(artifact, initial),
                    methodology(report),
                    page_footer(),
                ],
                className="page-shell",
            ),
        ]
    )


def create_app(artifact_path: str | Path = ARTIFACT_PATH) -> Dash:
    try:
        artifact = load_artifact(artifact_path)
    except FileNotFoundError:
        LOGGER.warning("Run `python model_train.py` to initialize the demo model.")
        artifact = None
    app = Dash(
        __name__,
        assets_folder=str(ROOT / "assets"),
        title="Retain · Customer Churn Explorer",
        update_title=None,
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"},
            {
                "name": "description",
                "content": "Explore customer churn risk and value-aware retention in a "
                "reproducible ML case study.",
            },
        ],
    )
    app.index_string = app.index_string.replace(
        "{%favicon%}", '<link rel="icon" type="image/svg+xml" href="/assets/favicon.svg">'
    )
    app.layout = layout(artifact)

    @app.callback(
        [Output(name, "value") for name in FEATURES],
        Input("example-profile", "value"),
        prevent_initial_call=True,
    )
    def load_profile(profile):
        return [PROFILES[profile][name] for name in FEATURES]

    @app.callback(
        [Output(name, "disabled") for name in [*INTERNET_ADDONS, "MultipleLines"]],
        Input("InternetService", "value"),
        Input("PhoneService", "value"),
    )
    def service_availability(internet, phone):
        return [internet == "No"] * len(INTERNET_ADDONS) + [phone == "No"]

    @app.callback(
        Output("prediction-result", "children"),
        Output("assessed-profile", "data"),
        Input("predict-button", "n_clicks"),
        [State(name, "value") for name in FEATURES],
        prevent_initial_call=True,
    )
    def predict(_clicks, *values):
        profile = dict(zip(FEATURES, values, strict=True))
        if artifact is None:
            return (status_panel("Demo unavailable", "Follow the repository quick start."), None)
        try:
            result = assessment(artifact, profile)
        except ValueError as exc:
            return (status_panel("Check your profile", str(exc), error=True), None)
        return (result, profile)

    @app.callback(
        Output("stale-notice", "children"),
        Output("prediction-result", "className"),
        [Input(name, "value") for name in FEATURES],
        Input("assessed-profile", "data"),
    )
    def mark_pending(*values):
        profile = dict(zip(FEATURES, values[:-1], strict=True))
        previous = values[-1]
        if previous is not None and profile != previous:
            return ("Profile changed. Run assessment to update this result.", "stale-result")
        return ("", "")

    @app.server.get("/healthz")
    def health():
        return ({"status": "ready" if artifact else "model_missing"}, 200 if artifact else 503)

    return app
