"""Reproduce the benchmark and build the artifact used by the Dash application."""

import argparse

from churn.data import ARTIFACT_PATH, DATA_PATH, REPORT_PATH
from churn.modeling import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DATA_PATH, help="Input Telco CSV")
    parser.add_argument("--artifact", default=ARTIFACT_PATH, help="Output joblib model")
    parser.add_argument("--report", default=REPORT_PATH, help="Output benchmark JSON")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run_training(args.data, args.artifact, args.report, args.seed)
    print("\nHoldout results (thresholds selected using training data only)")
    for name, metrics in report["models"].items():
        print(
            f"{name:31}  AP {metrics['average_precision']:.3f}  "
            f"ROC AUC {metrics['roc_auc']:.3f}  Recall {metrics['recall']:.3f}"
        )
    print(f"\nModel: {args.artifact}\nReport: {args.report}")


if __name__ == "__main__":
    main()
