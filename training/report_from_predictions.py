from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reports from saved validation predictions")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", type=str, default="final_convnext_tiny_288")
    return parser.parse_args()


def plot_confusion(output_path: Path, y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def top_confusions(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> list[dict]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    totals = cm.sum(axis=1)
    rows: list[dict] = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            if i == j or cm[i, j] == 0:
                continue
            rows.append(
                {
                    "true_class": true_name,
                    "pred_class": pred_name,
                    "count": int(cm[i, j]),
                    "pct_of_true": float(cm[i, j] / max(totals[i], 1)),
                }
            )
    return sorted(rows, key=lambda row: row["count"], reverse=True)


def main() -> None:
    args = parse_args()
    rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError("Prediction file is empty")

    class_names = list(rows[0]["probabilities"].keys())
    index = {name: i for i, name in enumerate(class_names)}
    y_true = np.array([index[row["true_class"]] for row in rows])
    y_pred = np.array([index[row["pred_class"]] for row in rows])
    probs = np.array([[row["probabilities"][name] for name in class_names] for row in rows])
    top3 = np.mean([truth in np.argsort(prob)[-3:] for truth, prob in zip(y_true, probs)])
    error_rows = [row for row in rows if row["true_class"] != row["pred_class"]]
    errors_by_class = Counter(row["true_class"] for row in error_rows)

    metrics = {
        "name": args.name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "top3_accuracy": float(top3),
        "evaluated_samples": int(len(rows)),
        "error_count": int(len(error_rows)),
        "class_names": class_names,
        "errors_by_class": {name: int(errors_by_class.get(name, 0)) for name in class_names},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "top_confusions.json", top_confusions(y_true, y_pred, class_names)[:50])
    (args.output_dir / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0),
        encoding="utf-8",
    )
    plot_confusion(args.output_dir / "confusion_matrix.png", y_true, y_pred, class_names, args.name)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
