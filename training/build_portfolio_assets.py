from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build static analysis assets for the portfolio site")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--web-public-dir", type=Path, default=Path("web/public"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_copy(src: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return "/" + dest.as_posix().split("web/public/", 1)[-1]


def row_probability(row: dict, class_name: str) -> float:
    return float(row["probabilities"].get(class_name, 0.0))


def margin(row: dict) -> float:
    vals = sorted((float(v) for v in row["probabilities"].values()), reverse=True)
    return vals[0] - vals[1] if len(vals) > 1 else vals[0]


def asset_name(row: dict, prefix: str, idx: int) -> str:
    digest = hashlib.sha1(str(row["path"]).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{idx:02d}_{row['true_class']}_to_{row['pred_class']}_{digest}.jpg"


def row_to_public(row: dict, dest_dir: Path, prefix: str, idx: int) -> dict:
    image_name = asset_name(row, prefix, idx)
    image = safe_copy(row["path"], dest_dir / image_name)
    probs = sorted(
        [{"class_name": name, "probability": float(prob)} for name, prob in row["probabilities"].items()],
        key=lambda item: item["probability"],
        reverse=True,
    )
    return {
        "id": Path(image_name).stem,
        "image": image,
        "true_class": row["true_class"],
        "top_prediction": row["pred_class"],
        "top_probability": row_probability(row, row["pred_class"]),
        "probabilities": probs,
        "margin": margin(row),
    }


def build_confusion_examples(rows: list[dict], output_dir: Path) -> list[dict]:
    errors = [row for row in rows if row["true_class"] != row["pred_class"]]
    pair_counts = Counter((row["true_class"], row["pred_class"]) for row in errors)
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in errors:
        by_pair[(row["true_class"], row["pred_class"])].append(row)

    confusions = []
    for rank, ((true_class, pred_class), count) in enumerate(pair_counts.most_common(16), start=1):
        candidates = sorted(by_pair[(true_class, pred_class)], key=lambda r: row_probability(r, r["pred_class"]), reverse=True)
        examples = [
            row_to_public(row, output_dir / "confusions", f"pair{rank:02d}", i)
            for i, row in enumerate(candidates[:4], start=1)
        ]
        confusions.append(
            {
                "true_class": true_class,
                "pred_class": pred_class,
                "count": int(count),
                "examples": examples,
            }
        )
    return confusions


def main() -> None:
    args = parse_args()
    rows = json.loads(args.predictions.read_text(encoding="utf-8"))

    analysis_dir = args.web_public_dir / "analysis"
    clean_dir(analysis_dir)
    metrics = json.loads((args.reports_dir / "final_metrics.json").read_text(encoding="utf-8"))
    top_confusions = json.loads((args.reports_dir / "top_confusions.json").read_text(encoding="utf-8"))
    noise_review = json.loads((args.reports_dir / "noise_review" / "review_summary.json").read_text(encoding="utf-8"))
    noise_review.pop("sheets", None)

    confusions = build_confusion_examples(rows, analysis_dir)

    payload = {
        "metrics": metrics,
        # Every non-zero pair, not a top-N slice. Truncating here left the
        # rendered matrix accounting for 172 of the 210 misses, with whole rows
        # blank for classes that do have errors.
        "top_confusions": top_confusions,
        "confusion_examples": confusions,
        "noise_review": noise_review,
    }
    write_json(args.web_public_dir / "data" / "analysis.json", payload)
    write_json(args.reports_dir / "analysis_manifest.json", payload)
    print(f"Wrote analysis assets to {analysis_dir}")


if __name__ == "__main__":
    main()
