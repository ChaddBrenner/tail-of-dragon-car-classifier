from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image

from common import ExperimentConfig, build_transforms, load_records, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build curated static gallery predictions")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--web-public-dir", type=Path, default=Path("web/public"))
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--display-count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


@torch.no_grad()
def predict(model, transform, path: str, device: torch.device) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    logits = model(tensor)
    return torch.softmax(logits, dim=1).cpu().numpy()[0]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    _, val_records, class_names, _ = load_records(args.data_dir, resplit=False, seed=args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ExperimentConfig(**checkpoint["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    transform = build_transforms(config, train=False)

    gallery_dir = args.web_public_dir / "gallery"
    data_dir = args.web_public_dir / "data"
    reports_public = args.web_public_dir / "reports"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_public.mkdir(parents=True, exist_ok=True)
    for stale_image in gallery_dir.glob("*.jpg"):
        stale_image.unlink()

    grouped: dict[str, list] = {}
    for record in val_records:
        grouped.setdefault(record.class_name, []).append(record)

    rng = np.random.default_rng(args.seed)
    samples = []
    sample_index = 0
    inference_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    model_version = f"{checkpoint['model_name']}-epoch{checkpoint.get('best_epoch', 'unknown')}-{inference_timestamp}"
    for class_name in class_names:
        choices = grouped.get(class_name, [])
        if not choices:
            continue
        indices = rng.choice(len(choices), size=min(args.samples_per_class, len(choices)), replace=False)
        for idx in indices:
            record = choices[int(idx)]
            sample_index += 1
            dest_name = f"{sample_index:03d}_{record.class_name}_{Path(record.path).stem[:20]}.jpg"
            dest_path = gallery_dir / dest_name
            shutil.copy2(record.path, dest_path)
            probs = predict(model, transform, record.path, device)
            ranked = [
                {"class_name": class_names[i], "probability": float(probs[i])}
                for i in np.argsort(probs)[::-1]
            ]
            samples.append(
                {
                    "id": dest_path.stem,
                    "image": f"/gallery/{dest_name}",
                    "true_class": record.class_name,
                    "top_prediction": ranked[0]["class_name"],
                    "top_probability": ranked[0]["probability"],
                    "probabilities": ranked,
                    "model_version": model_version,
                    "inference_timestamp": inference_timestamp,
                }
            )

    metrics = checkpoint.get("metrics", {})
    app_data = {
        "model": {
            "name": checkpoint["model_name"],
            "version": model_version,
            "classes": class_names,
            "metrics": metrics,
            "best_epoch": checkpoint.get("best_epoch"),
        },
        "gallery": {
            "display_count": min(args.display_count, len(samples)),
            "pool_size": len(samples),
            "samples_per_class": args.samples_per_class,
        },
        "samples": samples,
    }
    write_json(data_dir / "predictions.json", app_data)
    write_json(args.reports_dir / "gallery_manifest.json", app_data)

    for report_name in ["confusion_matrix.png", "classification_report.txt", "metrics.json", "top_confusions.json"]:
        found = list(args.checkpoint.parent.glob(report_name))
        if found:
            shutil.copy2(found[0], reports_public / report_name)
    print(f"Wrote {len(samples)} gallery samples to {gallery_dir}")


if __name__ == "__main__":
    main()
