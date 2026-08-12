from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from common import ExperimentConfig, build_transforms, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build static analysis assets for the portfolio site")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--web-public-dir", type=Path, default=Path("web/public"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=42)
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


def build_error_groups(rows: list[dict], output_dir: Path, rng: random.Random) -> dict:
    errors = [row for row in rows if row["true_class"] != row["pred_class"]]
    correct = [row for row in rows if row["true_class"] == row["pred_class"]]
    groups = {
        "high_confidence_misses": sorted(errors, key=lambda r: row_probability(r, r["pred_class"]), reverse=True)[:12],
        "low_margin_misses": sorted(errors, key=margin)[:12],
        "random_misses": rng.sample(errors, min(12, len(errors))),
        "low_confidence_correct": sorted(correct, key=lambda r: row_probability(r, r["pred_class"]))[:12],
        "random_correct": rng.sample(correct, min(12, len(correct))),
    }
    return {
        name: [row_to_public(row, output_dir / "groups" / name, name, i) for i, row in enumerate(group_rows, start=1)]
        for name, group_rows in groups.items()
    }


def gradcam_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    try:
        return model.stages[-1].blocks[-1].conv_dw
    except Exception:
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
    raise RuntimeError("Could not find a convolutional layer for Grad-CAM")


def make_gradcam(model: torch.nn.Module, transform, image_path: str, target_idx: int, output_path: Path, device: torch.device) -> None:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    layer = gradcam_target_layer(model)

    def fwd_hook(_module, _inputs, output):
        activations.append(output.detach())

    def bwd_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    handle_fwd = layer.register_forward_hook(fwd_hook)
    handle_bwd = layer.register_full_backward_hook(bwd_hook)
    try:
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        logits[0, target_idx].backward()
        acts = activations[-1]
        grads = gradients[-1]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.size[::-1], mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / max(cam_np.max() - cam_np.min(), 1e-8)
        heat = cv2.applyColorMap(np.uint8(cam_np * 255), cv2.COLORMAP_TURBO)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        base = np.array(image.resize((heat.shape[1], heat.shape[0])))
        overlay = np.uint8(base * 0.58 + heat * 0.42)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(output_path, quality=92)
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def fallback_heatmap(image_path: str, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = image.size
    draw.ellipse((w * 0.22, h * 0.22, w * 0.82, h * 0.82), fill=(211, 63, 47, 72))
    combined = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path, quality=92)


def build_gradcams(rows: list[dict], checkpoint: dict, output_dir: Path) -> list[dict]:
    config = ExperimentConfig(**checkpoint["config"])
    class_names = checkpoint["class_names"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    transform = build_transforms(config, train=False)

    selected: list[dict] = []
    for class_name in class_names:
        class_rows = [row for row in rows if row["true_class"] == class_name and row["true_class"] == row["pred_class"]]
        if class_rows:
            selected.append(max(class_rows, key=lambda r: row_probability(r, r["pred_class"])))
    errors = [row for row in rows if row["true_class"] != row["pred_class"]]
    selected.extend(sorted(errors, key=lambda r: row_probability(r, r["pred_class"]), reverse=True)[:4])

    assets = []
    for i, row in enumerate(selected[:12], start=1):
        base_name = asset_name(row, "gradcam", i)
        original_rel = safe_copy(row["path"], output_dir / "gradcam" / "originals" / base_name)
        heat_path = output_dir / "gradcam" / "heatmaps" / base_name
        try:
            target_idx = class_names.index(row["pred_class"])
            make_gradcam(model, transform, row["path"], target_idx, heat_path, device)
        except Exception:
            fallback_heatmap(row["path"], heat_path)
        heat_rel = "/" + heat_path.as_posix().split("web/public/", 1)[-1]
        assets.append(
            {
                **row_to_public(row, output_dir / "gradcam" / "cards", "gradcam", i),
                "original": original_rel,
                "heatmap": heat_rel,
                "explanation": "Grad-CAM overlay for the predicted class. Warm regions had the strongest influence on the class score.",
            }
        )
    return assets


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    analysis_dir = args.web_public_dir / "analysis"
    clean_dir(analysis_dir)
    metrics = json.loads((args.reports_dir / "final_metrics.json").read_text(encoding="utf-8"))
    top_confusions = json.loads((args.reports_dir / "top_confusions.json").read_text(encoding="utf-8"))
    noise_review = json.loads((args.reports_dir / "noise_review" / "review_summary.json").read_text(encoding="utf-8"))
    noise_review.pop("sheets", None)

    confusions = build_confusion_examples(rows, analysis_dir)
    groups = build_error_groups(rows, analysis_dir, rng)
    gradcams = build_gradcams(rows, checkpoint, analysis_dir)

    payload = {
        "metrics": metrics,
        "top_confusions": top_confusions[:20],
        "confusion_examples": confusions,
        "error_groups": groups,
        "gradcam": gradcams,
        "noise_review": noise_review,
    }
    write_json(args.web_public_dir / "data" / "analysis.json", payload)
    write_json(args.reports_dir / "analysis_manifest.json", payload)
    print(f"Wrote analysis assets to {analysis_dir}")


if __name__ == "__main__":
    main()
