from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
import timm
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from common import (
    ExperimentConfig,
    Record,
    RecordsDataset,
    build_transforms,
    get_device,
    load_records,
    set_seed,
    stratified_subsample,
    write_json,
    write_manifest,
)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def class_weights(records: Sequence[Record], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount([r.label_idx for r in records], minlength=num_classes)
    weights = counts.sum() / np.clip(counts, 1, None)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_loader(
    records: Sequence[Record],
    config: ExperimentConfig,
    device: torch.device,
    train: bool,
    num_classes: int,
) -> DataLoader:
    dataset = RecordsDataset(records, transform=build_transforms(config, train=train))
    sampler = None
    shuffle = train
    if train and config.weighted_sampler:
        labels = np.array([r.label_idx for r in records], dtype=np.int64)
        counts = np.bincount(labels, minlength=num_classes)
        weights = 1.0 / np.clip(counts, 1, None)
        sample_weights = weights[labels]
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    persistent_workers = config.num_workers > 0 and platform.system() != "Windows"
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers,
        drop_last=train and (config.mixup_alpha > 0 or config.cutmix_alpha > 0),
    )


def freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for module in [getattr(model, "head", None), getattr(model, "classifier", None), getattr(model, "fc", None)]:
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True


def unfreeze(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def optimizer_for(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    mixup_fn: Mixup | None,
    amp: bool,
    grad_accum_steps: int,
) -> float:
    model.train()
    total_loss = 0.0
    seen = 0
    optimizer.zero_grad(set_to_none=True)
    for step, (images, targets, _) in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if mixup_fn is not None:
            images, mixed_targets = mixup_fn(images, targets)
        else:
            mixed_targets = targets
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, mixed_targets) / grad_accum_steps
        scaler.scale(loss).backward()
        if step % grad_accum_steps == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        batch = images.size(0)
        total_loss += float(loss.detach().cpu()) * batch * grad_accum_steps
        seen += batch
    return total_loss / max(seen, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, tta: bool = False) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    total_loss = 0.0
    seen = 0
    all_true: list[int] = []
    all_pred: list[int] = []
    all_prob: list[np.ndarray] = []
    all_paths: list[str] = []
    for images, targets, paths in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        if tta:
            flipped = torch.flip(images, dims=[3])
            logits = (logits + model(flipped)) / 2
        loss = criterion(logits, targets)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        batch = images.size(0)
        total_loss += float(loss.detach().cpu()) * batch
        seen += batch
        all_true.extend(targets.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())
        all_prob.extend(probs.cpu().numpy())
        all_paths.extend(paths)
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    prob = np.stack(all_prob) if all_prob else np.empty((0, 0))
    top3 = np.mean([t in np.argsort(p)[-3:] for t, p in zip(y_true, prob)]) if len(y_true) else 0.0
    metrics = {
        "val_loss": total_loss / max(seen, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "top3_accuracy": float(top3),
        "evaluated_samples": int(seen),
    }
    return metrics, y_true, y_pred, prob, all_paths


def plot_confusion(output_path: Path, y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def top_confusions(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> list[dict]:
    rows: list[dict] = []
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    totals = cm.sum(axis=1)
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
    return sorted(rows, key=lambda r: r["count"], reverse=True)


def save_predictions(output_path: Path, paths: list[str], y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> None:
    rows = []
    for path, true_idx, prob in zip(paths, y_true, probs):
        rows.append(
            {
                "path": path,
                "true_class": class_names[int(true_idx)],
                "pred_class": class_names[int(np.argmax(prob))],
                "probabilities": {class_names[i]: float(prob[i]) for i in range(len(class_names))},
            }
        )
    write_json(output_path, rows)


def run_experiment(
    name: str,
    config: ExperimentConfig,
    train_records: Sequence[Record],
    val_records: Sequence[Record],
    class_names: list[str],
    output_dir: Path,
    device: torch.device,
    full_validation: bool,
) -> dict:
    set_seed(config.seed)
    exp_dir = output_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    effective_train = stratified_subsample(train_records, config.train_fraction, config.seed)
    effective_val = stratified_subsample(val_records, config.val_fraction, config.seed + 1)
    write_json(exp_dir / "config.json", config.to_dict())

    train_loader = build_loader(effective_train, config, device, True, len(class_names))
    val_loader = build_loader(effective_val, config, device, False, len(class_names))

    model = timm.create_model(config.model_name, pretrained=True, num_classes=len(class_names)).to(device)
    if config.freeze_epochs > 0:
        freeze_backbone(model)
        opt = optimizer_for(model, config.head_lr, config.weight_decay)
    else:
        opt = optimizer_for(model, config.lr, config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(config.epochs, 1))

    weights = class_weights(effective_train, len(class_names), device) if config.use_class_weights else None
    mixup_fn = None
    if config.mixup_alpha > 0 or config.cutmix_alpha > 0:
        mixup_fn = Mixup(
            mixup_alpha=config.mixup_alpha,
            cutmix_alpha=config.cutmix_alpha,
            prob=1.0,
            switch_prob=0.5,
            label_smoothing=config.label_smoothing,
            num_classes=len(class_names),
        )
        train_criterion: nn.Module = SoftTargetCrossEntropy()
    elif config.focal_gamma > 0:
        train_criterion = FocalLoss(config.focal_gamma, weight=weights, label_smoothing=config.label_smoothing)
    else:
        train_criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=config.label_smoothing)
    val_criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=config.amp and device.type == "cuda")

    best_score = -math.inf
    best_epoch = 0
    no_improve = 0
    history: list[dict] = []
    start = time.time()
    best_path = exp_dir / "best.pt"

    for epoch in range(1, config.epochs + 1):
        if config.freeze_epochs > 0 and epoch == config.freeze_epochs + 1:
            unfreeze(model)
            opt = optimizer_for(model, config.lr, config.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(config.epochs - epoch + 1, 1))
        train_loss = train_epoch(
            model,
            train_loader,
            opt,
            train_criterion,
            device,
            scaler,
            mixup_fn,
            config.amp,
            max(config.grad_accum_steps, 1),
        )
        metrics, y_true, y_pred, probs, paths = evaluate(model, val_loader, val_criterion, device, tta=config.tta)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(row)
        print(f"[{name}] epoch={epoch}/{config.epochs} train_loss={train_loss:.4f} acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}", flush=True)
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_name": config.model_name,
                    "class_names": class_names,
                    "state_dict": model.state_dict(),
                    "config": config.to_dict(),
                    "best_epoch": best_epoch,
                    "metrics": metrics,
                },
                best_path,
            )
            save_predictions(exp_dir / "best_val_predictions.json", paths, y_true, probs, class_names)
        else:
            no_improve += 1
        if no_improve >= config.patience:
            print(f"[{name}] early stop at epoch {epoch}", flush=True)
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    eval_records = val_records if full_validation else effective_val
    final_loader = build_loader(eval_records, config, device, False, len(class_names))
    final_metrics, y_true, y_pred, probs, paths = evaluate(model, final_loader, val_criterion, device, tta=config.tta)

    (exp_dir / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0),
        encoding="utf-8",
    )
    plot_confusion(exp_dir / "confusion_matrix.png", y_true, y_pred, class_names, name)
    save_predictions(exp_dir / "val_predictions.json", paths, y_true, probs, class_names)
    write_json(exp_dir / "top_confusions.json", top_confusions(y_true, y_pred, class_names)[:50])
    write_json(exp_dir / "history.json", history)
    payload = {
        "name": name,
        "model_name": config.model_name,
        "checkpoint": str(best_path.resolve()),
        "best_epoch": best_epoch,
        "runtime_seconds": time.time() - start,
        "effective_train_samples": len(effective_train),
        "effective_val_samples": len(effective_val),
        **final_metrics,
        "config": config.to_dict(),
    }
    write_json(exp_dir / "metrics.json", payload)
    return payload


def default_candidates(seed: int, batch_size: int, workers: int, train_fraction: float, val_fraction: float, epochs: int) -> list[ExperimentConfig]:
    return [
        ExperimentConfig("convnext_tiny", 288, epochs, batch_size, 2e-4, 1.4e-3, seed=seed, num_workers=workers, train_fraction=train_fraction, val_fraction=val_fraction, mixup_alpha=0.05, cutmix_alpha=0.1, random_erasing_prob=0.08, crop_scale_min=0.78),
        ExperimentConfig("tf_efficientnetv2_s.in21k_ft_in1k", 300, epochs, max(8, batch_size // 2), 1.8e-4, 1.2e-3, seed=seed + 1, num_workers=workers, train_fraction=train_fraction, val_fraction=val_fraction, mixup_alpha=0.1, cutmix_alpha=0.1, random_erasing_prob=0.10, crop_scale_min=0.72),
        ExperimentConfig("swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", 224, epochs, max(8, batch_size // 2), 1.2e-4, 8e-4, seed=seed + 2, num_workers=workers, train_fraction=train_fraction, val_fraction=val_fraction, mixup_alpha=0.1, cutmix_alpha=0.1, random_erasing_prob=0.05, crop_scale_min=0.80),
        ExperimentConfig("convnext_small", 288, epochs, max(8, batch_size // 2), 1.6e-4, 1.2e-3, seed=seed + 3, num_workers=workers, train_fraction=train_fraction, val_fraction=val_fraction, mixup_alpha=0.05, cutmix_alpha=0.1, random_erasing_prob=0.08, crop_scale_min=0.80, grad_accum_steps=1),
        ExperimentConfig("coatnet_0_rw_224.sw_in1k", 224, epochs, max(8, batch_size // 2), 1.2e-4, 8e-4, seed=seed + 4, num_workers=workers, train_fraction=train_fraction, val_fraction=val_fraction, mixup_alpha=0.05, cutmix_alpha=0.1, random_erasing_prob=0.06, crop_scale_min=0.80),
        ExperimentConfig("maxvit_tiny_rw_224.sw_in1k", 224, epochs, max(4, batch_size // 4), 1.0e-4, 7e-4, seed=seed + 5, num_workers=workers, train_fraction=train_fraction, val_fraction=val_fraction, mixup_alpha=0.05, cutmix_alpha=0.1, random_erasing_prob=0.06, crop_scale_min=0.82, grad_accum_steps=2),
        ExperimentConfig("vit_small_patch16_224.augreg_in21k_ft_in1k", 224, epochs, max(8, batch_size // 2), 1.0e-4, 7e-4, seed=seed + 6, num_workers=workers, train_fraction=train_fraction, val_fraction=val_fraction, mixup_alpha=0.1, cutmix_alpha=0.1, random_erasing_prob=0.05, crop_scale_min=0.82),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tail of Dragon car classifiers")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--mode", choices=["smoke", "search", "final"], default="search")
    parser.add_argument("--resplit", action="store_true")
    parser.add_argument("--search-epochs", type=int, default=1)
    parser.add_argument("--search-train-fraction", type=float, default=0.25)
    parser.add_argument("--search-val-fraction", type=float, default=0.35)
    parser.add_argument("--final-epochs", type=int, default=10)
    parser.add_argument("--final-model", type=str, default="")
    parser.add_argument("--final-img-size", type=int, default=288)
    parser.add_argument("--full-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    print(f"Using device: {device}", flush=True)
    train_records, val_records, class_names, split_meta = load_records(args.data_dir, args.resplit, seed=args.seed)
    write_json(args.output_dir / "split_metadata.json", split_meta)
    write_manifest(args.output_dir / "train_manifest.csv", train_records)
    write_manifest(args.output_dir / "val_manifest.csv", val_records)
    print(f"Split: train={len(train_records)} val={len(val_records)} classes={class_names}", flush=True)

    results: list[dict] = []
    if args.mode == "smoke":
        candidates = [ExperimentConfig("resnet18", 160, 1, 16, seed=args.seed, num_workers=0, train_fraction=0.01, val_fraction=0.01, use_randaugment=False, random_erasing_prob=0.0)]
    elif args.mode == "search":
        candidates = default_candidates(args.seed, args.batch_size, args.num_workers, args.search_train_fraction, args.search_val_fraction, args.search_epochs)
    else:
        model_name = args.final_model or "convnext_tiny"
        candidates = [
            ExperimentConfig(
                model_name=model_name,
                img_size=args.final_img_size,
                epochs=args.final_epochs,
                batch_size=args.batch_size,
                lr=1.8e-4,
                head_lr=1.2e-3,
                seed=args.seed,
                num_workers=args.num_workers,
                train_fraction=1.0,
                val_fraction=1.0,
                patience=3,
                mixup_alpha=0.05,
                cutmix_alpha=0.1,
                weighted_sampler=True,
                label_smoothing=0.08,
                random_erasing_prob=0.08,
                crop_scale_min=0.80,
                tta=True,
            )
        ]

    for i, cfg in enumerate(candidates):
        safe_model = cfg.model_name.replace("/", "_").replace(".", "_")
        name = f"{args.mode}_{i:02d}_{safe_model}_{cfg.img_size}"
        result = run_experiment(name, cfg, train_records, val_records, class_names, args.output_dir, device, full_validation=args.full_validation or args.mode == "final")
        results.append(result)

    leaderboard = sorted(results, key=lambda r: (r["macro_f1"], r["accuracy"]), reverse=True)
    write_json(args.output_dir / f"{args.mode}_leaderboard.json", leaderboard)
    print("Leaderboard:", flush=True)
    for row in leaderboard:
        print(f"{row['name']} model={row['model_name']} acc={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
