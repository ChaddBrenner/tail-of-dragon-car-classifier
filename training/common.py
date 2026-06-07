from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class Record:
    path: str
    label_idx: int
    class_name: str
    source_split: str


@dataclass
class ExperimentConfig:
    model_name: str
    img_size: int = 224
    epochs: int = 5
    batch_size: int = 64
    lr: float = 2e-4
    head_lr: float = 1e-3
    weight_decay: float = 2e-4
    freeze_epochs: int = 1
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    weighted_sampler: bool = True
    use_class_weights: bool = False
    focal_gamma: float = 0.0
    train_fraction: float = 1.0
    val_fraction: float = 1.0
    patience: int = 3
    num_workers: int = 4
    seed: int = 42
    crop_scale_min: float = 0.72
    crop_scale_max: float = 1.0
    randaugment_num_ops: int = 2
    randaugment_magnitude: int = 7
    use_randaugment: bool = True
    use_autoaugment: bool = False
    color_jitter: float = 0.08
    blur_prob: float = 0.05
    random_erasing_prob: float = 0.12
    tta: bool = False
    amp: bool = True
    grad_accum_steps: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discover_classes(data_dir: Path) -> list[str]:
    train_dir = data_dir / "train"
    val_dir = data_dir / "validation"
    source = train_dir if train_dir.exists() else val_dir
    if not source.exists():
        raise FileNotFoundError(f"Expected train/ or validation/ under {data_dir}")
    return sorted(p.name for p in source.iterdir() if p.is_dir() and not p.name.startswith("__"))


def _load_split(split_dir: Path, split: str, class_to_idx: dict[str, int]) -> list[Record]:
    records: list[Record] = []
    if not split_dir.exists():
        return records
    for class_name, label_idx in class_to_idx.items():
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                records.append(Record(str(path.resolve()), label_idx, class_name, split))
    return records


def remove_duplicate_leakage(train_records: list[Record], val_records: list[Record]) -> tuple[list[Record], int]:
    val_keys = {(r.class_name, Path(r.path).name) for r in val_records}
    cleaned = [r for r in train_records if (r.class_name, Path(r.path).name) not in val_keys]
    return cleaned, len(train_records) - len(cleaned)


def load_records(data_dir: Path, resplit: bool = False, val_ratio: float = 0.2, seed: int = 42) -> tuple[list[Record], list[Record], list[str], dict]:
    class_names = discover_classes(data_dir)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    train_records = _load_split(data_dir / "train", "train", class_to_idx)
    val_records = _load_split(data_dir / "validation", "validation", class_to_idx)

    if resplit:
        all_records = train_records + val_records
        labels = [r.label_idx for r in all_records]
        train_records, val_records = train_test_split(
            all_records,
            test_size=val_ratio,
            random_state=seed,
            stratify=labels,
        )
        train_records = [Record(r.path, r.label_idx, r.class_name, "train") for r in train_records]
        val_records = [Record(r.path, r.label_idx, r.class_name, "validation") for r in val_records]
        duplicate_removed = 0
    else:
        train_records, duplicate_removed = remove_duplicate_leakage(train_records, val_records)

    metadata = {
        "data_dir": str(data_dir.resolve()),
        "resplit": resplit,
        "val_ratio": val_ratio,
        "seed": seed,
        "classes": class_names,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "duplicate_leakage_removed": duplicate_removed,
    }
    return list(train_records), list(val_records), class_names, metadata


def stratified_subsample(records: Sequence[Record], fraction: float, seed: int) -> list[Record]:
    if fraction >= 0.999:
        return list(records)
    rng = random.Random(seed)
    grouped: dict[int, list[Record]] = {}
    for record in records:
        grouped.setdefault(record.label_idx, []).append(record)
    sampled: list[Record] = []
    for label_records in grouped.values():
        n = max(1, int(round(len(label_records) * fraction)))
        sampled.extend(rng.sample(label_records, min(n, len(label_records))))
    return sorted(sampled, key=lambda r: r.path)


class RecordsDataset(Dataset):
    def __init__(self, records: Sequence[Record], transform=None):
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        try:
            image = Image.open(record.path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (600, 400), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, record.label_idx, record.path


def build_transforms(config: ExperimentConfig, train: bool):
    if train:
        ops: list = [
            transforms.RandomResizedCrop(
                config.img_size,
                scale=(config.crop_scale_min, config.crop_scale_max),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        if config.color_jitter:
            ops.append(
                transforms.ColorJitter(
                    brightness=config.color_jitter,
                    contrast=config.color_jitter,
                    saturation=config.color_jitter,
                    hue=min(0.04, config.color_jitter / 4),
                )
            )
        if config.use_autoaugment:
            ops.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
        elif config.use_randaugment:
            ops.append(
                transforms.RandAugment(
                    num_ops=config.randaugment_num_ops,
                    magnitude=config.randaugment_magnitude,
                )
            )
        if config.blur_prob:
            ops.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                    p=config.blur_prob,
                )
            )
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        if config.random_erasing_prob:
            ops.append(transforms.RandomErasing(p=config.random_erasing_prob, scale=(0.02, 0.12), value="random"))
        return transforms.Compose(ops)

    return transforms.Compose(
        [
            transforms.Resize(int(config.img_size * 1.15), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_manifest(path: Path, records: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label_idx", "class_name", "source_split"])
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

