from __future__ import annotations

import hashlib
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def build_dataloaders(config: dict[str, Any]) -> tuple[dict[str, DataLoader], list[str]]:
    root = Path(config["data_dir"])
    required = {split: root / split for split in ("train", "val", "test")}
    missing = [str(path) for path in required.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Final RQ7.1 runs require explicit train/val/test directories. Missing: "
            + ", ".join(missing)
        )

    train_transform, eval_transform = build_transforms(int(config["image_size"]))
    datasets_by_split = {
        "train": datasets.ImageFolder(required["train"], transform=train_transform),
        "val": datasets.ImageFolder(required["val"], transform=eval_transform),
        "test": datasets.ImageFolder(required["test"], transform=eval_transform),
    }
    class_names = datasets_by_split["train"].classes
    for split, dataset in datasets_by_split.items():
        if dataset.classes != class_names:
            raise ValueError(f"Class mapping for {split} does not match the training split")

    generator = torch.Generator().manual_seed(int(config["seed"]))
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=split == "train",
            num_workers=int(config["num_workers"]),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=int(config["num_workers"]) > 0,
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets_by_split.items()
    }
    return loaders, class_names


def audit_splits(data_dir: str | Path) -> dict[str, Any]:
    root = Path(data_dir)
    split_counts: dict[str, Counter[str]] = {}
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for split in ("train", "val", "test"):
        split_root = root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_root}")
        counts: Counter[str] = Counter()
        for path in sorted(split_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = path.relative_to(split_root).parts[0]
            counts[label] += 1
            digest = _sha256_file(path)
            hashes[digest].append((split, str(path.relative_to(root))))
        split_counts[split] = counts

    leaked = []
    for digest, occurrences in hashes.items():
        if len({split for split, _ in occurrences}) > 1:
            leaked.append({"sha256": digest, "occurrences": occurrences})

    return {
        "class_counts": {split: dict(sorted(counts.items())) for split, counts in split_counts.items()},
        "totals": {split: sum(counts.values()) for split, counts in split_counts.items()},
        "cross_split_exact_duplicate_groups": leaked,
        "cross_split_exact_duplicate_count": len(leaked),
    }


def quarantine_cross_split_duplicates(data_dir: str | Path, quarantine_dir: str | Path) -> list[dict[str, Any]]:
    """Move exact duplicate images crossing splits out of the active dataset.

    The train split is preferred as the kept copy, then val, then test. This is
    intended for disposable Colab runtime copies of the dataset; the durable
    Drive source is left unchanged.
    """
    root = Path(data_dir)
    quarantine_root = Path(quarantine_dir)
    report = audit_splits(root)
    moved: list[dict[str, Any]] = []
    split_priority = {"train": 0, "val": 1, "test": 2}

    for duplicate_group in report["cross_split_exact_duplicate_groups"]:
        occurrences = duplicate_group["occurrences"]
        keep_split, keep_relative = sorted(
            occurrences,
            key=lambda item: (split_priority.get(item[0], 99), item[1]),
        )[0]
        for split, relative in occurrences:
            if split == keep_split and relative == keep_relative:
                continue
            source = root / relative
            if not source.exists():
                continue
            destination = quarantine_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            counter = 1
            final_destination = destination
            while final_destination.exists():
                final_destination = destination.with_name(
                    f"{destination.stem}.duplicate{counter}{destination.suffix}"
                )
                counter += 1
            shutil.move(str(source), str(final_destination))
            moved.append(
                {
                    "sha256": duplicate_group["sha256"],
                    "kept": keep_relative,
                    "moved": relative,
                    "quarantine_path": str(final_destination),
                }
            )

    return moved
