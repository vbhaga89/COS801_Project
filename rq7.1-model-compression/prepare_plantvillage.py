from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from tqdm.auto import tqdm


def safe_label_name(label: str) -> str:
    label = label.strip().replace("___", "__")
    label = re.sub(r"[^\w.\-]+", "_", label)
    return label.strip("_")


def read_split(path: Path) -> list[str]:
    rows: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("raw/color/") and line.lower().endswith((".jpg", ".jpeg", ".png")):
            rows.append(line)
    return rows


def label_from_rel(rel_path: str) -> str:
    return safe_label_name(Path(rel_path).parts[2])


def group_for_rel(rel_path: str, leaf_map: dict[str, str]) -> str:
    variants = [
        rel_path,
        rel_path.replace("raw/", ""),
        str(Path(rel_path)),
        Path(rel_path).name,
    ]
    for key in variants:
        if key in leaf_map:
            return str(leaf_map[key])
    return Path(rel_path).stem


def extract_once(zip_path: Path, work_dir: Path) -> None:
    marker = work_dir / ".extracted_data_zip"
    if marker.exists() and marker.read_text(encoding="utf-8") == str(zip_path):
        print("Using existing extracted archive at", work_dir)
        return

    print("Extracting data.zip. This can take several minutes...")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(work_dir)
    marker.write_text(str(zip_path), encoding="utf-8")


def copy_split(
    *,
    split_name: str,
    rel_paths: list[str],
    source_root: Path,
    output: Path,
    leaf_map: dict[str, str],
    manifest_rows: list[dict[str, str]],
) -> None:
    print(f"Writing {split_name}: {len(rel_paths)} images")
    for rel_path in tqdm(rel_paths):
        label = label_from_rel(rel_path)
        source = source_root / rel_path
        if not source.exists():
            raise FileNotFoundError(source)

        destination_dir = output / split_name / label
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)

        manifest_rows.append(
            {
                "split": split_name,
                "class_name": label,
                "source_path": rel_path,
                "path": str(destination.relative_to(output)),
                "group": group_for_rel(rel_path, leaf_map),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the original RGB/color PlantVillage dataset for RQ7.1 in "
            "ImageFolder train/val/test layout. This uses the official image archive "
            "and color split files exposed by the PlantVillage Hugging Face mirror."
        )
    )
    parser.add_argument("--output", type=Path, default=Path("/content/drive/MyDrive/COS801_data/rq71_plantvillage"))
    parser.add_argument("--repo-id", default="mohanty/PlantVillage")
    parser.add_argument("--work-dir", type=Path, default=Path("/content/plantvillage_hf"))
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument(
        "--val-fraction-of-train",
        type=float,
        default=0.125,
        help="Validation fraction carved from the provided color training split.",
    )
    parser.add_argument(
        "--allow-image-level-val-split",
        action="store_true",
        help="Fallback only if no usable grouping signal is available.",
    )
    # Kept for backward-compatible Colab cells. The archive mirror exposes config
    # 'default', but this script uses concrete files rather than datasets configs.
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    print("Downloading official PlantVillage RGB/color resources from Hugging Face...")
    data_zip = Path(hf_hub_download(args.repo_id, "data.zip", repo_type="dataset"))
    train_txt = Path(hf_hub_download(args.repo_id, "splits/color_train.txt", repo_type="dataset"))
    test_txt = Path(hf_hub_download(args.repo_id, "splits/color_test.txt", repo_type="dataset"))

    try:
        leaf_map_path = Path(hf_hub_download(args.repo_id, "leaf_grouping/leaf-map.json", repo_type="dataset"))
        leaf_map: dict[str, str] = json.loads(leaf_map_path.read_text(encoding="utf-8"))
        print("Loaded leaf map entries:", len(leaf_map))
    except Exception as exc:
        if not args.allow_image_level_val_split:
            raise RuntimeError(
                "Could not load PlantVillage leaf grouping metadata. "
                "Rerun with --allow-image-level-val-split only if you accept documenting this limitation."
            ) from exc
        print("Leaf map unavailable; using filename stems as grouping fallback:", exc)
        leaf_map = {}

    extract_once(data_zip, args.work_dir)

    raw_color_candidates = list(args.work_dir.rglob("raw/color"))
    if not raw_color_candidates:
        raise FileNotFoundError("Could not find raw/color after extracting data.zip")
    source_root = raw_color_candidates[0].parent.parent
    print("Raw color folder:", raw_color_candidates[0])

    train_rels = read_split(train_txt)
    test_rels = read_split(test_txt)
    classes = sorted({label_from_rel(path) for path in train_rels + test_rels})

    print("Color train list:", len(train_rels))
    print("Color test list:", len(test_rels))
    print("Classes:", len(classes))
    if len(classes) != 38:
        raise RuntimeError(f"Expected 38 PlantVillage classes, found {len(classes)}")

    labels = [label_from_rel(path) for path in train_rels]
    groups = [group_for_rel(path, leaf_map) for path in train_rels]
    if len(set(groups)) > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_fraction_of_train, random_state=args.seed)
        train_idx, val_idx = next(splitter.split(train_rels, labels, groups))
        split_note = "group-aware validation split using PlantVillage leaf map / filename grouping"
    elif args.allow_image_level_val_split:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_fraction_of_train, random_state=args.seed)
        train_idx, val_idx = next(splitter.split(train_rels, labels))
        split_note = "stratified image-level validation split fallback; document as a limitation"
    else:
        raise RuntimeError("No usable grouping signal found; rerun with --allow-image-level-val-split only if acceptable.")

    split_map = {
        "train": [train_rels[int(index)] for index in train_idx],
        "val": [train_rels[int(index)] for index in val_idx],
        "test": test_rels,
    }

    manifest_rows: list[dict[str, str]] = []
    for split_name, rel_paths in split_map.items():
        copy_split(
            split_name=split_name,
            rel_paths=rel_paths,
            source_root=source_root,
            output=args.output,
            leaf_map=leaf_map,
            manifest_rows=manifest_rows,
        )

    manifest_path = args.output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "class_name", "source_path", "path", "group"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("Prepared PlantVillage RGB/color dataset for RQ7.1")
    print("Output:", args.output)
    print("Manifest:", manifest_path)
    print("Split note:", split_note)
    print({split_name: len(rel_paths) for split_name, rel_paths in split_map.items()})


if __name__ == "__main__":
    main()
