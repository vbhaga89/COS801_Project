from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rq71.compression import (  # noqa: E402
    benchmark_onnx,
    evaluate_onnx,
    export_onnx,
    parameter_count,
    static_int8_quantize,
    structured_prune,
)
from rq71.data import audit_splits, build_dataloaders  # noqa: E402
from rq71.engine import evaluate, train_distillation, train_supervised  # noqa: E402
from rq71.models import SUPPORTED_MODELS, build_model, set_backbone_trainable  # noqa: E402


RESULT_FIELDS = [
    "timestamp_utc",
    "variant",
    "model",
    "teacher_model",
    "compression",
    "seed",
    "num_classes",
    "accuracy",
    "macro_f1",
    "baseline_accuracy",
    "accuracy_drop",
    "baseline_macro_f1",
    "macro_f1_drop",
    "parameters",
    "parameter_reduction",
    "artifact",
    "model_size_mb",
    "compression_ratio",
    "median_latency_ms",
    "p95_latency_ms",
    "meets_accuracy_budget",
    "meets_size_budget",
    "meets_latency_budget",
    "meets_all_budgets",
    "python",
    "torch",
]


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def output_directory(config: dict[str, Any], model_name: str) -> Path:
    path = Path(config["output_dir"]) / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def checkpoint_payload(model: torch.nn.Module, model_name: str, classes: list[str]) -> dict[str, Any]:
    return {"model_name": model_name, "classes": classes, "state_dict": model.cpu().state_dict()}


def load_checkpoint(path: str | Path) -> tuple[torch.nn.Module, str, list[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    name = payload["model_name"]
    classes = payload["classes"]
    model = build_model(name, len(classes), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    return model, name, classes


def result_row(
    variant: str,
    model_name: str,
    compression: str,
    metrics: dict[str, float],
    parameters: int,
    artifact_path: Path,
    benchmark: dict[str, float],
    config: dict[str, Any],
    num_classes: int,
) -> dict[str, Any]:
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "model": model_name,
        "teacher_model": "",
        "compression": compression,
        "seed": config["seed"],
        "num_classes": num_classes,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "baseline_accuracy": "",
        "accuracy_drop": "",
        "baseline_macro_f1": "",
        "macro_f1_drop": "",
        "parameters": parameters,
        "parameter_reduction": "",
        "artifact": str(artifact_path),
        "compression_ratio": "",
        **benchmark,
        "meets_accuracy_budget": "",
        "meets_size_budget": benchmark["model_size_mb"] <= float(config["max_model_size_mb"]),
        "meets_latency_budget": benchmark["median_latency_ms"] <= float(config["max_median_latency_ms"]),
        "meets_all_budgets": "",
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    return row


def enrich_against_baseline(
    config: dict[str, Any],
    row: dict[str, Any],
    teacher_model: str,
) -> dict[str, Any]:
    path = Path(config["output_dir"]) / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError("Run the matching FP32 baseline before a compression experiment")
    table = pd.read_csv(path)
    candidates = table[(table["variant"] == "baseline") & (table["model"] == teacher_model)]
    if candidates.empty:
        raise RuntimeError(f"No FP32 baseline result found for {teacher_model}")
    baseline = candidates.iloc[-1]
    row["teacher_model"] = teacher_model
    row["baseline_accuracy"] = float(baseline["accuracy"])
    row["accuracy_drop"] = float(baseline["accuracy"]) - float(row["accuracy"])
    row["baseline_macro_f1"] = float(baseline["macro_f1"])
    row["macro_f1_drop"] = float(baseline["macro_f1"]) - float(row["macro_f1"])
    row["compression_ratio"] = float(baseline["model_size_mb"]) / float(row["model_size_mb"])
    row["meets_accuracy_budget"] = row["accuracy_drop"] <= float(config["max_accuracy_drop"])
    row["meets_all_budgets"] = bool(
        row["meets_accuracy_budget"]
        and row["meets_size_budget"]
        and row["meets_latency_budget"]
    )
    return row


def append_result(config: dict[str, Any], row: dict[str, Any]) -> None:
    path = Path(config["output_dir"]) / "metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def command_eda(config: dict[str, Any]) -> None:
    report = audit_splits(config["data_dir"])
    path = Path(config["output_dir"]) / "eda_and_leakage_audit.json"
    save_json(path, report)
    records = [
        {"split": split, "class": class_name, "count": count}
        for split, counts in report["class_counts"].items()
        for class_name, count in counts.items()
    ]
    counts_table = pd.DataFrame.from_records(records)
    counts_table.to_csv(Path(config["output_dir"]) / "class_counts.csv", index=False)
    pivot = counts_table.pivot(index="class", columns="split", values="count").fillna(0)
    axis = pivot.plot(kind="bar", stacked=False, figsize=(18, 7), title="Images per class and split")
    axis.set_ylabel("Image count")
    axis.figure.tight_layout()
    axis.figure.savefig(Path(config["output_dir"]) / "class_distribution.png", dpi=160)
    plt.close(axis.figure)
    print(json.dumps(report["totals"], indent=2))
    duplicates = report["cross_split_exact_duplicate_count"]
    if duplicates:
        raise RuntimeError(f"Found {duplicates} exact duplicate groups crossing data splits")
    print(f"Leakage audit passed; report written to {path}")


def command_train(config: dict[str, Any], model_name: str) -> None:
    loaders, classes = build_dataloaders(config)
    model = build_model(model_name, len(classes), pretrained=True)
    set_backbone_trainable(model, False)
    model, head_history = train_supervised(
        model,
        loaders["train"],
        loaders["val"],
        device(),
        int(config["linear_head_epochs"]),
        float(config["learning_rate"]),
        float(config["weight_decay"]),
        config["max_train_batches"],
        config["max_eval_batches"],
    )
    history = [{"stage": "linear_head", **row} for row in head_history]
    remaining_epochs = max(0, int(config["epochs"]) - int(config["linear_head_epochs"]))
    if remaining_epochs:
        set_backbone_trainable(model, True)
        model, fine_tune_history = train_supervised(
            model,
            loaders["train"],
            loaders["val"],
            device(),
            remaining_epochs,
            float(config["learning_rate"]) / 10.0,
            float(config["weight_decay"]),
            config["max_train_batches"],
            config["max_eval_batches"],
        )
        history.extend({"stage": "fine_tune", **row} for row in fine_tune_history)
    test_metrics = evaluate(model.to(device()), loaders["test"], device(), config["max_eval_batches"])
    out = output_directory(config, model_name)
    checkpoint = out / "baseline.pt"
    torch.save(checkpoint_payload(model, model_name, classes), checkpoint)
    onnx_path = export_onnx(model, out / "baseline_fp32.onnx", int(config["image_size"]))
    benchmark = benchmark_onnx(
        onnx_path,
        int(config["image_size"]),
        int(config["latency_warmup"]),
        int(config["latency_runs"]),
        int(config["cpu_threads"]),
    )
    save_json(out / "baseline_history.json", history)
    row = result_row("baseline", model_name, "none_fp32", test_metrics, parameter_count(model), onnx_path, benchmark, config, len(classes))
    append_result(config, row)
    print(json.dumps(row, indent=2))


def command_quantize(config: dict[str, Any], checkpoint: str) -> None:
    loaders, _ = build_dataloaders(config)
    model, model_name, classes = load_checkpoint(checkpoint)
    out = output_directory(config, model_name)
    fp32_path = export_onnx(model, out / "baseline_fp32.onnx", int(config["image_size"]))
    generator = torch.Generator().manual_seed(int(config["seed"]))
    calibration_loader = DataLoader(
        loaders["val"].dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        generator=generator,
    )
    int8_path = static_int8_quantize(
        fp32_path,
        out / "baseline_int8.onnx",
        calibration_loader,
        int(config["calibration_batches"]),
    )
    metrics = evaluate_onnx(int8_path, loaders["test"], config["max_eval_batches"])
    benchmark = benchmark_onnx(
        int8_path,
        int(config["image_size"]),
        int(config["latency_warmup"]),
        int(config["latency_runs"]),
        int(config["cpu_threads"]),
    )
    row = result_row("quantized", model_name, "onnx_static_int8", metrics, parameter_count(model), int8_path, benchmark, config, len(classes))
    row = enrich_against_baseline(config, row, model_name)
    append_result(config, row)
    print(json.dumps(row, indent=2))


def command_prune(config: dict[str, Any], checkpoint: str) -> None:
    loaders, _ = build_dataloaders(config)
    model, model_name, classes = load_checkpoint(checkpoint)
    original_parameters = parameter_count(model)
    model = structured_prune(model, int(config["image_size"]), float(config["pruning_ratio"]))
    model, history = train_supervised(
        model,
        loaders["train"],
        loaders["val"],
        device(),
        int(config["fine_tune_epochs"]),
        float(config["learning_rate"]) / 10.0,
        float(config["weight_decay"]),
        config["max_train_batches"],
        config["max_eval_batches"],
    )
    metrics = evaluate(model.to(device()), loaders["test"], device(), config["max_eval_batches"])
    out = output_directory(config, model_name)
    artifact = export_onnx(model, out / "structured_pruned_fp32.onnx", int(config["image_size"]))
    benchmark = benchmark_onnx(artifact, int(config["image_size"]), int(config["latency_warmup"]), int(config["latency_runs"]), int(config["cpu_threads"]))
    row = result_row("pruned", model_name, "structured_channel_l2", metrics, parameter_count(model), artifact, benchmark, config, len(classes))
    row["parameter_reduction"] = 1.0 - parameter_count(model) / original_parameters
    row = enrich_against_baseline(config, row, model_name)
    save_json(out / "pruning_history.json", history)
    append_result(config, row)
    print(json.dumps(row, indent=2))


def command_distill(config: dict[str, Any], checkpoint: str) -> None:
    loaders, classes = build_dataloaders(config)
    teacher, teacher_name, checkpoint_classes = load_checkpoint(checkpoint)
    if checkpoint_classes != classes:
        raise ValueError("Checkpoint classes do not match dataset classes")
    student_name = "mobilenet_v3_small_050"
    student = build_model(student_name, len(classes), pretrained=False)
    student, history = train_distillation(
        teacher,
        student,
        loaders["train"],
        loaders["val"],
        device(),
        int(config["epochs"]),
        float(config["learning_rate"]),
        float(config["weight_decay"]),
        float(config["distillation_alpha"]),
        float(config["distillation_temperature"]),
        config["max_train_batches"],
        config["max_eval_batches"],
    )
    metrics = evaluate(student.to(device()), loaders["test"], device(), config["max_eval_batches"])
    out = output_directory(config, student_name)
    checkpoint_path = out / "distilled_student.pt"
    torch.save(checkpoint_payload(student, student_name, classes), checkpoint_path)
    artifact = export_onnx(student, out / "distilled_student_fp32.onnx", int(config["image_size"]))
    benchmark = benchmark_onnx(artifact, int(config["image_size"]), int(config["latency_warmup"]), int(config["latency_runs"]), int(config["cpu_threads"]))
    row = result_row("distilled", student_name, f"kd_from_{teacher_name}", metrics, parameter_count(student), artifact, benchmark, config, len(classes))
    row = enrich_against_baseline(config, row, teacher_name)
    save_json(out / "distillation_history.json", history)
    append_result(config, row)
    print(json.dumps(row, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RQ7.1 model-compression experiments")
    parser.add_argument("command", choices=("eda", "train", "quantize", "prune", "distill"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="mobilenet_v3_small")
    parser.add_argument("--checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.command == "eda":
        command_eda(config)
    elif args.command == "train":
        command_train(config, args.model)
    else:
        if not args.checkpoint:
            raise SystemExit(f"--checkpoint is required for {args.command}")
        {"quantize": command_quantize, "prune": command_prune, "distill": command_distill}[args.command](config, args.checkpoint)


if __name__ == "__main__":
    main()
