from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
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
from rq71.data import audit_splits, build_dataloaders, quarantine_cross_split_duplicates  # noqa: E402
from rq71.engine import evaluate, train_distillation, train_supervised  # noqa: E402
from rq71.models import SUPPORTED_MODELS, build_model, set_backbone_trainable  # noqa: E402
from rq71.selection import PruningAccuracyGateError, choose_pruning


class QuantizationAccuracyGateError(RuntimeError):
    """Validated INT8 candidates exist, but none passed the accuracy gate."""


RESULT_FIELDS = [
    "run_id",
    "config_hash",
    "platform",
    "cpu",
    "gpu",
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


def config_hash(config: dict[str, Any]) -> str:
    public_config = {key: value for key, value in config.items() if not str(key).startswith("_")}
    payload = json.dumps(public_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def hardware_metadata() -> dict[str, str]:
    cpu = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[-1].strip()
                break
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return {"platform": platform.platform(), "cpu": cpu, "gpu": gpu}


def output_directory(config: dict[str, Any], model_name: str) -> Path:
    path = Path(config["output_dir"]) / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def plot_history(history: list[dict[str, Any]], output: Path, title: str) -> None:
    if not history:
        return
    table = pd.DataFrame(history)
    table.to_csv(output.with_suffix(".csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = list(range(1, len(table) + 1))
    is_kd = 'distillation' in output.name
    if "train_loss" in table:
        if is_kd:
            kd_fig, kd_axis = plt.subplots(figsize=(8, 5))
            kd_axis.plot(x, table['train_loss'], label='KD training objective')
            kd_axis.set(xlabel='Sequential epoch', ylabel='Combined KD objective', title=title)
            kd_axis.legend()
            kd_fig.tight_layout()
            for suffix in ('png', 'pdf'):
                kd_fig.savefig(output.with_name(output.stem + '_objective.' + suffix))
            plt.close(kd_fig)
        else:
            axes[0].plot(x, table["train_loss"], marker="o", label="Train CE")
    if "loss" in table:
        axes[0].plot(x, table["loss"], marker="o", label="Validation CE")
    for i in range(1, len(history)):
        if history[i].get('stage') != history[i-1].get('stage'):
            axes[0].axvline(i+.5, color='gray', linestyle=':')
    axes[0].set(title=title + " loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[0].legend()
    for metric, label in (("accuracy", "Validation accuracy"), ("macro_f1", "Validation macro F1")):
        if metric in table:
            axes[1].plot(x, table[metric], marker="o", label=label)
    for metric in ('train_accuracy', 'train_macro_f1'):
        if metric in table:
            axes[1].plot(x, table[metric], linestyle='--', label=metric.replace('_', ' '))
    axes[1].set(title=title + " validation metrics", xlabel="Epoch", ylabel="Score", ylim=(0, 1.02))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)


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
    cfg_hash = config_hash(config)
    run_id = f"{variant}:{model_name}:{compression}:seed{config['seed']}:{cfg_hash}"
    row = {
        "run_id": run_id,
        "config_hash": cfg_hash,
        **hardware_metadata(),
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
    new = pd.DataFrame([{field: row.get(field, "") for field in RESULT_FIELDS}])
    if path.exists():
        existing = pd.read_csv(path)
        for field in RESULT_FIELDS:
            if field not in existing.columns:
                existing[field] = ""
        legacy_key = (
            existing["variant"].astype(str) + ":" + existing["model"].astype(str) + ":" +
            existing["compression"].astype(str) + ":seed" + existing["seed"].astype(str)
        )
        existing.loc[existing["run_id"].astype(str).isin(("", "nan")), "run_id"] = legacy_key
        combined = pd.concat([existing[RESULT_FIELDS], new], ignore_index=True)
    else:
        combined = new
    combined = combined.drop_duplicates(subset=["run_id"], keep="last")
    combined.to_csv(path, index=False)


def command_eda(config: dict[str, Any]) -> None:
    report = audit_splits(config["data_dir"], config.get("manifest_dir"))
    duplicates = report["cross_split_exact_duplicate_count"]
    if duplicates and not config.get("manifest_dir") and bool(config.get("quarantine_cross_split_duplicates", False)):
        quarantine_dir = Path(config["output_dir"]) / "quarantined_cross_split_duplicates"
        moved = quarantine_cross_split_duplicates(config["data_dir"], quarantine_dir)
        save_json(Path(config["output_dir"]) / "quarantined_cross_split_duplicates.json", moved)
        print(
            f"Quarantined {len(moved)} duplicate image files from {duplicates} "
            f"cross-split duplicate groups into {quarantine_dir}"
        )
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
    plot_history(history, out / "baseline_training_curves.png", f"{model_name} baseline")
    row = result_row("baseline", model_name, "none_fp32", test_metrics, parameter_count(model), onnx_path, benchmark, config, len(classes))
    append_result(config, row)
    print(json.dumps(row, indent=2))


def command_quantize_candidate(
    config: dict[str, Any],
    fp32_path: str,
    output_path: str,
    result_path: str,
    calibration_method: str,
    per_channel: bool,
    activation_type: str,
    op_types: list[str] | None,
    conv_scope: str = "all",
) -> None:
    """Run one INT8 candidate in an isolated process.

    Isolation is deliberate: histogram-based ORT calibrators can retain large
    activation buffers. Exiting this process guarantees that memory is returned
    before the next candidate starts.
    """
    loaders, _ = build_dataloaders(config)
    diagnostic_batch_size = int(config.get("int8_diagnostic_batch_size", 8))
    diagnostic_examples = int(config.get("int8_diagnostic_calibration_examples", 256))
    diagnostic_examples = max(1, min(diagnostic_examples, len(loaders["val"].dataset)))
    calibration_batches = max(1, (diagnostic_examples + diagnostic_batch_size - 1) // diagnostic_batch_size)

    generator = torch.Generator().manual_seed(int(config["seed"]))
    calibration_loader = DataLoader(
        loaders["val"].dataset,
        batch_size=diagnostic_batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    # Check the common preprocessing stage independently of integer conversion.
    # Save evidence before raising so a failed candidate remains diagnosable.
    preprocessing_audit = {}
    nodes_to_exclude: list[str] = []
    if conv_scope not in {"all", "standard_only", "depthwise_only", "early_block_fp32"}:
        raise ValueError(f"Unknown Conv diagnostic scope: {conv_scope}")
    if conv_scope != "all" and (not config.get("int8_diagnostic_only") or op_types != ["Conv"]):
        raise ValueError("Conv category isolation requires diagnostic-only mode and Conv-only operations")
    def validate_prepared(prepared_path):
        import onnx
        from collections import Counter
        graph_counts = dict(Counter(node.op_type for node in onnx.load(str(prepared_path)).graph.node))
        preprocessing_audit["prepared_operator_counts"] = graph_counts
        if conv_scope != "all":
            graph = onnx.load(str(prepared_path)).graph
            weights = {value.name: list(value.dims) for value in graph.initializer}
            shapes = {value.name: [d.dim_value if d.HasField("dim_value") else None for d in value.type.tensor_type.shape.dim]
                      for value in list(graph.input) + list(graph.value_info)}
            audit = []
            for node in graph.node:
                if node.op_type != "Conv":
                    continue
                group = next((int(a.i) for a in node.attribute if a.name == "group"), 1)
                shape = weights.get(node.input[1])
                input_shape = shapes.get(node.input[0], [])
                channels = input_shape[1] if len(input_shape) > 1 else None
                if not node.name:
                    raise ValueError("Conv exclusion requires named prepared nodes")
                if group == 1:
                    category = "standard"
                elif shape and shape[1] == 1 and shape[0] % group == 0 and channels == group:
                    category = "depthwise"
                else:
                    raise ValueError(f"Unclassified grouped Conv: {node.name}")
                if conv_scope == "early_block_fp32":
                    included = not node.name.startswith(("/features/features.0/", "/features/features.1/"))
                else:
                    included = category == conv_scope.removesuffix("_only")
                if not included:
                    nodes_to_exclude.append(node.name)
                audit.append(dict(name=node.name,category=category,group=group,weight_shape=shape,
                                  input_channels=channels,quantize=included))
            counts = Counter(row["category"] for row in audit)
            if dict(counts) != {"standard": 41, "depthwise": 11}:
                raise ValueError(f"Prepared Conv structure differs from audited seed-27 graph: {counts}")
            if len({row["name"] for row in audit}) != len(audit):
                raise ValueError("Prepared Conv names must be unique")
            if conv_scope == "early_block_fp32" and len(nodes_to_exclude) != 5:
                raise ValueError("Expected exactly five stem/first-block Conv exclusions")
            preprocessing_audit.update(conv_scope=conv_scope,conv_nodes=audit,nodes_to_exclude=list(nodes_to_exclude))
            print(f"PREPARED CONV SCOPE {conv_scope}: quantise={sum(row['quantize'] for row in audit)} exclude={len(nodes_to_exclude)}",flush=True)
        raw = evaluate_onnx(fp32_path, loaders["val"], config["max_eval_batches"])
        prepared = evaluate_onnx(prepared_path, loaders["val"], config["max_eval_batches"])
        gap = abs(raw["accuracy"] - prepared["accuracy"])
        preprocessing_audit.update(raw_fp32=raw, prepared_fp32=prepared, accuracy_gap=gap)
        save_json(Path(result_path).with_suffix(".preprocessing.json"), preprocessing_audit)
        if gap > float(config.get("max_fp32_onnx_accuracy_gap", 0.002)):
            raise RuntimeError(f"Preprocessing parity failed before INT8 calibration: gap={gap:.6f}")

    print(f"Calibration source=validation; requested={diagnostic_examples}; batch_size={diagnostic_batch_size}; seed={config['seed']}", flush=True)
    artifact = static_int8_quantize(
        fp32_path,
        output_path,
        calibration_loader,
        calibration_batches,
        calibration_method=calibration_method,
        per_channel=per_channel,
        activation_type=activation_type,
        op_types_to_quantize=op_types,
        prepared_validator=validate_prepared,
        nodes_to_exclude=nodes_to_exclude if conv_scope != "all" else None,
    )
    metrics = evaluate_onnx(artifact, loaders["val"], config["max_eval_batches"])
    import onnx
    from collections import Counter
    quantized_graph = onnx.load(str(artifact))
    operator_counts = dict(Counter(node.op_type for node in quantized_graph.graph.node))
    matching_ops = sum(preprocessing_audit["prepared_operator_counts"].get(op, 0) for op in op_types) if op_types else None
    save_json(
        Path(result_path),
        {
            "calibration_method": calibration_method,
            "per_channel": per_channel,
            "activation_type": activation_type,
            "op_types_to_quantize": op_types,
            "conv_scope": conv_scope,
            "nodes_to_exclude": nodes_to_exclude,
            "calibration_examples_requested": diagnostic_examples,
            "calibration_batch_size": diagnostic_batch_size,
            "artifact": str(artifact),
            "model_size_mb": Path(artifact).stat().st_size / (1024 * 1024),
            "prepared_matching_operator_count": matching_ops,
            "no_matching_operators": matching_ops == 0,
            "result_operator_counts": operator_counts,
            "quantized_initializer_count": sum(value.data_type in (onnx.TensorProto.INT8, onnx.TensorProto.UINT8) for value in quantized_graph.graph.initializer),
            "validation_metrics": metrics,
            "preprocessing_audit": preprocessing_audit,
        },
    )


def command_quantize(config: dict[str, Any], checkpoint: str) -> None:
    loaders, _ = build_dataloaders(config)
    model, model_name, classes = load_checkpoint(checkpoint)
    out = output_directory(config, model_name)

    # Model/configuration selection is validation-only. The test split is not
    # evaluated until one INT8 candidate has passed the validation sanity gate.
    fp32_path = export_onnx(model, out / "baseline_fp32.onnx", int(config["image_size"]))
    pytorch_val_metrics = evaluate(model.to(device()), loaders["val"], device(), config["max_eval_batches"])
    fp32_val_metrics = evaluate_onnx(fp32_path, loaders["val"], config["max_eval_batches"])
    fp32_gap = abs(pytorch_val_metrics["accuracy"] - fp32_val_metrics["accuracy"])
    allowed_fp32_gap = float(config.get("max_fp32_onnx_accuracy_gap", 0.002))
    if fp32_gap > allowed_fp32_gap:
        raise RuntimeError(
            f"FP32 ONNX validation parity failed: PyTorch={pytorch_val_metrics['accuracy']:.6f}, "
            f"ONNX={fp32_val_metrics['accuracy']:.6f}, gap={fp32_gap:.6f}"
        )

    # Each candidate runs in its own subprocess. This prevents Entropy/
    # Percentile calibration memory from accumulating across candidates and
    # lets the sweep continue even if one candidate is killed by the OS.
    diagnostic_grid = config.get(
        "int8_diagnostic_grid",
        [
            {"calibration_method": "minmax", "per_channel": True, "activation_type": "quint8"},
            {"calibration_method": "minmax", "per_channel": True, "activation_type": "qint8"},
            {"calibration_method": "minmax", "per_channel": False, "activation_type": "quint8"},
            {"calibration_method": "percentile", "per_channel": True, "activation_type": "quint8"},
            {"calibration_method": "entropy", "per_channel": True, "activation_type": "quint8"},
            {
                "calibration_method": "minmax",
                "per_channel": True,
                "activation_type": "quint8",
                "op_types_to_quantize": ["Conv", "Gemm"],
            },
        ],
    )

    candidate_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    script_path = Path(__file__).resolve()
    for index, setting in enumerate(diagnostic_grid, start=1):
        method = str(setting.get("calibration_method", "minmax")).lower()
        per_channel = bool(setting.get("per_channel", True))
        activation_type = str(setting.get("activation_type", "quint8")).lower()
        op_types = setting.get("op_types_to_quantize")
        conv_scope = str(setting.get("conv_scope", "all"))
        label = f"c{index}_{method}_{activation_type}_{'pc' if per_channel else 'pt'}"
        if op_types:
            label += "_" + "-".join(str(x).lower() for x in op_types)
        if conv_scope != "all":
            label += "_" + conv_scope
        candidate_path = out / f"int8_{label}.onnx"
        candidate_json = out / f"int8_{label}.json"
        if candidate_json.exists():
            candidate_json.unlink()

        cmd = [
            sys.executable, str(script_path), "quantize-candidate",
            "--config", str(config["_config_path"]),
            "--fp32-path", str(fp32_path),
            "--candidate-output", str(candidate_path),
            "--candidate-result", str(candidate_json),
            "--calibration-method", method,
            "--activation-type", activation_type,
            "--per-channel", "1" if per_channel else "0",
        ]
        if op_types:
            cmd.extend(["--op-types", ",".join(str(x) for x in op_types)])
        cmd.extend(["--conv-scope", conv_scope])

        completed = subprocess.run(cmd, text=True, capture_output=True)
        if completed.returncode != 0 or not candidate_json.exists():
            failure = {
                "candidate": index,
                "calibration_method": method,
                "per_channel": per_channel,
                "activation_type": activation_type,
                "op_types_to_quantize": op_types,
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-2000:],
            }
            failures.append(failure)
            print(
                f"INT8 candidate {index}: FAILED (returncode={completed.returncode}) "
                f"method={method}, activation={activation_type}, per_channel={per_channel}, op_types={op_types}"
            )
            continue

        result = json.loads(candidate_json.read_text(encoding="utf-8"))
        result["validation_accuracy_drop"] = fp32_val_metrics["accuracy"] - result["validation_metrics"]["accuracy"]
        result["validation_macro_f1_drop"] = fp32_val_metrics["macro_f1"] - result["validation_metrics"]["macro_f1"]
        candidate_results.append(result)
        print(
            f"INT8 candidate {index}: method={method}, activation={activation_type}, "
            f"per_channel={per_channel}, op_types={op_types}, "
            f"val_accuracy={result['validation_metrics']['accuracy']:.6f}, "
            f"val_macro_f1={result['validation_metrics']['macro_f1']:.6f}"
        )

    diagnostic_path = out / "quantization_diagnostics.json"
    diagnostic = {
        "diagnostic_only": bool(config.get("int8_diagnostic_only", False)),
        "fp32_model_size_mb": fp32_path.stat().st_size / (1024 * 1024),
        "selection_split": "validation",
        "pytorch_fp32_validation": pytorch_val_metrics,
        "onnx_fp32_validation": fp32_val_metrics,
        "fp32_parity_gap": fp32_gap,
        "diagnostic_calibration_examples": int(config.get("int8_diagnostic_calibration_examples", 256)),
        "diagnostic_batch_size": int(config.get("int8_diagnostic_batch_size", 8)),
        "candidates": candidate_results,
        "failed_candidates": failures,
    }

    if config.get("int8_diagnostic_only", False):
        diagnostic["status"] = "diagnostic_complete" if len(candidate_results) == len(diagnostic_grid) else "diagnostic_incomplete"
        diagnostic["test_evaluated"] = False
        diagnostic["selection_performed"] = False
        save_json(diagnostic_path, diagnostic)
        print(f"VALIDATION-ONLY DIAGNOSTIC: {len(candidate_results)}/{len(diagnostic_grid)} candidates; {diagnostic_path}", flush=True)
        if failures:
            raise RuntimeError("Incomplete INT8 diagnostic; inspect saved candidate failures")
        return

    if not candidate_results:
        save_json(diagnostic_path, diagnostic)
        raise RuntimeError(
            f"Every INT8 diagnostic candidate failed before validation. Test split was not evaluated. "
            f"See {diagnostic_path}."
        )

    selected = max(
        candidate_results,
        key=lambda row: (row["validation_metrics"]["macro_f1"], row["validation_metrics"]["accuracy"]),
    )
    diagnostic["selected"] = selected
    save_json(diagnostic_path, diagnostic)

    allowed_int8_drop = float(config.get("max_int8_sanity_accuracy_drop", 0.10))
    validation_drop = float(selected["validation_accuracy_drop"])
    if validation_drop > allowed_int8_drop:
        diagnostic["status"] = "ineligible_accuracy_gate"
        diagnostic["test_evaluated"] = False
        diagnostic["allowed_accuracy_drop"] = allowed_int8_drop
        save_json(diagnostic_path, diagnostic)
        raise QuantizationAccuracyGateError(
            f"INT8 validation sanity gate failed after isolated diagnostic sweep: "
            f"FP32 ONNX={fp32_val_metrics['accuracy']:.6f}, "
            f"best INT8={selected['validation_metrics']['accuracy']:.6f}, "
            f"drop={validation_drop:.6f}. Test split was not evaluated and invalid result "
            f"was not added to metrics.csv. See {diagnostic_path}."
        )

    # Only the selected configuration is allowed to touch the test split.
    selected_path = Path(selected["artifact"])
    pytorch_test_metrics = evaluate(model.to(device()), loaders["test"], device(), config["max_eval_batches"])
    fp32_test_metrics = evaluate_onnx(fp32_path, loaders["test"], config["max_eval_batches"])
    int8_test_metrics = evaluate_onnx(selected_path, loaders["test"], config["max_eval_batches"])
    test_drop = fp32_test_metrics["accuracy"] - int8_test_metrics["accuracy"]

    final_path = out / "baseline_int8.onnx"
    if selected_path != final_path:
        import shutil
        shutil.copy2(selected_path, final_path)

    validation = {
        "selection_split": "validation",
        "selected_configuration": {
            "calibration_method": selected["calibration_method"],
            "per_channel": selected["per_channel"],
            "activation_type": selected["activation_type"],
            "op_types_to_quantize": selected.get("op_types_to_quantize"),
        },
        "pytorch_fp32_validation": pytorch_val_metrics,
        "onnx_fp32_validation": fp32_val_metrics,
        "selected_int8_validation": selected["validation_metrics"],
        "pytorch_fp32_test": pytorch_test_metrics,
        "onnx_fp32_test": fp32_test_metrics,
        "onnx_int8_test": int8_test_metrics,
        "fp32_parity_gap_validation": fp32_gap,
        "int8_validation_accuracy_drop": validation_drop,
        "int8_test_accuracy_drop": test_drop,
    }
    save_json(out / "quantization_validation.json", validation)

    benchmark = benchmark_onnx(
        final_path,
        int(config["image_size"]),
        int(config["latency_warmup"]),
        int(config["latency_runs"]),
        int(config["cpu_threads"]),
    )
    op_label = "allops" if not selected.get("op_types_to_quantize") else "-".join(selected["op_types_to_quantize"])
    compression_name = (
        f"onnx_static_int8_qdq_{selected['calibration_method']}_{selected['activation_type']}_"
        f"{'per_channel' if selected['per_channel'] else 'per_tensor'}_{op_label}"
    )
    row = result_row(
        "quantized", model_name, compression_name, int8_test_metrics,
        parameter_count(model), final_path, benchmark, config, len(classes)
    )
    row = enrich_against_baseline(config, row, model_name)
    append_result(config, row)
    print(json.dumps(row, indent=2))

def command_prune(config: dict[str, Any], checkpoint: str) -> None:
    # Pruning-ratio tuning is validation-only. The test split is reserved for
    # the single ratio selected later by command_prune_select.
    loaders, _ = build_dataloaders(config)
    baseline_model, model_name, classes = load_checkpoint(checkpoint)
    baseline_val = evaluate(baseline_model.to(device()), loaders["val"], device(), config["max_eval_batches"])
    original_parameters = parameter_count(baseline_model)

    ratio = float(config["pruning_ratio"])
    model, _, _ = load_checkpoint(checkpoint)
    model = structured_prune(model, int(config["image_size"]), ratio)
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
    val_metrics = evaluate(model.to(device()), loaders["val"], device(), config["max_eval_batches"])
    out = output_directory(config, model_name)
    tag = f"r{ratio:.2f}"
    checkpoint_path = out / f"structured_pruned_{tag}.pt"
    torch.save(checkpoint_payload(model, model_name, classes), checkpoint_path)
    artifact = export_onnx(model, out / f"structured_pruned_{tag}_fp32.onnx", int(config["image_size"]))
    benchmark = benchmark_onnx(artifact, int(config["image_size"]), int(config["latency_warmup"]), int(config["latency_runs"]), int(config["cpu_threads"]))
    candidate = {
        "selection_split": "validation",
        "model": model_name,
        "pruning_ratio": ratio,
        "baseline_validation": baseline_val,
        "validation_metrics": val_metrics,
        "validation_accuracy_drop": baseline_val["accuracy"] - val_metrics["accuracy"],
        "validation_macro_f1_drop": baseline_val["macro_f1"] - val_metrics["macro_f1"],
        "parameters": parameter_count(model),
        "parameter_reduction": 1.0 - parameter_count(model) / original_parameters,
        "checkpoint": str(checkpoint_path),
        "artifact": str(artifact),
        **benchmark,
    }
    save_json(out / f"pruning_candidate_{tag}.json", candidate)
    save_json(out / f"pruning_history_{tag}.json", history)
    plot_history(history, out / f"pruning_training_curves_{tag}.png", f"{model_name} structured pruning {tag}")
    print(json.dumps(candidate, indent=2))


def command_prune_select(config: dict[str, Any], checkpoint: str) -> None:
    # Select the most aggressive ratio that satisfies the validation accuracy
    # budget, then evaluate that one candidate on test exactly once.
    loaders, _ = build_dataloaders(config)
    _, model_name, classes = load_checkpoint(checkpoint)
    out = output_directory(config, model_name)
    candidate_files = sorted(out.glob("pruning_candidate_r*.json"))
    if not candidate_files:
        raise RuntimeError("No pruning candidates found. Run the pruning ablations first.")
    candidates = [load_config(path) for path in candidate_files]
    allowed = float(config["max_accuracy_drop"])
    try:
        selected = choose_pruning(candidates, allowed)
    except PruningAccuracyGateError:
        save_json(out / 'pruning_selection.json', dict(status='ineligible_accuracy_gate',
            max_validation_accuracy_drop=allowed, candidates=candidates,
            selected=None, test_evaluated=False))
        raise

    # The structurally pruned PyTorch model cannot be reconstructed using the
    # original unpruned EfficientNet architecture because channel dimensions
    # have changed. Use the already-exported ONNX candidate for final testing.
    artifact = Path(selected["artifact"])
    if not artifact.exists():
        raise FileNotFoundError(f"Selected pruning artifact not found: {artifact}")

    test_metrics = evaluate_onnx(
        artifact,
        loaders["test"],
        config["max_eval_batches"],
    )
    benchmark = benchmark_onnx(artifact, int(config["image_size"]), int(config["latency_warmup"]), int(config["latency_runs"]), int(config["cpu_threads"]))
    ratio = float(selected["pruning_ratio"])
    compression_name = f"structured_channel_l2_r{ratio:.2f}"
    row = result_row("pruned", model_name, compression_name, test_metrics, int(selected["parameters"]), artifact, benchmark, config, len(classes))
    row["parameter_reduction"] = float(selected["parameter_reduction"])
    row = enrich_against_baseline(config, row, model_name)
    append_result(config, row)
    summary = {
        "selection_rule": "highest pruning ratio within validation accuracy-drop budget",
        "max_validation_accuracy_drop": allowed,
        "candidates": candidates,
        "selected": selected,
        "final_test_result": row,
    }
    save_json(out / "pruning_selection.json", summary)
    print(json.dumps(summary, indent=2))


def command_distill(config: dict[str, Any], checkpoint: str) -> None:
    """Compare a small CE-only student with the same student trained using KD.

    Both students are assessed on validation first. Test evaluation is
    performed only after training/validation is complete so that test data
    does not influence training or model selection.
    """
    loaders, classes = build_dataloaders(config)

    teacher, teacher_name, checkpoint_classes = load_checkpoint(checkpoint)
    if checkpoint_classes != classes:
        raise ValueError("Checkpoint classes do not match dataset classes")

    student_name = "mobilenet_v3_small_050"
    # Keep the two teacher experiments independent.  Without the teacher
    # subdirectory, the second track would overwrite the first track's
    # student checkpoints, histories, ONNX files, and comparison JSON.
    out = output_directory(config, student_name) / f"teacher_{teacher_name}"
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. CONTROL: small student trained normally with cross-entropy
    # ---------------------------------------------------------------
    print("\n=== Training CE-only student control ===")

    control = build_model(
        student_name,
        len(classes),
        pretrained=False,
    )

    control, control_history = train_supervised(
        control,
        loaders["train"],
        loaders["val"],
        device(),
        int(config["epochs"]),
        float(config["learning_rate"]),
        float(config["weight_decay"]),
        config["max_train_batches"],
        config["max_eval_batches"],
    )

    control_val = evaluate(
        control.to(device()),
        loaders["val"],
        device(),
        config["max_eval_batches"],
    )

    control_checkpoint = out / "student_control_ce.pt"
    torch.save(
        checkpoint_payload(control, student_name, classes),
        control_checkpoint,
    )

    control_artifact = export_onnx(
        control,
        out / "student_control_ce_fp32.onnx",
        int(config["image_size"]),
    )

    # ---------------------------------------------------------------
    # 2. KNOWLEDGE-DISTILLED STUDENT
    # ---------------------------------------------------------------
    print("\n=== Training knowledge-distilled student ===")

    student = build_model(
        student_name,
        len(classes),
        pretrained=False,
    )

    student, kd_history = train_distillation(
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

    kd_val = evaluate(
        student.to(device()),
        loaders["val"],
        device(),
        config["max_eval_batches"],
    )

    kd_checkpoint = out / "distilled_student.pt"
    torch.save(
        checkpoint_payload(student, student_name, classes),
        kd_checkpoint,
    )

    kd_artifact = export_onnx(
        student,
        out / "distilled_student_fp32.onnx",
        int(config["image_size"]),
    )

    # ---------------------------------------------------------------
    # 3. Validation comparison
    # ---------------------------------------------------------------
    validation_comparison = {
        "selection_split": "validation",
        "teacher_model": teacher_name,
        "student_model": student_name,
        "control_validation": control_val,
        "distilled_validation": kd_val,
        "validation_accuracy_gain_from_kd":
            float(kd_val["accuracy"]) - float(control_val["accuracy"]),
        "validation_macro_f1_gain_from_kd":
            float(kd_val["macro_f1"]) - float(control_val["macro_f1"]),
        "distillation_alpha": float(config["distillation_alpha"]),
        "distillation_temperature":
            float(config["distillation_temperature"]),
    }

    save_json(
        out / "distillation_validation_comparison.json",
        validation_comparison,
    )

    print("\n=== Validation comparison ===")
    print(json.dumps(validation_comparison, indent=2))

    # ---------------------------------------------------------------
    # 4. Final test evaluation
    #
    # Both are pre-planned experimental variants. The validation
    # comparison above is completed before test evaluation.
    # ---------------------------------------------------------------
    print("\n=== Final test evaluation ===")

    control_test = evaluate(
        control.to(device()),
        loaders["test"],
        device(),
        config["max_eval_batches"],
    )

    kd_test = evaluate(
        student.to(device()),
        loaders["test"],
        device(),
        config["max_eval_batches"],
    )

    # ---------------------------------------------------------------
    # 5. Deployment measurements
    # ---------------------------------------------------------------
    control_benchmark = benchmark_onnx(
        control_artifact,
        int(config["image_size"]),
        int(config["latency_warmup"]),
        int(config["latency_runs"]),
        int(config["cpu_threads"]),
    )

    kd_benchmark = benchmark_onnx(
        kd_artifact,
        int(config["image_size"]),
        int(config["latency_warmup"]),
        int(config["latency_runs"]),
        int(config["cpu_threads"]),
    )

    control_row = result_row(
        "student_control",
        student_name,
        f"ce_only_control_for_{teacher_name}",
        control_test,
        parameter_count(control),
        control_artifact,
        control_benchmark,
        config,
        len(classes),
    )

    control_row = enrich_against_baseline(
        config,
        control_row,
        teacher_name,
    )

    compression_name = (
        f"kd_from_{teacher_name}"
        f"_a{float(config['distillation_alpha']):.2f}"
        f"_t{float(config['distillation_temperature']):.1f}"
    )

    kd_row = result_row(
        "distilled",
        student_name,
        compression_name,
        kd_test,
        parameter_count(student),
        kd_artifact,
        kd_benchmark,
        config,
        len(classes),
    )

    kd_row = enrich_against_baseline(
        config,
        kd_row,
        teacher_name,
    )

    # Direct KD-vs-control effects
    kd_row["control_accuracy"] = float(control_test["accuracy"])
    kd_row["accuracy_gain_vs_control"] = (
        float(kd_test["accuracy"]) -
        float(control_test["accuracy"])
    )
    kd_row["control_macro_f1"] = float(control_test["macro_f1"])
    kd_row["macro_f1_gain_vs_control"] = (
        float(kd_test["macro_f1"]) -
        float(control_test["macro_f1"])
    )

    # ---------------------------------------------------------------
    # 6. Save histories/results
    # ---------------------------------------------------------------
    save_json(
        out / "student_control_history.json",
        control_history,
    )

    save_json(
        out / "distillation_history.json",
        kd_history,
    )

    plot_history(
        control_history,
        out / "student_control_training_curves.png",
        f"{student_name} CE-only control",
    )

    plot_history(
        kd_history,
        out / "distillation_training_curves.png",
        f"{student_name} distillation",
    )

    append_result(config, control_row)
    append_result(config, kd_row)

    final_summary = {
        "teacher_model": teacher_name,
        "student_model": student_name,
        "validation_comparison": validation_comparison,
        "control_test_result": control_row,
        "distilled_test_result": kd_row,
    }

    save_json(
        out / "distillation_comparison.json",
        final_summary,
    )

    print("\n=== Final distillation comparison ===")
    print(json.dumps(final_summary, indent=2))


def command_full(config: dict[str, Any]) -> None:
    """Run the complete RQ7.1 experiment in the predefined order.

    Selection decisions use validation data only:
      1. EDA / leakage audit
      2. Train both FP32 baselines
      3. Run an independent compression track for MobileNetV3-Small
      4. Run an independent compression track for EfficientNet-B0
      5. Generate one comparative report covering both architectures

    The distillation settings are taken directly from the configuration.
    The optional distillation_grid is deliberately not searched here so
    that a reproduction run does not introduce post-hoc KD tuning.
    """

    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "full_run_manifest.json"

    manifest: dict[str, Any] = {
        "status": "running",
        "seed": int(config["seed"]),
        "config_hash": config_hash(config),
        "stages": [],
        "selection_policy": {
            "architecture_comparison": (
                "no reference-model selection; apply every compression "
                "method independently to both FP32 baselines"
            ),
            "int8": "highest validation macro-F1, then validation accuracy",
            "pruning": (
                "highest pruning ratio within validation "
                "accuracy-drop budget"
            ),
            "distillation": (
                "fixed preconfigured alpha and temperature; "
                "no distillation-grid search"
            ),
        },
        "distillation_alpha": float(
            config["distillation_alpha"]
        ),
        "distillation_temperature": float(
            config["distillation_temperature"]
        ),
        "pruning_ratios": [
            float(x)
            for x in config.get(
                "pruning_ratios",
                [0.1, 0.2, 0.3],
            )
        ],
    }

    def record(stage: str, **details: Any) -> None:
        manifest["stages"].append(
            {
                "stage": stage,
                **details,
            }
        )
        save_json(manifest_path, manifest)

    try:
        # ----------------------------------------------------
        # 1. EDA / leakage gate
        # ----------------------------------------------------
        print("\n" + "=" * 80)
        print("FULL RUN — STAGE 1: EDA / LEAKAGE AUDIT")
        print("=" * 80)

        command_eda(config)

        record(
            "eda",
            status="complete",
        )

        # ----------------------------------------------------
        # 2. FP32 baselines
        # ----------------------------------------------------
        baseline_models = [
            "mobilenet_v3_small",
            "efficientnet_b0",
        ]

        for model_name in baseline_models:

            print("\n" + "=" * 80)
            print(
                f"FULL RUN — BASELINE: {model_name}"
            )
            print("=" * 80)

            # Reset before each architecture so order does not give either
            # baseline a different random initialisation/data-order budget.
            set_seed(int(config["seed"]))
            command_train(
                config,
                model_name,
            )

            record(
                f"baseline_{model_name}",
                status="complete",
            )

        # ----------------------------------------------------
        # 3–4. Parallel compression tracks
        # ----------------------------------------------------
        compression_plan = {
            "policy": "compress both baselines independently",
            "models": baseline_models,
            "methods": ["static_int8", "structured_pruning", "knowledge_distillation"],
            "baseline_comparison": "each compressed result is compared with its matching FP32 baseline",
        }
        save_json(root / "compression_plan.json", compression_plan)
        record("compression_plan", status="complete", **compression_plan)

        original_pruning_ratio = config.get("pruning_ratio")
        ratios = [
            float(x)
            for x in config.get("pruning_ratios", [0.1, 0.2, 0.3])
        ]

        for model_name in baseline_models:
            teacher_checkpoint = output_directory(config, model_name) / "baseline.pt"
            if not teacher_checkpoint.exists():
                raise FileNotFoundError(teacher_checkpoint)

            print("\n" + "=" * 80)
            print(f"FULL RUN — COMPRESSION TRACK: {model_name}")
            print("=" * 80)

            set_seed(int(config["seed"]))
            try:
                command_quantize(config, str(teacher_checkpoint))
            except QuantizationAccuracyGateError as exc:
                # Only a known accuracy rejection is recoverable. Data,
                # parity, export and execution errors must still fail fast.
                record(
                    f"quantization_{model_name}", status="ineligible_accuracy_gate",
                    model=model_name, error=str(exc), test_evaluated=False,
                    diagnostics=str(output_directory(config, model_name) / "quantization_diagnostics.json"),
                )
                print(f"INT8 INELIGIBLE — continuing independent methods for {model_name}: {exc}", flush=True)
            else:
                record(f"quantization_{model_name}", status="complete", model=model_name)

            for ratio in ratios:
                print("\n" + "=" * 80)
                print(f"FULL RUN — {model_name} STRUCTURED PRUNING {ratio:.0%}")
                print("=" * 80)
                candidate_config = dict(config)
                candidate_config["pruning_ratio"] = ratio
                set_seed(int(config["seed"]))
                command_prune(candidate_config, str(teacher_checkpoint))
                record(
                    f"pruning_{model_name}_{ratio:.2f}",
                    status="complete",
                    model=model_name,
                    pruning_ratio=ratio,
                )

            set_seed(int(config["seed"]))
            try:
                command_prune_select(config, str(teacher_checkpoint))
            except PruningAccuracyGateError as exc:
                record(f'pruning_selection_{model_name}', status='ineligible_accuracy_gate',
                       model=model_name, error=str(exc), test_evaluated=False)
                print(f'PRUNING INELIGIBLE: {model_name}: {exc}', flush=True)
            else:
                record(f"pruning_selection_{model_name}", status="complete", model=model_name)

            set_seed(int(config["seed"]))
            command_distill(config, str(teacher_checkpoint))
            record(f"distillation_{model_name}", status="complete", model=model_name)

        if original_pruning_ratio is not None:
            config["pruning_ratio"] = original_pruning_ratio

        # ----------------------------------------------------
        # 7. Final report
        # ----------------------------------------------------
        print("\n" + "=" * 80)
        print("FULL RUN — FINAL REPORT")
        print("=" * 80)

        command_report(config)

        record(
            "report",
            status="complete",
        )

        incomplete = [stage for stage in manifest["stages"] if stage.get("status") != "complete"]
        manifest["status"] = "partial" if incomplete else "complete"
        manifest["incomplete_stages"] = [stage["stage"] for stage in incomplete]
        manifest["compression_tracks"] = baseline_models

        save_json(
            manifest_path,
            manifest,
        )

        print("\n" + "=" * 80)
        print(f"RQ7.1 FULL RUN {manifest['status'].upper()}")
        print("=" * 80)

        print(
            f"\nResults directory:\n{root}"
        )

        print(
            f"\nManifest:\n{manifest_path}"
        )

    except Exception as exc:

        manifest["status"] = "failed"
        manifest["error_type"] = (
            type(exc).__name__
        )
        manifest["error"] = str(exc)

        save_json(
            manifest_path,
            manifest,
        )

        print("\n" + "=" * 80)
        print("RQ7.1 FULL RUN FAILED")
        print("=" * 80)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print(
            f"\nFailure recorded in:\n"
            f"{manifest_path}"
        )

        raise



def command_report(config: dict[str, Any]) -> None:
    root = Path(config["output_dir"])
    from report_training_figures import render
    render(root)
    metrics_path = root / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    table = pd.read_csv(metrics_path)
    table = table.sort_values("timestamp_utc").drop_duplicates(
        subset=["variant", "model", "compression", "seed"], keep="last"
    )
    table["valid_result"] = True
    invalid_int8 = (
        table["variant"].eq("quantized")
        & table["accuracy_drop"].fillna(0).gt(float(config.get("max_int8_sanity_accuracy_drop", 0.10)))
    )
    table.loc[invalid_int8, "valid_result"] = False
    clean = table[table["valid_result"]].copy()
    table.to_csv(root / "metrics_audited.csv", index=False)
    clean.to_csv(root / "final_comparison.csv", index=False)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for x, xlabel, filename in (("model_size_mb", "Model size (MB)", "accuracy_vs_size.png"), ("median_latency_ms", "Median CPU latency (ms)", "accuracy_vs_latency.png")):
        fig, axis = plt.subplots(figsize=(8, 5))
        for _, row in clean.iterrows():
            axis.scatter(row[x], row["accuracy"], s=65)
            axis.annotate(f"{row['variant']} - {row['model']}", (row[x], row["accuracy"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        axis.set(xlabel=xlabel, ylabel="Test accuracy", title="RQ7.1 compression trade-off")
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=180)
        plt.close(fig)
    print(clean[["variant", "model", "compression", "accuracy", "macro_f1", "model_size_mb", "median_latency_ms", "accuracy_drop"]].to_string(index=False))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RQ7.1 model-compression experiments")
    parser.add_argument("command", choices=("eda", "train", "quantize", "quantize-candidate", "prune", "prune-select", "distill", "report", "full"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="mobilenet_v3_small")
    parser.add_argument("--checkpoint")
    parser.add_argument("--fp32-path")
    parser.add_argument("--candidate-output")
    parser.add_argument("--candidate-result")
    parser.add_argument("--calibration-method", default="minmax")
    parser.add_argument("--activation-type", default="quint8")
    parser.add_argument("--per-channel", choices=("0", "1"), default="1")
    parser.add_argument("--op-types", default="")
    parser.add_argument("--conv-scope", choices=("all", "standard_only", "depthwise_only", "early_block_fp32"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["_config_path"] = str(Path(args.config).resolve())
    set_seed(int(config["seed"]))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.command == "eda":
        command_eda(config)
    elif args.command == "train":
        command_train(config, args.model)
    elif args.command == "report":
        command_report(config)
    elif args.command == "full":
        command_full(config)
    elif args.command == "quantize-candidate":
        required = [args.fp32_path, args.candidate_output, args.candidate_result]
        if not all(required):
            raise SystemExit("quantize-candidate requires --fp32-path, --candidate-output and --candidate-result")
        op_types = [item for item in args.op_types.split(",") if item] or None
        command_quantize_candidate(
            config,
            args.fp32_path,
            args.candidate_output,
            args.candidate_result,
            args.calibration_method,
            args.per_channel == "1",
            args.activation_type,
            op_types,
            args.conv_scope,
        )
    else:
        if not args.checkpoint:
            raise SystemExit(f"--checkpoint is required for {args.command}")
        {"quantize": command_quantize, "prune": command_prune, "prune-select": command_prune_select, "distill": command_distill}[args.command](config, args.checkpoint)


if __name__ == "__main__":
    main()
