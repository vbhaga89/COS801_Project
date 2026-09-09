from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch_pruning as tp
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from sklearn.metrics import accuracy_score, f1_score


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def structured_prune(model: nn.Module, image_size: int, pruning_ratio: float) -> nn.Module:
    model = model.cpu().eval()
    example = torch.randn(1, 3, image_size, image_size)
    output_layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.Linear) and module.out_features == model.classifier[-1].out_features
    ]
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example,
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=pruning_ratio,
        ignored_layers=output_layers,
    )
    pruner.step()
    return model


def export_onnx(model: nn.Module, output_path: str | Path, image_size: int) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    sample = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        sample,
        path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return path


class LoaderCalibrationReader(CalibrationDataReader):
    def __init__(self, loader: Any, batches: int):
        self._iterator = iter(loader)
        self._remaining = batches

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._remaining <= 0:
            return None
        try:
            images, _ = next(self._iterator)
        except StopIteration:
            return None
        self._remaining -= 1
        return {"input": images.cpu().numpy().astype(np.float32)}


def static_int8_quantize(
    fp32_path: str | Path,
    int8_path: str | Path,
    calibration_loader: Any,
    calibration_batches: int,
) -> Path:
    output = Path(int8_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(output),
        calibration_data_reader=LoaderCalibrationReader(calibration_loader, calibration_batches),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    return output


def benchmark_onnx(
    model_path: str | Path,
    image_size: int,
    warmup: int,
    runs: int,
    cpu_threads: int,
) -> dict[str, float]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = cpu_threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    sample = np.random.default_rng(0).normal(size=(1, 3, image_size, image_size)).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {"input": sample})
    timings = []
    for _ in range(runs):
        started = time.perf_counter_ns()
        session.run(None, {"input": sample})
        timings.append((time.perf_counter_ns() - started) / 1_000_000)
    ordered = sorted(timings)
    p95_index = min(len(ordered) - 1, max(0, round(0.95 * len(ordered)) - 1))
    return {
        "median_latency_ms": float(statistics.median(timings)),
        "p95_latency_ms": float(ordered[p95_index]),
        "model_size_mb": Path(model_path).stat().st_size / (1024 * 1024),
    }


def evaluate_onnx(
    model_path: str | Path,
    loader: Any,
    max_batches: int | None = None,
) -> dict[str, float]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    labels: list[int] = []
    predictions: list[int] = []
    total_loss = 0.0
    total_examples = 0
    with torch.inference_mode():
        for index, (images, targets) in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            logits_array = session.run(None, {"input": images.cpu().numpy().astype(np.float32)})[0]
            logits = torch.from_numpy(logits_array)
            total_loss += float(torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item())
            total_examples += targets.numel()
            labels.extend(targets.tolist())
            predictions.extend(logits.argmax(dim=1).tolist())
    if not labels:
        raise RuntimeError("Evaluation loader produced no examples")
    return {
        "loss": total_loss / total_examples,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "examples": float(total_examples),
    }
