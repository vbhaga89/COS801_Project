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
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
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
        self._loader = loader
        self._batches = batches
        self.rewind()

    def rewind(self) -> None:
        self._iterator = iter(self._loader)
        self._remaining = self._batches

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._remaining <= 0:
            return None
        try:
            images, _ = next(self._iterator)
        except StopIteration:
            return None
        self._remaining -= 1
        values = images.cpu().numpy().astype(np.float32)
        if values.ndim != 4 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError("Calibration requires finite NCHW RGB tensors")
        if self._remaining == self._batches - 1 or self._remaining % 10 == 0:
            print(
                f"Calibration batch {self._batches - self._remaining}/{self._batches}: "
                f"shape={values.shape}, range=[{values.min():.5f}, {values.max():.5f}]",
                flush=True,
            )
        return {"input": values}


def static_int8_quantize(
    fp32_path: str | Path,
    int8_path: str | Path,
    calibration_loader: Any,
    calibration_batches: int,
    calibration_method: str = "minmax",
    per_channel: bool = True,
    activation_type: str = "quint8",
    op_types_to_quantize: list[str] | None = None,
    prepared_validator: Any = None,
    nodes_to_exclude: list[str] | None = None,
) -> Path:
    source = Path(fp32_path)
    output = Path(int8_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared = output.with_name(output.stem + "_prepared.onnx")

    method_key = calibration_method.strip().lower()
    methods = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    if method_key not in methods:
        raise ValueError(
            f"Unsupported calibration method {calibration_method!r}; "
            f"choose from {sorted(methods)}"
        )
    activation_key = activation_type.strip().lower()
    activation_types = {"quint8": QuantType.QUInt8, "qint8": QuantType.QInt8}
    if activation_key not in activation_types:
        raise ValueError(
            f"Unsupported activation type {activation_type!r}; choose from {sorted(activation_types)}"
        )

    quant_pre_process(
        str(source),
        str(prepared),
        skip_symbolic_shape=False,
        skip_onnx_shape=False,
        skip_optimization=False,
    )
    try:
        if prepared_validator is not None:
            prepared_validator(prepared)
        print(f"Starting INT8 calibration: {output.name}", flush=True)
        quantize_static(
            model_input=str(prepared),
            model_output=str(output),
            calibration_data_reader=LoaderCalibrationReader(calibration_loader, calibration_batches),
            quant_format=QuantFormat.QDQ,
            activation_type=activation_types[activation_key],
            weight_type=QuantType.QInt8,
            calibrate_method=methods[method_key],
            per_channel=per_channel,
            reduce_range=False,
            op_types_to_quantize=op_types_to_quantize,
            nodes_to_exclude=nodes_to_exclude,
            extra_options={
                "ActivationSymmetric": False,
                "WeightSymmetric": True,
                "ForceQuantizeNoInputCheck": False,
            },
        )
    finally:
        if prepared.exists():
            prepared.unlink()
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
            if index == 0 or (index + 1) % 25 == 0:
                print(f"ONNX evaluation {Path(model_path).name}: {index + 1} batches, {total_examples} examples", flush=True)
    if not labels:
        raise RuntimeError("Evaluation loader produced no examples")
    return {
        "loss": total_loss / total_examples,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "examples": float(total_examples),
    }
