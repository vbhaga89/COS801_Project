from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import nn


def _limited(loader: Iterable[Any], max_batches: int | None) -> Iterable[Any]:
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        yield batch


def evaluate(
    model: nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    total_loss = 0.0
    total_examples = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for images, targets in _limited(loader, max_batches):
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            total_loss += float(criterion(logits, targets).item())
            total_examples += targets.numel()
            labels.extend(targets.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    if not labels:
        raise RuntimeError("Evaluation loader produced no examples")
    return {
        "loss": total_loss / total_examples,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "examples": float(total_examples),
    }


def train_supervised(
    model: nn.Module,
    train_loader: Iterable[Any],
    val_loader: Iterable[Any],
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_f1 = -math.inf
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        for images, targets in _limited(train_loader, max_train_batches):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, val_loader, device, max_eval_batches)
        history.append({"epoch": float(epoch), **metrics})
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_distillation(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: Iterable[Any],
    val_loader: Iterable[Any],
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    alpha: float,
    temperature: float,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    teacher.to(device).eval()
    student.to(device)
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_f1 = -math.inf

    for epoch in range(1, epochs + 1):
        student.train()
        for images, targets in _limited(train_loader, max_train_batches):
            images, targets = images.to(device), targets.to(device)
            with torch.inference_mode():
                teacher_logits = teacher(images)
            student_logits = student(images)
            hard_loss = F.cross_entropy(student_logits, targets)
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature**2)
            loss = alpha * hard_loss + (1.0 - alpha) * soft_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = evaluate(student, val_loader, device, max_eval_batches)
        history.append({"epoch": float(epoch), **metrics})
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}

    if best_state is not None:
        student.load_state_dict(best_state)
    return student, history
