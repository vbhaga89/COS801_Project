from __future__ import annotations

import torch.nn as nn
from torchvision import models


SUPPORTED_MODELS = ("efficientnet_b0", "mobilenet_v3_small", "mobilenet_v3_small_050")


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if name == "mobilenet_v3_small_050":
        if pretrained:
            raise ValueError("The 0.5x student has no matching torchvision pretrained weights")
        model = models.mobilenet_v3_small(weights=None, width_mult=0.5)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    raise ValueError(f"Unknown model {name!r}; choose one of {SUPPORTED_MODELS}")


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
