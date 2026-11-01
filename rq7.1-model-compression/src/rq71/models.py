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


class FrozenLinearProbe(nn.Module):
    """Fixed ImageNet features with one trainable linear classifier."""
    def __init__(self, backbone: nn.Module, feature_dim: int, classes: int):
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.head = nn.Linear(feature_dim, classes)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # Freeze BN buffers and dropout, not only gradients.
        return self

    def forward(self, images):
        import torch
        with torch.no_grad():
            features = self.backbone(images)
        return self.head(features)


def build_linear_probe(name: str, classes: int, pretrained: bool = True):
    backbone = build_model(name, classes, pretrained)
    # Use pooled convolutional features, not MobileNet's nonlinear classifier.
    feature_dim = backbone.classifier[0].in_features if name.startswith('mobilenet') else backbone.classifier[-1].in_features
    backbone.classifier = nn.Identity()
    return FrozenLinearProbe(backbone, feature_dim, classes)
