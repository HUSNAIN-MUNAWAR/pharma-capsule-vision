"""Model factory for the offline demo and transfer-learning production baseline."""

from __future__ import annotations

import logging

from torch import Tensor, nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    convnext_tiny,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
)

LOGGER = logging.getLogger(__name__)


class SmallCNN(nn.Module):
    """Compact model used for deterministic CPU smoke tests."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.features(inputs)
        return self.classifier(features.flatten(1))


def build_model(
    name: str,
    num_classes: int = 2,
    pretrained: bool = False,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Create a classifier with an explicit, auditable transfer-learning policy."""
    if name == "small_cnn":
        return SmallCNN(num_classes=num_classes)
    if name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        head_prefix = "classifier."
    elif name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head_prefix = "fc."
    elif name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        head_prefix = "classifier."
    elif name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        head_prefix = "classifier."
    else:
        raise ValueError(
            f"Unsupported model: {name}. Choose convnext_tiny, small_cnn, resnet18, "
            "mobilenet_v3_small, or efficientnet_b0."
        )

    if freeze_backbone and pretrained:
        for parameter_name, parameter in model.named_parameters():
            if not parameter_name.startswith(head_prefix):
                parameter.requires_grad = False
    elif freeze_backbone:
        LOGGER.warning("Ignoring freeze_backbone because %s has no pretrained weights", name)
    LOGGER.info("Built %s (pretrained=%s, freeze_backbone=%s)", name, pretrained, freeze_backbone)
    return model
