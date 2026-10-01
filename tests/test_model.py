import torch

from defect_detector.model import build_model


def test_small_model_has_binary_output() -> None:
    model = build_model("small_cnn", num_classes=2)
    output = model(torch.randn(2, 3, 64, 64))
    assert tuple(output.shape) == (2, 2)


def test_resnet_head_is_binary() -> None:
    model = build_model("resnet18", num_classes=2)
    assert model.fc.out_features == 2


def test_convnext_head_is_binary() -> None:
    model = build_model("convnext_tiny", num_classes=2)
    assert model.classifier[-1].out_features == 2
