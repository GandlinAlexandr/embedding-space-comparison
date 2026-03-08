from __future__ import annotations

import torch
import torch.nn as nn


class VGGConvGAP(nn.Module):
    """
    VGG: «настоящие» эмбеддинги из сверточных признаков:
    features -> Global Average Pooling -> flatten  => (B, 512)

    Это НЕ выход classifier (4096), а представление на основе сверточных признаков.
    """

    def __init__(self, vgg: nn.Module):
        super().__init__()
        self.features = vgg.features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = self.flatten(x)
        return x


def as_resnet_feature_extractor(model: nn.Module) -> nn.Module:
    # resnet18/resnet34/resnet50/resnet101/wide_resnet50_2
    if hasattr(model, "fc"):
        model.fc = nn.Identity()
    return model


def as_vit_feature_extractor(model: nn.Module) -> nn.Module:
    # vit_b_16
    if hasattr(model, "heads"):
        model.heads = nn.Identity()
    return model


def as_densenet_feature_extractor(model: nn.Module) -> nn.Module:
    # densenet121
    if hasattr(model, "classifier"):
        model.classifier = nn.Identity()
    return model


def as_mobilenet_v2_feature_extractor(model: nn.Module) -> nn.Module:
    # mobilenet_v2
    # У torchvision MobileNetV2 classifier = Sequential(Dropout, Linear).
    # Identity оставляет выход (B, 1280) после глобального average pooling и flatten.
    if hasattr(model, "classifier"):
        model.classifier = nn.Identity()
    return model


def as_vgg_classifier4096(model: nn.Module) -> nn.Module:
    """
    Старое поведение VGG:
    classifier[-1] = Identity() => 4096-мерный выход предпоследнего полносвязного слоя.
    """
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        layers = list(model.classifier.children())
        if len(layers) > 0:
            layers[-1] = nn.Identity()
            model.classifier = nn.Sequential(*layers)
    return model


def as_vgg_conv512(model: nn.Module) -> nn.Module:
    """
    Сверточные признаки + GAP => 512-мерный вектор.
    """
    return VGGConvGAP(model)
