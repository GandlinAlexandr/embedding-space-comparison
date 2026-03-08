from __future__ import annotations

from typing import Callable, Dict

import torch


def build_torchvision_model(model_name: str) -> torch.nn.Module:
    """
    Фабрика torchvision-моделей с фиксированными весами.

    ВАЖНО:
    - веса заданы явно, чтобы обновления torchvision не меняли поведение «тихо».
    """
    import torchvision.models as M

    name = model_name.lower()

    factories: Dict[str, Callable[[], torch.nn.Module]] = {
        "resnet18": lambda: M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1),
        "resnet34": lambda: M.resnet34(weights=M.ResNet34_Weights.IMAGENET1K_V1),
        "resnet50": lambda: M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V1),
        "resnet101": lambda: M.resnet101(weights=M.ResNet101_Weights.IMAGENET1K_V1),
        "wide_resnet50_2": lambda: M.wide_resnet50_2(weights=M.Wide_ResNet50_2_Weights.IMAGENET1K_V1),

        "vgg11": lambda: M.vgg11(weights=M.VGG11_Weights.IMAGENET1K_V1),
        "vgg16": lambda: M.vgg16(weights=M.VGG16_Weights.IMAGENET1K_V1),

        "densenet121": lambda: M.densenet121(weights=M.DenseNet121_Weights.IMAGENET1K_V1),
        "mobilenet_v2": lambda: M.mobilenet_v2(weights=M.MobileNet_V2_Weights.IMAGENET1K_V1),

        "vit_b_16": lambda: M.vit_b_16(weights=M.ViT_B_16_Weights.IMAGENET1K_V1),
    }

    if name not in factories:
        raise ValueError(f"Неизвестная torchvision-модель: {model_name}. Доступные варианты: {list(factories.keys())}")

    return factories[name]()