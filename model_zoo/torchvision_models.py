from __future__ import annotations

from typing import Callable, Dict, Optional

import torch


def build_torchvision_model(
    model_name: str,
    weights_enum: Optional[str] = None,
) -> torch.nn.Module:
    """
    Фабрика torchvision-моделей с фиксированными весами.

    ВАЖНО:
    - веса заданы явно, чтобы обновления torchvision не меняли поведение «тихо».
    """
    import torchvision.models as M

    name = model_name.lower()

    factories: Dict[str, Callable[[Optional[object]], torch.nn.Module]] = {
        "resnet18": lambda weights: M.resnet18(weights=weights),
        "resnet34": lambda weights: M.resnet34(weights=weights),
        "resnet50": lambda weights: M.resnet50(weights=weights),
        "resnet101": lambda weights: M.resnet101(weights=weights),
        "wide_resnet50_2": lambda weights: M.wide_resnet50_2(weights=weights),
        "wide_resnet101_2": lambda weights: M.wide_resnet101_2(weights=weights),
        "vgg11": lambda weights: M.vgg11(weights=weights),
        "vgg13": lambda weights: M.vgg13(weights=weights),
        "vgg16": lambda weights: M.vgg16(weights=weights),
        "vgg19": lambda weights: M.vgg19(weights=weights),
        "densenet121": lambda weights: M.densenet121(weights=weights),
        "mobilenet_v2": lambda weights: M.mobilenet_v2(weights=weights),
        "vit_b_16": lambda weights: M.vit_b_16(weights=weights),
        "vit_b_32": lambda weights: M.vit_b_32(weights=weights),
        "vit_l_16": lambda weights: M.vit_l_16(weights=weights),
        "vit_l_32": lambda weights: M.vit_l_32(weights=weights),
    }

    if name not in factories:
        raise ValueError(
            f"Неизвестная torchvision-модель: {model_name}. Доступные варианты: {list(factories.keys())}"
        )

    weights = M.get_weight(weights_enum) if weights_enum else None
    return factories[name](weights)
