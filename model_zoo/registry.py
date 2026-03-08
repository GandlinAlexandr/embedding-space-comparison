from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch

from model_zoo.feature_extractors import (
    as_densenet_feature_extractor,
    as_mobilenet_v2_feature_extractor,
    as_resnet_feature_extractor,
    as_vit_feature_extractor,
    as_vgg_classifier4096,
    as_vgg_conv512,
)
from model_zoo.torchvision_models import build_torchvision_model


@dataclass(frozen=True)
class ModelSpec:
    """
    Описание модели для эмбеддингов:
    - base_name: какая torchvision-модель создаётся
    - extractor_id: строка для логов и воспроизводимости
    - apply_extractor: функция, превращающая модель в экстрактор признаков
    """

    base_name: str
    extractor_id: str
    apply_extractor: Callable[[torch.nn.Module], torch.nn.Module]


def _make_registry() -> Dict[str, ModelSpec]:
    """
    Реестр "имя модели -> спецификация".

    ВАЖНО:
    - здесь удобно добавлять новые модели/варианты;
    - extractor_id должен быть стабильным, это часть определения эмбеддингов.
    """
    reg: Dict[str, ModelSpec] = {}

    # Семейство ResNet (fc -> Identity)
    reg["resnet18"] = ModelSpec(
        base_name="resnet18",
        extractor_id="resnet_fc_identity",
        apply_extractor=as_resnet_feature_extractor,
    )
    reg["resnet34"] = ModelSpec(
        base_name="resnet34",
        extractor_id="resnet_fc_identity",
        apply_extractor=as_resnet_feature_extractor,
    )
    reg["resnet50"] = ModelSpec(
        base_name="resnet50",
        extractor_id="resnet_fc_identity",
        apply_extractor=as_resnet_feature_extractor,
    )
    reg["resnet101"] = ModelSpec(
        base_name="resnet101",
        extractor_id="resnet_fc_identity",
        apply_extractor=as_resnet_feature_extractor,
    )
    reg["wide_resnet50_2"] = ModelSpec(
        base_name="wide_resnet50_2",
        extractor_id="resnet_fc_identity",
        apply_extractor=as_resnet_feature_extractor,
    )

    # ViT
    reg["vit_b_16"] = ModelSpec(
        base_name="vit_b_16",
        extractor_id="vit_heads_identity",
        apply_extractor=as_vit_feature_extractor,
    )

    # DenseNet / MobileNet
    reg["densenet121"] = ModelSpec(
        base_name="densenet121",
        extractor_id="densenet_classifier_identity",
        apply_extractor=as_densenet_feature_extractor,
    )
    reg["mobilenet_v2"] = ModelSpec(
        base_name="mobilenet_v2",
        extractor_id="mobilenetv2_classifier_identity",
        apply_extractor=as_mobilenet_v2_feature_extractor,
    )

    # VGG11: два варианта (чтобы было честно и воспроизводимо)
    reg["vgg11_fc4096"] = ModelSpec(
        base_name="vgg11",
        extractor_id="vgg_classifier_last_identity_4096",
        apply_extractor=as_vgg_classifier4096,
    )
    reg["vgg11_conv512"] = ModelSpec(
        base_name="vgg11",
        extractor_id="vgg_features_gap_512",
        apply_extractor=as_vgg_conv512,
    )
    reg["vgg11"] = reg["vgg11_conv512"]

    # VGG16: два варианта (чтобы было честно и воспроизводимо)
    reg["vgg16_fc4096"] = ModelSpec(
        base_name="vgg16",
        extractor_id="vgg_classifier_last_identity_4096",
        apply_extractor=as_vgg_classifier4096,
    )
    reg["vgg16_conv512"] = ModelSpec(
        base_name="vgg16",
        extractor_id="vgg_features_gap_512",
        apply_extractor=as_vgg_conv512,
    )
    reg["vgg16"] = reg["vgg16_conv512"]

    return reg


_REGISTRY = _make_registry()


def available_models() -> List[str]:
    return sorted(_REGISTRY.keys())


def get_model(model_name: str) -> Tuple[torch.nn.Module, ModelSpec]:
    """
    Создать модель и применить экстрактор признаков согласно реестру.
    Возвращает (model, spec) — spec полезен для логирования.
    """
    name = model_name.lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Неизвестная модель: {model_name}. Доступные варианты: {available_models()}"
        )

    spec = _REGISTRY[name]

    model = build_torchvision_model(spec.base_name)
    model = spec.apply_extractor(model)

    return model, spec
