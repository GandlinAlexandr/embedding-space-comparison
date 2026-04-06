from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

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
    - weights_enum: какие torchvision-веса используются
    - extractor_id: строка для логов и воспроизводимости
    - input_size: какой квадратный spatial-size ожидает протокол извлечения
    - apply_extractor: функция, превращающая модель в экстрактор признаков
    """

    base_name: str
    weights_enum: Optional[str]
    extractor_id: str
    input_size: int
    apply_extractor: Callable[[torch.nn.Module], torch.nn.Module]


def _make_registry() -> Dict[str, ModelSpec]:
    """
    Реестр "имя модели -> спецификация".

    ВАЖНО:
    - здесь удобно добавлять новые модели/варианты;
    - extractor_id должен быть стабильным, это часть определения эмбеддингов.
    """
    reg: Dict[str, ModelSpec] = {}

    def add(
        name: str,
        *,
        base_name: str,
        weights_enum: str,
        extractor_id: str,
        input_size: int = 224,
        apply_extractor: Callable[[torch.nn.Module], torch.nn.Module],
    ) -> None:
        reg[name] = ModelSpec(
            base_name=base_name,
            weights_enum=weights_enum,
            extractor_id=extractor_id,
            input_size=input_size,
            apply_extractor=apply_extractor,
        )

    # Семейство ResNet (fc -> Identity)
    add(
        "resnet18",
        base_name="resnet18",
        weights_enum="ResNet18_Weights.IMAGENET1K_V1",
        extractor_id="resnet_fc_identity",
        apply_extractor=as_resnet_feature_extractor,
    )
    add(
        "resnet34",
        base_name="resnet34",
        weights_enum="ResNet34_Weights.IMAGENET1K_V1",
        extractor_id="resnet_fc_identity",
        apply_extractor=as_resnet_feature_extractor,
    )
    add(
        "resnet50",
        base_name="resnet50",
        weights_enum="ResNet50_Weights.IMAGENET1K_V1",
        extractor_id="resnet_fc_identity",
        apply_extractor=as_resnet_feature_extractor,
    )
    add(
        "resnet50_v2",
        base_name="resnet50",
        weights_enum="ResNet50_Weights.IMAGENET1K_V2",
        extractor_id="resnet_fc_identity",
        apply_extractor=as_resnet_feature_extractor,
    )
    add(
        "resnet101",
        base_name="resnet101",
        weights_enum="ResNet101_Weights.IMAGENET1K_V1",
        extractor_id="resnet_fc_identity",
        apply_extractor=as_resnet_feature_extractor,
    )
    add(
        "resnet101_v2",
        base_name="resnet101",
        weights_enum="ResNet101_Weights.IMAGENET1K_V2",
        extractor_id="resnet_fc_identity",
        apply_extractor=as_resnet_feature_extractor,
    )
    add(
        "wide_resnet50_2",
        base_name="wide_resnet50_2",
        weights_enum="Wide_ResNet50_2_Weights.IMAGENET1K_V1",
        extractor_id="resnet_fc_identity",
        apply_extractor=as_resnet_feature_extractor,
    )
    add(
        "wide_resnet50_2_v2",
        base_name="wide_resnet50_2",
        weights_enum="Wide_ResNet50_2_Weights.IMAGENET1K_V2",
        extractor_id="resnet_fc_identity",
        apply_extractor=as_resnet_feature_extractor,
    )
    add(
        "wide_resnet101_2",
        base_name="wide_resnet101_2",
        weights_enum="Wide_ResNet101_2_Weights.IMAGENET1K_V1",
        extractor_id="resnet_fc_identity",
        apply_extractor=as_resnet_feature_extractor,
    )
    add(
        "wide_resnet101_2_v2",
        base_name="wide_resnet101_2",
        weights_enum="Wide_ResNet101_2_Weights.IMAGENET1K_V2",
        extractor_id="resnet_fc_identity",
        apply_extractor=as_resnet_feature_extractor,
    )

    # ViT
    add(
        "vit_b_16",
        base_name="vit_b_16",
        weights_enum="ViT_B_16_Weights.IMAGENET1K_V1",
        extractor_id="vit_heads_identity",
        apply_extractor=as_vit_feature_extractor,
    )
    add(
        "vit_b_16_swag_e2e",
        base_name="vit_b_16",
        weights_enum="ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1",
        extractor_id="vit_heads_identity",
        input_size=384,
        apply_extractor=as_vit_feature_extractor,
    )
    add(
        "vit_b_16_swag_linear",
        base_name="vit_b_16",
        weights_enum="ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1",
        extractor_id="vit_heads_identity",
        apply_extractor=as_vit_feature_extractor,
    )
    add(
        "vit_b_32",
        base_name="vit_b_32",
        weights_enum="ViT_B_32_Weights.IMAGENET1K_V1",
        extractor_id="vit_heads_identity",
        apply_extractor=as_vit_feature_extractor,
    )
    add(
        "vit_l_16",
        base_name="vit_l_16",
        weights_enum="ViT_L_16_Weights.IMAGENET1K_V1",
        extractor_id="vit_heads_identity",
        apply_extractor=as_vit_feature_extractor,
    )
    add(
        "vit_l_16_swag_linear",
        base_name="vit_l_16",
        weights_enum="ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1",
        extractor_id="vit_heads_identity",
        apply_extractor=as_vit_feature_extractor,
    )
    add(
        "vit_l_32",
        base_name="vit_l_32",
        weights_enum="ViT_L_32_Weights.IMAGENET1K_V1",
        extractor_id="vit_heads_identity",
        apply_extractor=as_vit_feature_extractor,
    )

    # DenseNet / MobileNet
    add(
        "densenet121",
        base_name="densenet121",
        weights_enum="DenseNet121_Weights.IMAGENET1K_V1",
        extractor_id="densenet_classifier_identity",
        apply_extractor=as_densenet_feature_extractor,
    )
    add(
        "mobilenet_v2",
        base_name="mobilenet_v2",
        weights_enum="MobileNet_V2_Weights.IMAGENET1K_V1",
        extractor_id="mobilenetv2_classifier_identity",
        apply_extractor=as_mobilenet_v2_feature_extractor,
    )

    # VGG11: два варианта (чтобы было честно и воспроизводимо)
    add(
        "vgg11_fc4096",
        base_name="vgg11",
        weights_enum="VGG11_Weights.IMAGENET1K_V1",
        extractor_id="vgg_classifier_last_identity_4096",
        apply_extractor=as_vgg_classifier4096,
    )
    add(
        "vgg11_conv512",
        base_name="vgg11",
        weights_enum="VGG11_Weights.IMAGENET1K_V1",
        extractor_id="vgg_features_gap_512",
        apply_extractor=as_vgg_conv512,
    )
    reg["vgg11"] = reg["vgg11_conv512"]

    # VGG13: те же два extractor-варианта, как и для остальных VGG
    add(
        "vgg13_fc4096",
        base_name="vgg13",
        weights_enum="VGG13_Weights.IMAGENET1K_V1",
        extractor_id="vgg_classifier_last_identity_4096",
        apply_extractor=as_vgg_classifier4096,
    )
    add(
        "vgg13_conv512",
        base_name="vgg13",
        weights_enum="VGG13_Weights.IMAGENET1K_V1",
        extractor_id="vgg_features_gap_512",
        apply_extractor=as_vgg_conv512,
    )
    reg["vgg13"] = reg["vgg13_conv512"]

    # VGG16: два варианта (чтобы было честно и воспроизводимо)
    add(
        "vgg16_fc4096",
        base_name="vgg16",
        weights_enum="VGG16_Weights.IMAGENET1K_V1",
        extractor_id="vgg_classifier_last_identity_4096",
        apply_extractor=as_vgg_classifier4096,
    )
    add(
        "vgg16_conv512",
        base_name="vgg16",
        weights_enum="VGG16_Weights.IMAGENET1K_V1",
        extractor_id="vgg_features_gap_512",
        apply_extractor=as_vgg_conv512,
    )
    reg["vgg16"] = reg["vgg16_conv512"]

    # VGG19: два варианта (чтобы было честно и воспроизводимо)
    add(
        "vgg19_fc4096",
        base_name="vgg19",
        weights_enum="VGG19_Weights.IMAGENET1K_V1",
        extractor_id="vgg_classifier_last_identity_4096",
        apply_extractor=as_vgg_classifier4096,
    )
    add(
        "vgg19_conv512",
        base_name="vgg19",
        weights_enum="VGG19_Weights.IMAGENET1K_V1",
        extractor_id="vgg_features_gap_512",
        apply_extractor=as_vgg_conv512,
    )
    reg["vgg19"] = reg["vgg19_conv512"]

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

    model = build_torchvision_model(spec.base_name, spec.weights_enum)
    model = spec.apply_extractor(model)

    return model, spec
