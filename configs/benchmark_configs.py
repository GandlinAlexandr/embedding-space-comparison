"""
Конфигурации benchmark'ов.

Сейчас здесь зафиксирован benchmark прошлого ВКР, чтобы:
- не держать список из 21 моделей только в переписке или ноутбуках;
- воспроизводимо запускать extraction / downstream / evaluation;
- не дублировать family-map в нескольких местах.

Имена ниже соответствуют текущему model_zoo.registry проекта.
"""

from __future__ import annotations

from typing import Dict, List


# ============================================================
# Benchmark ВКР 2024/2025
# ============================================================

# 21 конфигурация моделей:
# - дубли прошлого года реализованы через разные torchvision weights
# - короткие имена остаются canonical aliases для V1-вариантов
VKR_2024_2025_MODEL_NAMES: List[str] = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet50_v2",
    "resnet101",
    "resnet101_v2",
    "wide_resnet50_2",
    "wide_resnet50_2_v2",
    "wide_resnet101_2",
    "wide_resnet101_2_v2",
    "vgg11",
    "vgg13",
    "vgg16",
    "vgg19",
    "vit_b_16",
    "vit_b_16_swag_e2e",
    "vit_b_16_swag_linear",
    "vit_b_32",
    "vit_l_16",
    "vit_l_16_swag_linear",
    "vit_l_32",
]


VKR_2024_2025_TARGET_DATASETS: List[str] = [
    "imagenet1k",
    "sun397",
    "food101",
    "flowers102",
]


VKR_2024_2025_FAMILY_MAP: Dict[str, str] = {
    "resnet18": "resnet",
    "resnet34": "resnet",
    "resnet50": "resnet",
    "resnet50_v2": "resnet",
    "resnet101": "resnet",
    "resnet101_v2": "resnet",
    "wide_resnet50_2": "resnet",
    "wide_resnet50_2_v2": "resnet",
    "wide_resnet101_2": "resnet",
    "wide_resnet101_2_v2": "resnet",
    "vgg11": "vgg",
    "vgg13": "vgg",
    "vgg16": "vgg",
    "vgg19": "vgg",
    "vit_b_16": "vit",
    "vit_b_16_swag_e2e": "vit",
    "vit_b_16_swag_linear": "vit",
    "vit_b_32": "vit",
    "vit_l_16": "vit",
    "vit_l_16_swag_linear": "vit",
    "vit_l_32": "vit",
}


VKR_2024_2025_RESNET_ONLY: List[str] = [
    model_name
    for model_name in VKR_2024_2025_MODEL_NAMES
    if VKR_2024_2025_FAMILY_MAP[model_name] == "resnet"
]


def benchmark_models_csv(model_names: List[str]) -> str:
    """
    Удобно для CLI-флагов вида --models a,b,c.
    """
    return ",".join(model_names)


VKR_2024_2025_MODEL_NAMES_CSV: str = benchmark_models_csv(VKR_2024_2025_MODEL_NAMES)
VKR_2024_2025_RESNET_ONLY_CSV: str = benchmark_models_csv(VKR_2024_2025_RESNET_ONLY)

