from __future__ import annotations

"""
Утилиты для датасетов.

Зачем отдельный модуль:
- не трогаем основную логику расчёта метрик/даунстрима;
- проще добавлять новые датасеты (ImageNet, CIFAR-100, etc.);
- проще гарантировать, что рефакторинг не меняет результаты: все преобразования и загрузки лежат в одном месте.

ВАЖНО:
- Для сохранения воспроизводимости на CIFAR-10 здесь намеренно оставлены
  ровно те же transforms, что были в scripts/run_extract_embeddings.py:
  Resize(224) + ToTensor() + Normalize(ImageNet mean/std).
"""

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def default_transform_imagenet224() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(224),  # чтобы всё приводилось к размеру под ImageNet-модели
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def build_dataset(
    dataset_name: str,
    data_root: str,
    split: str,
    transform: Optional[transforms.Compose] = None
) -> torch.utils.data.Dataset:
    """
    Построить датасет для извлечения эмбеддингов.

    split: train|test
    """
    dataset_name = dataset_name.lower()
    split = split.lower()
    if split not in ("train", "test"):
        raise ValueError("split должен быть 'train' или 'test'")

    # По умолчанию используем тот же transform, что раньше.
    if transform is None:
        transform = default_transform_imagenet224()

    if dataset_name in ("cifar10", "cifar-10", "cifar_10"):
        return datasets.CIFAR10(
            root=data_root,
            train=(split == "train"),
            download=True,
            transform=transform
        )

    # TODO: сюда удобно добавлять другие датасеты.
    raise ValueError(f"Unknown dataset: {dataset_name}")


def build_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = False
) -> DataLoader:
    # ВАЖНО: shuffle=False, чтобы индексы subset_indices.npy однозначно соответствовали порядку объектов.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


def load_labels(dataset_name: str, data_root: str, split: str) -> np.ndarray:
    """
    Загрузить метки (targets) для downstream-оценки.

    """
    dataset_name = dataset_name.lower()
    split = split.lower()
    if split not in ("train", "test"):
        raise ValueError("split должен быть 'train' или 'test'")

    if dataset_name in ("cifar10", "cifar-10", "cifar_10"):
        ds = datasets.CIFAR10(root=data_root, train=(split == "train"), download=True)
        return np.asarray(ds.targets, dtype=np.int64)

    # TODO: расширение под другие датасеты.
    raise ValueError(f"Unknown dataset: {dataset_name}")