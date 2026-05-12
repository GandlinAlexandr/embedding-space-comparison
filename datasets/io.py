from __future__ import annotations

"""
Утилиты для датасетов.
"""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class IndexedDataset(Dataset):
    """
    Тонкая обёртка над датасетом для фиксированного подмножества индексов.

    Нужна для SUN397, где torchvision не предоставляет split=train/test
    напрямую, а проекту важно сохранить единый интерфейс.
    """

    def __init__(self, base_dataset: Dataset, indices: Sequence[int]):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        return self.base_dataset[int(self.indices[idx])]


def default_transform_imagenet224(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def _canonical_dataset_name(dataset_name: str) -> str:
    name = dataset_name.lower().strip()

    aliases = {
        "cifar10": "cifar10",
        "cifar-10": "cifar10",
        "cifar_10": "cifar10",
        "food101": "food101",
        "food-101": "food101",
        "food_101": "food101",
        "flowers102": "flowers102",
        "flowers-102": "flowers102",
        "flowers_102": "flowers102",
        "sun397": "sun397",
        "sun-397": "sun397",
        "sun_397": "sun397",
        "imagenet": "imagenet1k",
        "imagenet1k": "imagenet1k",
        "imagenet-1k": "imagenet1k",
        "imagenet_1k": "imagenet1k",
        "ilsvrc2012": "imagenet1k",
    }

    if name not in aliases:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return aliases[name]


def _validate_project_split(split: str) -> str:
    split = split.lower().strip()
    if split not in ("train", "val", "test", "trainval"):
        raise ValueError(
            "split должен быть одним из: 'train', 'val', 'test', 'trainval'"
        )
    return split


def _torchvision_split_name(dataset_name: str, split: str) -> str:
    """
    Отображает project split во внутренние split-имена torchvision.
    """
    if dataset_name == "imagenet1k":
        if split == "train":
            return "train"
        if split in {"val", "test"}:
            return "val"
    return split


def _extract_labels(dataset: Dataset) -> np.ndarray:
    """
    Преобразует targets/_labels/... к единому np.int64.
    """
    if isinstance(dataset, ConcatDataset):
        parts = [_extract_labels(child) for child in dataset.datasets]
        return np.concatenate(parts, axis=0).astype(np.int64)

    for attr in ("targets", "_labels"):
        if hasattr(dataset, attr):
            values = getattr(dataset, attr)
            return np.asarray(values, dtype=np.int64)

    if hasattr(dataset, "samples"):
        return np.asarray([label for _, label in dataset.samples], dtype=np.int64)

    raise ValueError(
        "Не удалось извлечь labels из датасета. " f"Тип: {type(dataset).__name__}"
    )


def _sun397_partition_paths(data_root: str, partition: int = 1) -> tuple[Path, Path]:
    """
    Ищет официальные partition-файлы SUN397.

    Поддерживаем два типичных расположения:
    - <data_root>/SUN397/Partitions/Training_01.txt
    - <data_root>/Partitions/Training_01.txt
    """
    part_name = f"{partition:02d}"
    candidates = [
        Path(data_root) / "SUN397" / "Partitions",
        Path(data_root) / "Partitions",
    ]

    for base in candidates:
        train_path = base / f"Training_{part_name}.txt"
        test_path = base / f"Testing_{part_name}.txt"
        if train_path.exists() and test_path.exists():
            return train_path, test_path

    raise RuntimeError(
        "Для SUN397 не найдены официальные partition-файлы. "
        "Ожидаются файлы вида Training_01.txt и Testing_01.txt "
        "в <data_root>/SUN397/Partitions или <data_root>/Partitions."
    )


def _normalize_sun397_relpath(path: Path, data_dir: Path) -> str:
    rel = path.relative_to(data_dir).as_posix()
    return f"/{rel}" if not rel.startswith("/") else rel


def _load_sun397_indices(
    base_dataset: datasets.SUN397,
    split: str,
    partition: int = 1,
) -> np.ndarray:
    train_path, test_path = _sun397_partition_paths(
        base_dataset.root, partition=partition
    )
    list_path = train_path if split == "train" else test_path

    path_to_idx = {
        _normalize_sun397_relpath(path, base_dataset._data_dir): idx
        for idx, path in enumerate(base_dataset._image_files)
    }

    indices = []
    missing = []
    with open(list_path, "r", encoding="utf-8") as f:
        for raw in f:
            rel = raw.strip().replace("\\", "/")
            if not rel:
                continue
            if not rel.startswith("/"):
                rel = f"/{rel}"
            idx = path_to_idx.get(rel)
            if idx is None:
                missing.append(rel)
                continue
            indices.append(idx)

    if missing:
        preview = ", ".join(missing[:3])
        raise RuntimeError(
            "Не удалось сопоставить часть путей из SUN397 partition-файла "
            f"{list_path}. Примеры: {preview}"
        )

    return np.asarray(indices, dtype=np.int64)


def _build_sun397_dataset(
    data_root: str,
    split: str,
    transform: Optional[transforms.Compose],
) -> Dataset:
    base = datasets.SUN397(
        root=data_root,
        download=True,
        transform=transform,
    )
    indices = _load_sun397_indices(base, split=split, partition=1)
    return IndexedDataset(base, indices)


def _build_raw_dataset(
    dataset_name: str,
    data_root: str,
    split: str,
    transform: Optional[transforms.Compose],
) -> Dataset:
    tv_split = _torchvision_split_name(dataset_name, split)

    if dataset_name == "cifar10":
        if split in {"val", "trainval"}:
            raise ValueError(
                "Для cifar10 поддерживаются только split='train' и split='test'."
            )
        return datasets.CIFAR10(
            root=data_root,
            train=(split == "train"),
            download=True,
            transform=transform,
        )

    if dataset_name == "food101":
        if split in {"val", "trainval"}:
            raise ValueError(
                "Для food101 в текущем протоколе поддерживаются только split='train' и split='test'."
            )
        return datasets.Food101(
            root=data_root,
            split=tv_split,
            download=True,
            transform=transform,
        )

    if dataset_name == "flowers102":
        if split == "trainval":
            train_ds = datasets.Flowers102(
                root=data_root,
                split="train",
                download=True,
                transform=transform,
            )
            val_ds = datasets.Flowers102(
                root=data_root,
                split="val",
                download=True,
                transform=transform,
            )
            return ConcatDataset([train_ds, val_ds])
        return datasets.Flowers102(
            root=data_root,
            split=tv_split,
            download=True,
            transform=transform,
        )

    if dataset_name == "sun397":
        if split in {"val", "trainval"}:
            raise ValueError(
                "Для sun397 в benchmark-протоколе поддерживаются только split='train' и split='test'."
            )
        return _build_sun397_dataset(
            data_root=data_root,
            split=split,
            transform=transform,
        )

    if dataset_name == "imagenet1k":
        if split == "trainval":
            raise ValueError(
                "Для imagenet1k split='trainval' не поддерживается: используйте 'train' и 'val'/'test'."
            )
        return datasets.ImageNet(
            root=data_root,
            split=tv_split,
            transform=transform,
        )

    raise ValueError(f"Unknown dataset: {dataset_name}")


def build_dataset(
    dataset_name: str,
    data_root: str,
    split: str,
    transform: Optional[transforms.Compose] = None,
) -> torch.utils.data.Dataset:
    """
    Построить датасет для извлечения эмбеддингов.

    split: train|val|test|trainval
    """
    dataset_name = _canonical_dataset_name(dataset_name)
    split = _validate_project_split(split)

    if transform is None:
        transform = default_transform_imagenet224()

    return _build_raw_dataset(
        dataset_name=dataset_name,
        data_root=data_root,
        split=split,
        transform=transform,
    )


def build_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = False,
) -> DataLoader:

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def load_labels(dataset_name: str, data_root: str, split: str) -> np.ndarray:
    """
    Загрузить метки (targets) для downstream-оценки.

    """
    dataset_name = _canonical_dataset_name(dataset_name)
    split = _validate_project_split(split)

    ds = _build_raw_dataset(
        dataset_name=dataset_name,
        data_root=data_root,
        split=split,
        transform=None,
    )

    if isinstance(ds, IndexedDataset):
        base_labels = _extract_labels(ds.base_dataset)
        return np.asarray(base_labels[ds.indices], dtype=np.int64)

    return _extract_labels(ds)
