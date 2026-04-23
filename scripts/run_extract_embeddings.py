"""
Шаг 0 протокола.

Назначение:
- загрузить модель
- прогнать датасет
- сохранить эмбеддинги в формате .npy

Выход:
data/embeddings/{...}/{model_name}.npy
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets.io import (
    build_dataset,
    build_loader,
    default_transform_imagenet224,
)  # вынесено в отдельный пакет datasets/
from model_zoo.registry import (
    get_model,
    available_models,
)  # вынесено в отдельный пакет model_zoo/


def _subset_indices(n: int, num_samples: int, strategy: str, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)

    if num_samples <= 0 or num_samples >= n:
        return np.arange(n, dtype=np.int64)

    if strategy == "random":
        idx = rng.choice(n, size=num_samples, replace=False)
        idx.sort()
        return idx.astype(np.int64)

    raise ValueError(f"Неизвестная стратегия подвыборки: {strategy}")


def extract_embeddings_from_model(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    model = model.to(device)
    model.eval()

    all_embs: List[np.ndarray] = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Батчи", unit="batch"):
            x = x.to(device)
            emb = model(x)
            if isinstance(emb, (tuple, list)):
                emb = emb[0]
            emb = emb.detach().cpu().numpy()
            all_embs.append(emb)

    return np.concatenate(all_embs, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Извлечь эмбеддинги из torchvision-моделей."
    )
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--split", type=str, default="test", help="train|val|test|trainval")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # ВАЖНО:
    # - теперь vgg16 по умолчанию означает vgg16_conv512 (см. model_zoo.registry)
    # - если нужно старое поведение, укажи vgg16_fc4096
    parser.add_argument(
        "--models",
        type=str,
        default="resnet18,resnet50,vgg16,vit_b_16,wide_resnet50_2",
        help=f"Имена моделей через запятую. Доступные варианты: {', '.join(available_models())}",
    )

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Пересчитывать эмбеддинги даже если файл уже существует.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="Необязательный размер подвыборки (0 = использовать весь сплит).",
    )
    parser.add_argument(
        "--subset_strategy",
        type=str,
        default="random",
        help="Стратегия подвыборки (по умолчанию: random).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    base_dataset = build_dataset(args.dataset, args.data_root, split=args.split)

    # Необязательная подвыборка
    subset_indices = _subset_indices(
        len(base_dataset), args.num_samples, args.subset_strategy, args.seed
    )
    if len(subset_indices) != len(base_dataset):

        # Исправление: сохраняем индексы, чтобы downstream мог взять те же объекты
        idx_path = os.path.join(args.output_dir, "subset_indices.npy")
        np.save(idx_path, np.asarray(subset_indices, dtype=np.int64))
        print(f"Сохранены индексы подвыборки: {idx_path}")
    else:
        print(f"Используется полный сплит: {args.split} (n={len(base_dataset)})")

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    for model_name in model_names:
        out_path = os.path.join(args.output_dir, f"{model_name}.npy")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"\n=== Модель: {model_name} ===")
            print(f"Пропуск: файл уже существует: {out_path}")
            continue

        print(f"\n=== Модель: {model_name} ===")

        model, spec = get_model(model_name)
        print(
            f"Базовая модель: {spec.base_name} | экстрактор: {spec.extractor_id} | "
            f"input_size={spec.input_size}"
        )

        dataset = build_dataset(
            args.dataset,
            args.data_root,
            split=args.split,
            transform=default_transform_imagenet224(spec.input_size),
        )
        if len(subset_indices) != len(base_dataset):
            dataset = Subset(dataset, subset_indices)

        loader = build_loader(
            dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )

        embs = extract_embeddings_from_model(model, loader, device=device)
        np.save(out_path, embs.astype(np.float32))
        print(f"Сохранено: {out_path} | shape={embs.shape}")


if __name__ == "__main__":
    main()
