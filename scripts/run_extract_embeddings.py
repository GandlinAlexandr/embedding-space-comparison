from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets.io import (
    build_dataset,
    build_loader,
    default_transform_imagenet224,
    load_labels,
)  # вынесено в отдельный пакет datasets/
from model_zoo.registry import (
    get_model,
    available_models,
)


def _stratified_subset_indices(
    labels: np.ndarray,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    y = np.asarray(labels).reshape(-1)
    n = int(y.size)
    if num_samples <= 0 or num_samples >= n:
        return np.arange(n, dtype=np.int64)

    classes, counts = np.unique(y, return_counts=True)
    n_classes = int(classes.size)
    if num_samples < n_classes:
        raise ValueError(
            f"Для stratified sample_size={num_samples} меньше числа классов={n_classes}."
        )

    exact = counts.astype(np.float64) * (float(num_samples) / float(n))
    quotas = np.floor(exact).astype(np.int64)
    quotas = np.maximum(quotas, 1)
    quotas = np.minimum(quotas, counts)

    while int(np.sum(quotas)) > num_samples:
        removable = np.where(quotas > 1)[0]
        if removable.size == 0:
            raise ValueError(
                "Не удалось построить stratified подвыборку с заданным sample_size."
            )
        frac = exact[removable] - np.floor(exact[removable])
        cls_pos = removable[int(np.argmin(frac))]
        quotas[cls_pos] -= 1

    while int(np.sum(quotas)) < num_samples:
        room = np.where(quotas < counts)[0]
        if room.size == 0:
            break
        deficit = exact[room] - quotas[room].astype(np.float64)
        cls_pos = room[int(np.argmax(deficit))]
        quotas[cls_pos] += 1

    selected: List[np.ndarray] = []
    for cls, q in zip(classes, quotas):
        cls_idx = np.where(y == cls)[0]
        chosen = rng.choice(cls_idx, size=int(q), replace=False)
        selected.append(chosen.astype(np.int64))
    idx = np.concatenate(selected)
    idx.sort()
    return idx.astype(np.int64)


def _subset_indices(
    n: int,
    num_samples: int,
    strategy: str,
    seed: int,
    labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    rng = np.random.RandomState(seed)

    if num_samples <= 0 or num_samples >= n:
        return np.arange(n, dtype=np.int64)

    if strategy == "random":
        idx = rng.choice(n, size=num_samples, replace=False)
        idx.sort()
        return idx.astype(np.int64)

    if strategy == "stratified":
        if labels is None:
            raise ValueError("Для subset_strategy=stratified нужны labels.")
        return _stratified_subset_indices(labels, num_samples, seed)

    raise ValueError(f"Неизвестная стратегия подвыборки: {strategy}")


def _dataset_key(dataset: str, split: str) -> str:
    return f"{dataset}_{split}"


def _sampled_dataset_key(
    dataset: str,
    split: str,
    num_samples: int,
    seed: int,
    strategy: str,
) -> str:
    key = f"{_dataset_key(dataset, split)}_s{int(num_samples)}_seed{int(seed)}"
    if str(strategy) != "random":
        key = f"{key}_{strategy}"
    return key


def _resolve_output_dir(args: argparse.Namespace) -> str:
    if args.output_dir:
        return args.output_dir
    if int(args.num_samples) > 0:
        key = _sampled_dataset_key(
            args.dataset,
            args.split,
            args.num_samples,
            args.seed,
            args.subset_strategy,
        )
        return os.path.join("data", "embeddings", "samples", key)
    return os.path.join("data", "embeddings", _dataset_key(args.dataset, args.split))


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
    parser.add_argument(
        "--split", type=str, default="test", help="train|val|test|trainval"
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help=(
            "Куда сохранять эмбеддинги. Если пусто: полный сплит -> "
            "data/embeddings/<dataset>_<split>, подвыборка -> "
            "data/embeddings/samples/<dataset>_<split>_sN_seedS_<strategy>."
        ),
    )

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
        "--sample_size",
        type=int,
        default=None,
        help="Псевдоним для --num_samples.",
    )
    parser.add_argument(
        "--subset_strategy",
        type=str,
        choices=["random", "stratified"],
        default="stratified",
        help="Стратегия подвыборки: stratified или random.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.sample_size is not None:
        args.num_samples = int(args.sample_size)
    args.output_dir = _resolve_output_dir(args)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    base_dataset = build_dataset(args.dataset, args.data_root, split=args.split)

    base_labels = load_labels(args.dataset, args.data_root, args.split).astype(np.int64)
    if int(base_labels.shape[0]) != int(len(base_dataset)):
        raise RuntimeError(
            f"Длина labels ({base_labels.shape[0]}) не совпадает с датасетом ({len(base_dataset)})."
        )

    # Необязательная подвыборка
    subset_indices = _subset_indices(
        len(base_dataset),
        args.num_samples,
        args.subset_strategy,
        args.seed,
        labels=base_labels,
    )
    subset_labels = np.asarray(base_labels[subset_indices], dtype=np.int64)
    if len(subset_indices) != len(base_dataset):
        idx_path = os.path.join(args.output_dir, "subset_indices.npy")
        labels_path = os.path.join(args.output_dir, "labels.npy")
        np.save(idx_path, np.asarray(subset_indices, dtype=np.int64))
        np.save(labels_path, subset_labels)
        classes, counts = np.unique(subset_labels, return_counts=True)
        class_counts = {str(int(c)): int(n) for c, n in zip(classes, counts)}
        with open(
            os.path.join(args.output_dir, "class_counts.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(class_counts, f, ensure_ascii=False, indent=2)
        print(f"Сохранены индексы подвыборки: {idx_path}")
        print(f"Сохранены метки подвыборки: {labels_path}")
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

    manifest = {
        "schema_version": 1,
        "kind": "extracted_embeddings",
        "dataset": args.dataset,
        "split": args.split,
        "dataset_key": (
            _sampled_dataset_key(
                args.dataset,
                args.split,
                args.num_samples,
                args.seed,
                args.subset_strategy,
            )
            if int(args.num_samples) > 0 and int(args.num_samples) < len(base_dataset)
            else _dataset_key(args.dataset, args.split)
        ),
        "sampled": bool(
            int(args.num_samples) > 0 and int(args.num_samples) < len(base_dataset)
        ),
        "sample_size": int(len(subset_indices)),
        "full_size": int(len(base_dataset)),
        "seed": int(args.seed),
        "subset_strategy": str(args.subset_strategy),
        "class_counts": {
            str(int(c)): int(n)
            for c, n in zip(*np.unique(subset_labels, return_counts=True))
        },
        "subset_indices_path": (
            os.path.abspath(os.path.join(args.output_dir, "subset_indices.npy"))
            if len(subset_indices) != len(base_dataset)
            else ""
        ),
        "labels_path": (
            os.path.abspath(os.path.join(args.output_dir, "labels.npy"))
            if len(subset_indices) != len(base_dataset)
            else ""
        ),
        "models": model_names,
        "output_dir": os.path.abspath(args.output_dir),
    }
    with open(
        os.path.join(args.output_dir, "embeddings_manifest.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
