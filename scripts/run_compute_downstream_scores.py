from __future__ import annotations

import argparse
import copy
import json
import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from datasets.io import load_labels
from configs.text_benchmark_configs import (
    TEXT_EMBEDDING_MODEL_BY_ID,
    TEXT_EMBEDDING_MODEL_IDS,
    TEXT_EMBEDDING_TEXT20_MODEL_IDS,
)


def _set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Детерминизм порядка (cuDNN) без "строгого" режима, который требует CUBLAS_WORKSPACE_CONFIG.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _make_torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _load_embeddings_file(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.asarray(np.load(path), dtype=np.float32)
    if path.endswith(".npz"):
        data = np.load(path)
        if "arr_0" in data:
            return np.asarray(data["arr_0"], dtype=np.float32)
        keys = list(data.keys())
        if len(keys) == 1:
            return np.asarray(data[keys[0]], dtype=np.float32)
        for k in ["embeddings", "X"]:
            if k in data:
                return np.asarray(data[k], dtype=np.float32)
        return np.asarray(data[keys[0]], dtype=np.float32)
    raise ValueError(f"Неподдерживаемый файл эмбеддингов: {path}")


def _load_labels_file(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.asarray(np.load(path), dtype=np.int64)
    if path.endswith(".npz"):
        data = np.load(path)
        if "arr_0" in data:
            return np.asarray(data["arr_0"], dtype=np.int64)
        keys = list(data.keys())
        if len(keys) == 1:
            return np.asarray(data[keys[0]], dtype=np.int64)
        for k in ["labels", "y", "targets"]:
            if k in data:
                return np.asarray(data[k], dtype=np.int64)
        return np.asarray(data[keys[0]], dtype=np.int64)
    raise ValueError(f"Неподдерживаемый файл меток: {path}")


def _list_models(embeddings_dir: str) -> Dict[str, str]:
    files = {}
    for fn in os.listdir(embeddings_dir):
        if os.path.splitext(fn)[0] in {"labels", "targets", "subset_indices"}:
            continue
        if fn.endswith(".npy") or fn.endswith(".npz"):
            name = os.path.splitext(fn)[0]
            files[name] = os.path.join(embeddings_dir, fn)
    if not files:
        raise RuntimeError(f"В папке нет эмбеддингов: {embeddings_dir}")
    return files


def _parse_model_filter(raw: Optional[str]) -> Optional[list[str]]:
    raw = str(raw or "").strip()
    if not raw:
        return None
    if raw == "text20":
        return list(TEXT_EMBEDDING_TEXT20_MODEL_IDS)
    if raw == "primary":
        return list(TEXT_EMBEDDING_MODEL_IDS)
    names = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in names if x not in TEXT_EMBEDDING_MODEL_BY_ID]
    if unknown:
        raise ValueError(
            f"Unknown text model ids in --models: {unknown}. "
            f"Available: {TEXT_EMBEDDING_MODEL_IDS}"
        )
    return names


def _infer_dataset_name_from_path(path: Optional[str]) -> Optional[str]:
    """
    Пытается угадать имя датасета по имени папки.

    Примеры:
    - .../cifar10_train      -> cifar10
    - .../cifar10_test       -> cifar10
    - .../cifar10_train_xxx  -> cifar10
    - .../cifar10            -> cifar10
    """
    if not path:
        return None

    base = os.path.basename(os.path.normpath(path)).lower()

    m = re.match(r"^(.+?)_(trainval|train|val|test)(?:_.*)?$", base)
    if m:
        return m.group(1)

    return base


def _infer_split_from_path(path: Optional[str]) -> Optional[str]:
    """
    Пытается угадать split по имени папки эмбеддингов.
    """
    if not path:
        return None

    base = os.path.basename(os.path.normpath(path)).lower()
    m = re.match(r"^.+?_(trainval|train|val|test)(?:_.*)?$", base)
    if m:
        return m.group(1)

    return None


def _resolve_dataset_name(
    dataset_arg: Optional[str],
    embeddings_dir: Optional[str],
    train_embeddings_dir: Optional[str],
    test_embeddings_dir: Optional[str],
) -> Optional[str]:
    if dataset_arg:
        return dataset_arg.lower()

    for p in [embeddings_dir, train_embeddings_dir, test_embeddings_dir]:
        inferred = _infer_dataset_name_from_path(p)
        if inferred:
            return inferred

    return None


def _resolve_labels_holdout(
    *,
    labels_path: Optional[str],
    dataset_name: Optional[str],
    data_root: Optional[str],
    embeddings_dir: str,
) -> np.ndarray:
    """
    Метки для holdout-режима:
    - если labels_path задан, грузим из файла;
    - иначе пытаемся загрузить по (dataset_name, data_root, split), где split
      выводится из имени папки embeddings_dir.
      Если не удалось — считаем split='test' по умолчанию.
    """
    if labels_path:
        return _load_labels_file(labels_path)

    local_labels_path = os.path.join(embeddings_dir, "labels.npy")
    if os.path.exists(local_labels_path):
        return _load_labels_file(local_labels_path)

    if not dataset_name or not data_root:
        raise ValueError(
            "Если --labels_path не задан, нужно указать --dataset и --data_root "
            "(или чтобы --dataset можно было вывести из имени папки эмбеддингов)."
        )

    split = _infer_split_from_path(embeddings_dir) or "test"

    return load_labels(dataset_name, data_root, split).astype(np.int64)


def _resolve_labels_train_test(
    *,
    labels_path: Optional[str],
    dataset_name: Optional[str],
    data_root: Optional[str],
    train_embeddings_dir: Optional[str],
    test_embeddings_dir: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Метки для train/test-режима:
    - если labels_path задан, считаем это старым совместимым режимом и используем
      один и тот же массив меток для train и test;
    - иначе грузим train/test метки отдельно через datasets.io.load_labels.
    """
    if labels_path:
        y = _load_labels_file(labels_path)
        return y, y

    train_labels_path = (
        os.path.join(train_embeddings_dir, "labels.npy")
        if train_embeddings_dir is not None
        else ""
    )
    test_labels_path = (
        os.path.join(test_embeddings_dir, "labels.npy")
        if test_embeddings_dir is not None
        else ""
    )
    if train_labels_path and test_labels_path and os.path.exists(train_labels_path) and os.path.exists(test_labels_path):
        return _load_labels_file(train_labels_path), _load_labels_file(test_labels_path)

    if not dataset_name or not data_root:
        raise ValueError(
            "Для режима --train_embeddings_dir/--test_embeddings_dir без --labels_path "
            "нужно указать --dataset и --data_root "
            "(или чтобы --dataset можно было вывести из имени папок эмбеддингов)."
        )

    train_split = _infer_split_from_path(train_embeddings_dir) or "train"
    test_split = _infer_split_from_path(test_embeddings_dir) or "test"

    y_train = load_labels(dataset_name, data_root, train_split).astype(np.int64)
    y_test = load_labels(dataset_name, data_root, test_split).astype(np.int64)
    return y_train, y_test


class MLPProbe(nn.Module):
    def __init__(self, dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        # Совмещаем текущий пайплайн с архитектурой probe из прошлогоднего benchmark'а:
        # 3 линейных слоя, 2 ReLU и 1 Dropout.
        self.net = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return float((pred == y).float().mean().item())


@torch.no_grad()
def _eval_model_accuracy(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    device: torch.device,
) -> float:
    model.eval()
    logits = model(X.to(device))
    return _accuracy(logits, y.to(device))


def _train_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    n_classes: int,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    patience: int,
    min_delta: float,
    device: torch.device,
    seed: int,
    num_workers: int,
) -> Tuple[nn.Module, float]:
    dim = X_train.shape[1]
    model = MLPProbe(dim, n_classes=n_classes, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    ds = TensorDataset(X_train, y_train)
    g = _make_torch_generator(seed)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, generator=g, num_workers=num_workers
    )

    best_val = -1.0
    bad = 0
    best_state = copy.deepcopy(model.state_dict())

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        val_acc = _eval_model_accuracy(model, X_val, y_val, device=device)

        if val_acc > best_val + min_delta:
            best_val = val_acc
            bad = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, float(best_val)


def _eval_probe_holdout(
    X: np.ndarray,
    y: np.ndarray,
    *,
    val_size: float,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    patience: int,
    min_delta: float,
    num_workers: int,
) -> float:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=y
    )

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val).long()

    n_classes = int(np.max(y) + 1)

    _, best_val = _train_probe(
        X_train_t,
        y_train_t,
        X_val_t,
        y_val_t,
        n_classes,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        patience=patience,
        min_delta=min_delta,
        device=device,
        seed=seed,
        num_workers=num_workers,
    )

    return float(best_val)


def _eval_probe_train_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    val_size: float,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    patience: int,
    min_delta: float,
    num_workers: int,
) -> float:
    # Разделить обучающую выборку на обучающую и валидационную, затем выбрать лучший результат на валидационной выборке и вывести результат на тестовой выборке.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train
    )

    X_tr_t = torch.from_numpy(X_tr)
    y_tr_t = torch.from_numpy(y_tr).long()
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val).long()
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test).long()

    n_classes = int(np.max(y_train) + 1)

    model, _ = _train_probe(
        X_tr_t,
        y_tr_t,
        X_val_t,
        y_val_t,
        n_classes,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        patience=patience,
        min_delta=min_delta,
        device=device,
        seed=seed,
        num_workers=num_workers,
    )

    test_acc = _eval_model_accuracy(model, X_test_t, y_test_t, device=device)
    return float(test_acc)


def main():
    parser = argparse.ArgumentParser(
        description="Вычислить downstream-оценки probe-модели (простая MLP probe) для каждого файла эмбеддингов по моделям."
    )

    # Два режима:
    # 1) Отложенный режим: embeddings_dir содержит ОДНО разделение, мы выполняем разделение на обучающую и валидационную выборки внутри него.
    # 2) Обучающий/тестовый режим: train_embeddings_dir + test_embeddings_dir.
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=None,
        help="Режим holdout: папка с эмбеддингами моделей для одного сплита.",
    )
    parser.add_argument(
        "--train_embeddings_dir",
        type=str,
        default=None,
        help="Режим train/test: папка с TRAIN-эмбеддингами моделей.",
    )
    parser.add_argument(
        "--test_embeddings_dir",
        type=str,
        default=None,
        help="Режим train/test: папка с TEST-эмбеддингами моделей.",
    )

    parser.add_argument(
        "--labels_path",
        type=str,
        default=None,
        help="Необязательный путь к меткам (.npy или .npz) в том же порядке, что и эмбеддинги. "
        "Если не задан, метки будут загружены через datasets.io.load_labels по --dataset и --data_root.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Имя датасета (например, cifar10). Если не задано, будет предпринята попытка вывести его из имени папки эмбеддингов.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Корневая папка данных для загрузки меток через datasets.io.load_labels.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--probe_epochs",
        type=int,
        default=50,
        help="Максимальное число эпох обучения MLP-probe.",
    )
    parser.add_argument(
        "--probe_batch_size",
        type=int,
        default=256,
        help="Размер батча для обучения MLP-probe.",
    )
    parser.add_argument(
        "--probe_lr", type=float, default=1e-3, help="Скорость обучения для MLP-probe."
    )
    parser.add_argument(
        "--probe_weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay для MLP-probe.",
    )
    parser.add_argument(
        "--probe_dropout", type=float, default=0.3, help="Доля dropout для MLP-probe."
    )

    parser.add_argument(
        "--probe_val_size",
        type=float,
        default=0.1,
        help="Размер валидационной части, выделяемой из TRAIN.",
    )
    parser.add_argument(
        "--probe_patience",
        type=int,
        default=7,
        help="Терпение early stopping по валидационной точности.",
    )
    parser.add_argument(
        "--probe_min_delta",
        type=float,
        default=0.0,
        help="Минимальное улучшение val accuracy для сброса счётчика терпения.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="num_workers для DataLoader. По умолчанию 0 для детерминизма на разных платформах.",
    )

    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="",
        help="Если задано, результаты будут записаны в стандартную структуру внутри experiment_dir.",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="",
        help="Выходной json-файл с downstream-оценками по моделям. Если пусто, путь выводится из --experiment_dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Пересчитывать downstream-оценки даже если out_json уже существует.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="Необязательная метка для логов; на вычисления не влияет.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help=(
            "Необязательный фильтр моделей: comma-separated model ids или alias text20. "
            "Если пусто, используются все общие embedding-файлы."
        ),
    )

    args = parser.parse_args()
    requested_models = _parse_model_filter(args.models)

    if not args.out_json:
        if not args.experiment_dir:
            raise ValueError("Нужно указать либо --out_json, либо --experiment_dir.")
        dstdir = os.path.join(args.experiment_dir, "downstream")
        os.makedirs(dstdir, exist_ok=True)
        ds = _resolve_dataset_name(
            args.dataset,
            args.embeddings_dir,
            args.train_embeddings_dir,
            args.test_embeddings_dir,
        )
        if not ds:
            raise ValueError(
                "Не удалось вывести имя датасета для out_json. "
                "Укажи --dataset или --out_json явно."
            )
        args.out_json = os.path.join(dstdir, f"{ds}_mlp.json")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    if os.path.exists(args.out_json) and not args.overwrite:
        print(f"Пропуск downstream: файл уже существует: {args.out_json}")
        return

    use_train_test = (
        args.train_embeddings_dir is not None and args.test_embeddings_dir is not None
    )
    use_holdout = args.embeddings_dir is not None

    if (use_train_test and use_holdout) or (not use_train_test and not use_holdout):
        raise ValueError(
            "Укажи ровно один режим: либо (--train_embeddings_dir И --test_embeddings_dir), либо --embeddings_dir."
        )

    _set_determinism(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = _resolve_dataset_name(
        args.dataset,
        args.embeddings_dir,
        args.train_embeddings_dir,
        args.test_embeddings_dir,
    )

    results: Dict[str, float] = {}

    if use_train_test:
        train_models = _list_models(args.train_embeddings_dir)
        test_models = _list_models(args.test_embeddings_dir)

        common = sorted(set(train_models.keys()) & set(test_models.keys()))
        if requested_models is not None:
            missing = [m for m in requested_models if m not in common]
            if missing:
                raise RuntimeError(
                    f"--models содержит модели без train/test эмбеддингов: {missing}. "
                    f"Доступные общие модели: {common}"
                )
            common = [m for m in requested_models if m in set(common)]
        if not common:
            raise RuntimeError(
                "Нет общих моделей между train_embeddings_dir и test_embeddings_dir."
            )

        y_train, y_test = _resolve_labels_train_test(
            labels_path=args.labels_path,
            dataset_name=dataset_name,
            data_root=args.data_root,
            train_embeddings_dir=args.train_embeddings_dir,
            test_embeddings_dir=args.test_embeddings_dir,
        )

        for m in tqdm(common, desc="Модели", unit="model"):
            X_train = _load_embeddings_file(train_models[m])
            X_test = _load_embeddings_file(test_models[m])

            if X_train.shape[0] != y_train.shape[0]:
                raise RuntimeError(
                    f"Несовпадение размера меток: y_train={y_train.shape[0]} против X_train={X_train.shape[0]} для модели {m}"
                )
            if X_test.shape[0] != y_test.shape[0]:
                raise RuntimeError(
                    f"Несовпадение размера меток: y_test={y_test.shape[0]} против X_test={X_test.shape[0]} для модели {m}"
                )

            score = _eval_probe_train_test(
                X_train,
                y_train,
                X_test,
                y_test,
                val_size=args.probe_val_size,
                seed=args.seed,
                device=device,
                epochs=args.probe_epochs,
                batch_size=args.probe_batch_size,
                lr=args.probe_lr,
                weight_decay=args.probe_weight_decay,
                dropout=args.probe_dropout,
                patience=args.probe_patience,
                min_delta=args.probe_min_delta,
                num_workers=args.num_workers,
            )
            results[m] = float(score)

    else:
        models = _list_models(args.embeddings_dir)
        if requested_models is not None:
            missing = [m for m in requested_models if m not in models]
            if missing:
                raise RuntimeError(
                    f"--models содержит модели без эмбеддингов: {missing}. "
                    f"Доступные модели: {sorted(models.keys())}"
                )
            models = {m: models[m] for m in requested_models}

        y = _resolve_labels_holdout(
            labels_path=args.labels_path,
            dataset_name=dataset_name,
            data_root=args.data_root,
            embeddings_dir=args.embeddings_dir,
        )

        for m in tqdm(sorted(models.keys()), desc="Модели", unit="model"):
            X = _load_embeddings_file(models[m])
            if X.shape[0] != y.shape[0]:
                raise RuntimeError(
                    f"Несовпадение размера меток: y={y.shape[0]} против X={X.shape[0]} для модели {m}"
                )

            score = _eval_probe_holdout(
                X,
                y,
                val_size=args.probe_val_size,
                seed=args.seed,
                device=device,
                epochs=args.probe_epochs,
                batch_size=args.probe_batch_size,
                lr=args.probe_lr,
                weight_decay=args.probe_weight_decay,
                dropout=args.probe_dropout,
                patience=args.probe_patience,
                min_delta=args.probe_min_delta,
                num_workers=args.num_workers,
            )
            results[m] = float(score)

    # Для совместимости оставим task="main", если task_name не задан.
    task = args.task_name if args.task_name else "main"
    out: Dict[str, Dict[str, float]] = {m: {task: float(s)} for m, s in results.items()}

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Сохранён downstream json: {args.out_json}")


if __name__ == "__main__":
    main()
