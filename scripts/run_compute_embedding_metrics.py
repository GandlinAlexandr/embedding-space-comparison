"""
run_compute_embedding_metrics.py

Считает embedding-метрики между всеми парами моделей на ОДНОМ и том же датасете эмбеддингов,
и сохраняет результаты на диск.

Поддерживаемый сценарий:
- Есть папка embeddings_dir, в которой лежат файлы эмбеддингов для каждой модели.
- Для каждой метрики (конфиг из metric_configs.py) считаем матрицу pairwise значений:
    score[i, j] = metric(emb_i, emb_j)
  где emb_i и emb_j — эмбеддинги одной и той же выборки объектов, но полученные разными моделями.

ВАЖНО:
- Здесь только: загрузка эмбеддингов, подвыборка, перебор пар, сохранение результатов.

Форматы эмбеддингов:
- .npy: ожидается массив (N, D)
- .npz: пытаемся найти массив в ключах: "embeddings", "X", "arr_0"

НОВОЕ (ИНКРЕМЕНТ):
- Можно не пересчитывать всю матрицу при добавлении новых моделей.
- Флаг --incremental:
    * если файл метрики уже существует, мы расширяем матрицу (старый блок НЕ трогаем)
      и досчитываем ТОЛЬКО пары с новыми моделями.
    * если файл не существует — считаем как обычно.
- Для antisym-метрик в файле хранится уже антисимметричная матрица A.
  В incremental-режиме мы оставляем старый блок A_old как есть и досчитываем только новые пары,
  заполняя A[i,j] = m(i->j) - m(j->i).
- Для sym-метрик в файле хранится симметричная матрица sim.
  В incremental-режиме мы оставляем старый блок как есть и досчитываем только новые пары,
  заполняя sim[i,j] = 0.5*(m(i->j)+m(j->i)) и симметризуя.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import svd
from scipy.spatial.distance import cdist, pdist
from tqdm import tqdm

# ВАЖНО: запускаем как модуль: python -m scripts.run_compute_embedding_metrics
from configs.metric_configs import get_embedding_metric_configs


# ============================================================
# 0) Вспомогательные функции: загрузка эмбеддингов
# ============================================================


def _load_embeddings(path: str) -> np.ndarray:
    """
    Загружает эмбеддинги из .npy или .npz в массив float32 формы (N, D).
    """
    if path.endswith(".npy"):
        arr = np.load(path)
        return np.asarray(arr, dtype=np.float32)

    if path.endswith(".npz"):
        z = np.load(path)
        # Сначала пробуем стандартные ключи, иначе берём первый массив.
        for k in ["embeddings", "X", "arr_0"]:
            if k in z.files:
                return np.asarray(z[k], dtype=np.float32)
        return np.asarray(z[z.files[0]], dtype=np.float32)

    raise ValueError(f"Неподдерживаемый файл эмбеддингов: {path}")


def _list_models(embeddings_dir: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Возвращает: отсортированный список model_names и словарь name->file_path.
    """
    files = []
    for fn in os.listdir(embeddings_dir):
        if fn.endswith(".npy") or fn.endswith(".npz"):
            files.append(fn)

    if not files:
        raise RuntimeError(f"В {embeddings_dir} не найдено файлов .npy/.npz")

    model_to_path: Dict[str, str] = {}
    for fn in sorted(files):
        name = os.path.splitext(fn)[0]
        model_to_path[name] = os.path.join(embeddings_dir, fn)

    return sorted(model_to_path.keys()), model_to_path


def _model_list_paths(out_dir: str) -> Tuple[str, str]:
    """
    Пути к manifest-файлам со списком моделей, лежащим рядом с таблицами метрик.
    """
    json_path = os.path.join(out_dir, "model_names.json")
    txt_path = os.path.join(out_dir, "model_names.txt")
    return json_path, txt_path


def _load_saved_model_list(out_dir: str) -> List[str]:
    """
    Загружает ранее сохранённый список моделей из out_dir.
    Если файла нет — возвращает пустой список.
    """
    json_path, txt_path = _model_list_paths(out_dir)

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "model_names" in data:
            names = data["model_names"]
        else:
            names = data
        return [str(x) for x in names]

    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        return names

    return []


def _save_model_list(out_dir: str, model_names: List[str]) -> None:
    """
    Сохраняет список моделей рядом с таблицами метрик.
    """
    json_path, txt_path = _model_list_paths(out_dir)

    payload = {"model_names": list(model_names)}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        for name in model_names:
            f.write(f"{name}\n")


def _merge_model_lists_preserve_order(
    current_names: List[str], saved_names: List[str]
) -> List[str]:
    """
    Объединяет текущий список моделей с уже сохранённым manifest-списком.

    Логика:
    - сначала сохраняем старый порядок saved_names;
    - затем в конец дописываем новые модели из current_names, которых раньше не было.
    """
    merged: List[str] = []
    seen = set()

    for name in saved_names:
        if name not in seen:
            merged.append(name)
            seen.add(name)

    for name in current_names:
        if name not in seen:
            merged.append(name)
            seen.add(name)

    return merged


# ============================================================
# 1) RankMe (мягкий ранг) и метрика локального ранга отображения
# ============================================================


def rankme(s: np.ndarray) -> float:
    norm_1 = np.sum(np.abs(s))
    p_k = np.abs(s) / (norm_1 + 1e-10)
    entropy = -np.sum(p_k * np.log(p_k + 1e-10))
    return float(np.exp(entropy))


def _solve_local_linear_map_and_rank(Xc: np.ndarray, Yc: np.ndarray) -> float:
    """
    Решает Xc * M ≈ Yc и возвращает RankMe по сингулярным значениям M.
    """
    # lstsq быстрее, чем pinv-dot, на малых окрестностях.
    M, *_ = np.linalg.lstsq(Xc, Yc, rcond=None)
    s = svd(M, full_matrices=False, compute_uv=False)
    return rankme(s)


def _rff_features(
    X: np.ndarray, n_features: int = 256, gamma: float = 1.0, seed: int = 42
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    d = X.shape[1]
    W = rng.normal(loc=0.0, scale=np.sqrt(2 * gamma), size=(d, n_features)).astype(
        np.float32
    )
    b = rng.uniform(low=0.0, high=2 * np.pi, size=(n_features,)).astype(np.float32)
    Z = np.sqrt(2.0 / n_features) * np.cos(X @ W + b)
    return Z.astype(np.float32)


# ============================================================
# 2) Разбор конфигов (устойчивый к текущей схеме именования)
# ============================================================


@dataclass(frozen=True)
class MetricSpec:
    name: str
    kind: str  # "linear_knn" | "linear_eps" | "multiscale_knn" | "rff_knn"
    pair_agg: str  # "directed" | "antisym" | "sym"

    # параметры окрестности
    k: Optional[int] = None
    eps_percentile: Optional[int] = None
    k_list: Optional[Tuple[int, ...]] = None
    aggregator: str = "mean"

    # параметры глобального усреднения
    n_centers: int = 200

    # параметры RFF
    rff_n_features: int = 256
    rff_gamma: float = 1.0
    rff_seed: int = 42


def _infer_metric_spec(name: str, cfg: Any, default_n_centers: int = 200) -> MetricSpec:
    """
    Восстанавливает параметры по имени конфига и словарю конфигурации (если он есть).
    Поддерживает текущую схему именования:
      - local_map_rank_linear_knn_k10
      - ..._antisym
      - ..._sym
      - ...eps_percentile_10...
      - ...multiscale_knn_mean...
      - ...rff_knn_k10...
    """
    lower_name = name.lower()
    if "antisym" in lower_name:
        pair_agg = "antisym"
    elif re.search(r"(?:^|_)sym(?:$|_)", lower_name):
        pair_agg = "sym"
    else:
        pair_agg = "directed"

    # Пытаемся прочитать n_centers из словаря cfg, если он задан.
    n_centers = default_n_centers
    if isinstance(cfg, dict):
        for key in ["n_centers", "n_samples", "N_SAMPLES", "num_centers"]:
            if key in cfg:
                try:
                    n_centers = int(cfg[key])
                except Exception:
                    pass

    lower = name.lower()

    # linear knn
    m = re.search(r"linear_knn_k(\d+)", lower)
    if m:
        k = int(m.group(1))
        return MetricSpec(
            name=name, kind="linear_knn", pair_agg=pair_agg, k=k, n_centers=n_centers
        )

    # linear epsilon
    m = re.search(r"linear_eps_percentile_(\d+)", lower)
    if m:
        p = int(m.group(1))
        return MetricSpec(
            name=name,
            kind="linear_eps",
            pair_agg=pair_agg,
            eps_percentile=p,
            n_centers=n_centers,
        )

    # multiscale knn
    m = re.search(r"multiscale_knn_(mean|median|min|max)", lower)
    if m:
        agg = m.group(1)
        # Обычно k_list лежит в meta, но если его нет, используем значение по умолчанию.
        k_list = None
        if isinstance(cfg, dict):
            meta = cfg.get("meta", {})
            if isinstance(meta, dict) and "k_list" in meta:
                try:
                    k_list = tuple(int(x) for x in meta["k_list"])
                except Exception:
                    k_list = None
        if k_list is None:
            k_list = (5, 10, 20, 40)
        return MetricSpec(
            name=name,
            kind="multiscale_knn",
            pair_agg=pair_agg,
            k_list=k_list,
            aggregator=agg,
            n_centers=n_centers,
        )

    # rff knn
    m = re.search(r"rff_knn_k(\d+)", lower)
    if m:
        k = int(m.group(1))
        # Параметры RFF берём из meta, либо используем значения по умолчанию.
        rff_n_features = 256
        rff_gamma = 1.0
        rff_seed = 42
        if isinstance(cfg, dict):
            meta = cfg.get("meta", {})
            if isinstance(meta, dict):
                rff_n_features = int(meta.get("n_features", rff_n_features))
                rff_gamma = float(meta.get("gamma", rff_gamma))
                rff_seed = int(meta.get("rff_seed", rff_seed))
        return MetricSpec(
            name=name,
            kind="rff_knn",
            pair_agg=pair_agg,
            k=k,
            n_centers=n_centers,
            rff_n_features=rff_n_features,
            rff_gamma=rff_gamma,
            rff_seed=rff_seed,
        )

    raise ValueError(f"Не удалось восстановить спецификацию метрики по имени: {name}")


# ============================================================
# 3) Кэш окрестностей для заданной модели (центры + соседи)
# ============================================================


@dataclass
class NeighborCache:
    centers: np.ndarray  # (C, D)
    knn: Dict[int, np.ndarray]  # k -> (C, k) индексы
    eps: Dict[
        int, np.ndarray
    ]  # percentile -> список индексов, упакованный в object-массив
    X_norm: np.ndarray  # (N, D) нормализованные данные
    eps_values: Dict[int, float]  # percentile -> скалярный eps


def _zscore_rows(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sigma


def _build_neighbor_cache(
    X: np.ndarray,
    spec: MetricSpec,
    seed: int = 42,
) -> NeighborCache:
    """
    Предвычисляет:
    - centers: случайное подмножество строк (размер = spec.n_centers, либо N если меньше)
    - индексы k ближайших соседей для каждого центра (для k и/или k_list)
    - индексы eps-окрестностей для каждого центра (если используется eps_percentile)
    """
    rng = np.random.RandomState(seed)
    N = X.shape[0]

    # Нормализуем один раз для вычисления расстояний.
    Xn = _zscore_rows(X)

    # Центры.
    C = min(spec.n_centers, N)
    centers_idx = rng.choice(N, size=C, replace=False)
    centers = Xn[centers_idx]

    knn: Dict[int, np.ndarray] = {}
    eps: Dict[int, np.ndarray] = {}
    eps_values: Dict[int, float] = {}

    # Какие k нам нужны.
    ks: List[int] = []
    if spec.kind in {"linear_knn", "rff_knn"} and spec.k is not None:
        ks.append(spec.k)
    if spec.kind == "multiscale_knn" and spec.k_list is not None:
        ks.extend(list(spec.k_list))

    ks = sorted(set(ks))

    if ks:
        # Вычисляем расстояния от центров до всех точек.
        D = cdist(centers, Xn, metric="euclidean")
        # argpartition до Kmax
        Kmax = max(ks)
        nn = np.argpartition(D, kth=Kmax, axis=1)[:, :Kmax]
        # Сортируем внутри Kmax
        # для стабильного порядка соседей (это не критично).
        row = np.arange(nn.shape[0])[:, None]
        nn = nn[row, np.argsort(D[row, nn], axis=1)]
        for k in ks:
            knn[k] = nn[:, :k]

    # eps-окрестность:
    if spec.kind == "linear_eps" and spec.eps_percentile is not None:
        # Оцениваем eps по подвыборке попарных расстояний
        # (как у тебя в логах).
        sub_n = min(4000, N)
        sub_idx = rng.choice(N, size=sub_n, replace=False)
        d = pdist(Xn[sub_idx], metric="euclidean")
        eps_val = float(np.percentile(d, spec.eps_percentile))
        eps_values[spec.eps_percentile] = eps_val

        D = cdist(centers, Xn, metric="euclidean")
        for p in [spec.eps_percentile]:
            mask = D <= eps_val
            neigh = []
            for r in range(mask.shape[0]):
                neigh.append(np.where(mask[r])[0].astype(np.int32))
            eps[p] = np.array(neigh, dtype=object)

    return NeighborCache(
        centers=centers,
        knn=knn,
        eps=eps,
        X_norm=Xn,
        eps_values=eps_values,
    )


# ============================================================
# 4) Направленная метрика m(X->Y) для пары
# ============================================================


def _metric_directed_for_pair(
    spec: MetricSpec,
    X: np.ndarray,
    Y: np.ndarray,
    cache_X: NeighborCache,
    seed: int = 42,
) -> float:
    """
    Вычисляет направленную m(X->Y) как среднее по центрам:
      - выбираем окрестность в X вокруг каждого центра (kNN или eps)
      - берём соответствующие строки в Y (те же индексы)
      - решаем локальное линейное отображение и считаем RankMe по сингулярным значениям
    """
    Xn = cache_X.X_norm
    Yn = _zscore_rows(Y)

    if spec.kind == "rff_knn":
        Xn = _rff_features(
            Xn,
            n_features=spec.rff_n_features,
            gamma=spec.rff_gamma,
            seed=spec.rff_seed,
        )
        Yn = _rff_features(
            Yn,
            n_features=spec.rff_n_features,
            gamma=spec.rff_gamma,
            seed=spec.rff_seed,
        )

    vals = []

    if spec.kind in {"linear_knn", "rff_knn"}:
        assert spec.k is not None
        nn = cache_X.knn[spec.k]
        for idxs in nn:
            Xc = Xn[idxs]
            Yc = Yn[idxs]
            vals.append(_solve_local_linear_map_and_rank(Xc, Yc))

    elif spec.kind == "multiscale_knn":
        assert spec.k_list is not None
        per_scale = []
        for k in spec.k_list:
            nn = cache_X.knn[int(k)]
            tmp = []
            for idxs in nn:
                Xc = Xn[idxs]
                Yc = Yn[idxs]
                tmp.append(_solve_local_linear_map_and_rank(Xc, Yc))
            per_scale.append(np.asarray(tmp, dtype=np.float32))
        # Агрегируем масштабы.
        stack = np.stack(per_scale, axis=0)  # (S, C)
        if spec.aggregator == "mean":
            vals = list(np.mean(stack, axis=0))
        elif spec.aggregator == "median":
            vals = list(np.median(stack, axis=0))
        elif spec.aggregator == "min":
            vals = list(np.min(stack, axis=0))
        elif spec.aggregator == "max":
            vals = list(np.max(stack, axis=0))
        else:
            raise ValueError(f"Неизвестный агрегатор: {spec.aggregator}")

    elif spec.kind == "linear_eps":
        assert spec.eps_percentile is not None
        neigh = cache_X.eps[spec.eps_percentile]
        for idxs in neigh:
            if idxs.size < 2:
                continue
            Xc = Xn[idxs]
            Yc = Yn[idxs]
            vals.append(_solve_local_linear_map_and_rank(Xc, Yc))

    else:
        raise ValueError(f"Неизвестный spec.kind: {spec.kind}")

    if not vals:
        return float("nan")
    return float(np.mean(vals))


# ============================================================
# 5) Вспомогательные функции для инкрементального ввода-вывода (.npz)
# ============================================================


def _load_existing_metric_npz(
    path: str,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    if "matrix" in data.files:
        M = np.asarray(data["matrix"], dtype=np.float32)
    elif "scores" in data.files:
        M = np.asarray(data["scores"], dtype=np.float32)
    else:
        raise KeyError(
            f"В существующем файле метрики нет ни 'matrix', ни 'scores': {path}"
        )

    if "model_names" not in data.files:
        raise KeyError(
            f"В существующем файле метрики отсутствует 'model_names': {path}"
        )
    names = list(data["model_names"].tolist())

    meta: Dict[str, Any] = {}
    if "meta_json" in data.files:
        mj = data["meta_json"]
        mj = mj.item() if getattr(mj, "shape", None) == () else mj.tolist()
        if isinstance(mj, str):
            try:
                meta = json.loads(mj)
            except Exception:
                meta = {}
        elif isinstance(mj, dict):
            meta = mj

    return M, names, meta


def _ensure_meta_compatible(meta_old: Dict[str, Any], new_spec: MetricSpec) -> None:
    """
    В режиме инкрементного вычисления: запрещаем расширение, если metric_spec отличается.
    """
    try:
        old_spec = meta_old.get("metric_spec", None)
        if old_spec is None:
            return
        if old_spec != asdict(new_spec):
            raise RuntimeError(
                "Инкрементальный режим: у существующего файла метрики другой metric_spec.\n"
                f"Существующий: {old_spec}\n"
                f"Новый:        {asdict(new_spec)}\n"
                "Расширение отменено, чтобы не смешивать несовместимые матрицы."
            )
    except Exception:
        # Если анализ метаданных неисправен, не надо сразу падать; всё же безопаснее продолжить?
        # Мы выбираем быть строгими ТОЛЬКО тогда, когда можем надёжно сравнить данные.
        return


def _build_model_list_incremental(current: List[str], old: List[str]) -> List[str]:
    out = list(old)
    seen = set(old)
    for m in current:
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def _extend_matrix_with_old_block(M_old: np.ndarray, n_total: int) -> np.ndarray:
    M_new = np.full((n_total, n_total), np.nan, dtype=np.float32)
    n_old = M_old.shape[0]
    M_new[:n_old, :n_old] = M_old
    return M_new


# ============================================================
# 6) Основной запуск
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Вычислить попарные метрики эмбеддингов для всех файлов моделей."
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        required=True,
        help="Папка с эмбеддингами моделей (.npy/.npz).",
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="",
        help="Если задано, записывать результаты в стандартную структуру внутри этой папки эксперимента.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Куда сохранять вычисленные матрицы метрик (.npz). Если пусто, путь берётся из --experiment_dir.",
    )
    parser.add_argument(
        "--include",
        type=str,
        default="",
        help="Имена конфигов для включения через запятую (пусто = все).",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Имена конфигов для исключения через запятую.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Случайный seed для подвыборки строк."
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Если флаг задан и файл метрики существует, расширить его и вычислить только отсутствующие пары (старый блок не трогать).",
    )
    args = parser.parse_args()

    if not args.out_dir:
        if not args.experiment_dir:
            raise ValueError("Нужно указать либо --out_dir, либо --experiment_dir.")
        args.out_dir = os.path.join(args.experiment_dir, "metric_matrices")

    os.makedirs(args.out_dir, exist_ok=True)

    model_names_current, model_to_path = _list_models(args.embeddings_dir)

    # Общий manifest списка моделей в out_dir:
    # - создаётся автоматически;
    # - при добавлении новых моделей дополняется;
    # - лежит рядом с .npz таблицами.
    saved_model_names = _load_saved_model_list(args.out_dir)
    manifest_model_names = _merge_model_lists_preserve_order(
        model_names_current, saved_model_names
    )
    _save_model_list(args.out_dir, manifest_model_names)

    print(f"Найдено {len(model_names_current)} моделей в embeddings_dir:")
    for m in model_names_current:
        print(f"  - {m}")

    print(
        f"\nСохранён / обновлён общий список моделей в out_dir: {len(manifest_model_names)}"
    )
    for name in manifest_model_names:
        print(f"  * {name}")

    cfgs = get_embedding_metric_configs()

    include = (
        [x.strip() for x in args.include.split(",") if x.strip()]
        if args.include
        else []
    )
    exclude = (
        set([x.strip() for x in args.exclude.split(",") if x.strip()])
        if args.exclude
        else set()
    )

    chosen = []
    for name in cfgs.keys():
        if include and name not in include:
            continue
        if name in exclude:
            continue
        chosen.append(name)

    if not chosen:
        raise RuntimeError(
            "Не выбрано ни одного конфига метрик. Проверь --include/--exclude."
        )

    print(f"\nБудет вычислено {len(chosen)} конфигураций метрик:")
    for name in chosen:
        print(f"  - {name}")

    # Загружаем все эмбеддинги в память один раз.
    embeddings: Dict[str, np.ndarray] = {}
    N0 = None
    for m in model_names_current:
        X = _load_embeddings(model_to_path[m])
        embeddings[m] = X
        if N0 is None:
            N0 = X.shape[0]
        else:
            if X.shape[0] != N0:
                raise RuntimeError(
                    f"У всех эмбеддингов должно совпадать N. У {m}: {X.shape[0]} против {N0}"
                )

    # Необязательная подвыборка,
    # делаем ОДИН общий индекс для всех моделей, чтобы пары были сопоставимы.
    sample_sizes = []
    for name in chosen:
        cfg = cfgs[name]
        ss = cfg.get("sample_size", None) if isinstance(cfg, dict) else None
        if ss is not None:
            sample_sizes.append(int(ss))
    sample_size = min(sample_sizes) if sample_sizes else None

    if sample_size is not None and sample_size > 0 and sample_size < N0:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(N0, size=sample_size, replace=False)
        idx.sort()
        for m in embeddings.keys():
            embeddings[m] = embeddings[m][idx]
        print(f"\nПодвыбраны строки: N={N0} -> {sample_size}")

    # Вычисление отдельно для каждой метрики.
    for name in chosen:
        cfg = cfgs[name]
        spec = _infer_metric_spec(
            name,
            cfg.get("meta", {}) if isinstance(cfg, dict) else cfg,
            default_n_centers=200,
        )

        out_path = os.path.join(args.out_dir, f"{name}.npz")
        # ------------------------------------------------------------
        # Создание кэшей для каждой модели (центры + соседи) один раз для каждой метрики.
        # Важно: кэш зависит от спецификации (k / eps / multiscale / rff).
        # ------------------------------------------------------------

        # Создаём "главные" кэши на основе нормализованного X (zscore) и повторно используем индексы для всех пар.
        # Для повышения скорости: кэш для каждой модели i использует X_i для соседей (направленных).
        caches: Dict[str, NeighborCache] = {}

        def get_cache_for_model_i(model_i: str) -> NeighborCache:
            if model_i not in caches:
                caches[model_i] = _build_neighbor_cache(
                    embeddings[model_i], spec, seed=args.seed
                )
            return caches[model_i]

        # ============================================================
        # ИНКРЕМЕНТАЛЬНО: определить имена моделей и инициализировать матрицу
        # ============================================================
        if args.incremental and os.path.exists(out_path):
            M_old, names_old, meta_old = _load_existing_metric_npz(out_path)
            _ensure_meta_compatible(meta_old, spec)

            # Привязать к старому порядку, добавить новые модели, существующие в embeddings_dir
            model_names = _build_model_list_incremental(model_names_current, names_old)

            # Дополнительно обновляем общий manifest в директории:
            # если у конкретной таблицы был более старый список, сохраняем его порядок
            # и дописываем новые модели в конец.
            manifest_model_names = _merge_model_lists_preserve_order(
                model_names, _load_saved_model_list(args.out_dir)
            )
            _save_model_list(args.out_dir, manifest_model_names)

            # Расширить матрицу (старый блок копируется как есть)
            out_matrix = _extend_matrix_with_old_block(M_old, n_total=len(model_names))

            print(
                f"\n[INCREMENTAL] Расширяем существующий файл метрики: {os.path.basename(out_path)}"
            )
            print(
                f"[INCREMENTAL] Старые модели: {len(names_old)} | Текущие модели: {len(model_names_current)} | Всего: {len(model_names)}"
            )
            if len(model_names) == len(names_old):
                print(
                    "[INCREMENTAL] Новых моделей не обнаружено; файл будет только (пере)сохранён как есть (без пересчёта)."
                )
        else:
            # Обычный режим: используем только текущие модели
            model_names = list(model_names_current)
            out_matrix = np.full(
                (len(model_names), len(model_names)), np.nan, dtype=np.float32
            )

            manifest_model_names = _merge_model_lists_preserve_order(
                model_names, _load_saved_model_list(args.out_dir)
            )
            _save_model_list(args.out_dir, manifest_model_names)

        # ============================================================
        # Вычисление значений
        # - directed: out_matrix — направленная M (может быть несимметричной)
        # - antisym: out_matrix — антисимметричная A
        # - sym: out_matrix — симметричная sim
        # ============================================================

        if spec.pair_agg == "antisym":
            # out_matrix хранит A. Мы никогда не трогаем старый блок. Вычисляем только NaN.
            # Для пары (i, j) нужны оба направленных значения: m(i->j) и m(j->i).
            # Тогда A[i,j] = m(i->j) - m(j->i), A[j,i] = -A[i,j].
            # Диагональ равна 0.
            for i, mi in enumerate(tqdm(model_names, desc="Модель i", unit="model")):
                # Если в строке нигде нет NaN, можно быстро пропустить шаг.
                if not np.isnan(out_matrix[i]).any():
                    continue

                Xi = embeddings[mi]
                cache_i = get_cache_for_model_i(mi)

                for j, mj in enumerate(model_names):
                    if i == j:
                        if np.isnan(out_matrix[i, j]):
                            out_matrix[i, j] = np.float32(0.0)
                        continue

                    # Если вычисления уже сделаны (с обеих сторон), ничего не делаем.
                    if not np.isnan(out_matrix[i, j]) and not np.isnan(
                        out_matrix[j, i]
                    ):
                        continue

                    # Вычисляем в обоих направлениях
                    Yj = embeddings[mj]
                    # m(i->j)
                    mij = _metric_directed_for_pair(
                        spec, Xi, Yj, cache_i, seed=args.seed
                    )

                    # m(j->i)
                    Xj = embeddings[mj]
                    cache_j = get_cache_for_model_i(mj)
                    mji = _metric_directed_for_pair(
                        spec, Xj, Xi, cache_j, seed=args.seed
                    )

                    aij = np.float32(mij - mji)
                    out_matrix[i, j] = aij
                    out_matrix[j, i] = np.float32(-aij)

            # Обеспечивает точную антисимметрию и нулевую диагональ
            np.fill_diagonal(out_matrix, 0.0)

        elif spec.pair_agg == "sym":
            # out_matrix хранит симметричные значения sim. Мы никогда не трогаем старый блок. Вычисляем только NaN.
            # Для пары (i, j) нужны оба направленных значения: m(i->j) и m(j->i).
            # Тогда sim[i,j] = 0.5*(m(i->j) + m(j->i)), sim[j,i] = sim[i,j].
            # Диагональ равна 0.
            for i, mi in enumerate(tqdm(model_names, desc="Модель i", unit="model")):
                # Если в строке нигде нет NaN, можно быстро пропустить шаг.
                if not np.isnan(out_matrix[i]).any():
                    continue

                Xi = embeddings[mi]
                cache_i = get_cache_for_model_i(mi)

                for j, mj in enumerate(model_names):
                    if i == j:
                        if np.isnan(out_matrix[i, j]):
                            out_matrix[i, j] = np.float32(0.0)
                        continue

                    # Если вычисления уже сделаны (с обеих сторон), ничего не делаем.
                    if not np.isnan(out_matrix[i, j]) and not np.isnan(
                        out_matrix[j, i]
                    ):
                        continue

                    Yj = embeddings[mj]
                    mij = _metric_directed_for_pair(
                        spec, Xi, Yj, cache_i, seed=args.seed
                    )

                    Xj = embeddings[mj]
                    cache_j = get_cache_for_model_i(mj)
                    mji = _metric_directed_for_pair(
                        spec, Xj, Xi, cache_j, seed=args.seed
                    )

                    sij = np.float32(0.5 * (mij + mji))
                    out_matrix[i, j] = sij
                    out_matrix[j, i] = sij

            # Обеспечиваем симметрию и нулевую диагональ
            np.fill_diagonal(out_matrix, 0.0)

        else:
            # out_matrix хранит направленные M. Старый блок не трогаем, только заполняем NaN.
            for i, mi in enumerate(tqdm(model_names, desc="Модель i", unit="model")):
                # Если в строке нигде нет NaN, можно быстро пропустить шаг.
                if not np.isnan(out_matrix[i]).any():
                    continue

                Xi = embeddings[mi]
                cache_i = get_cache_for_model_i(mi)

                for j, mj in enumerate(model_names):
                    if not np.isnan(out_matrix[i, j]):
                        continue
                    Yj = embeddings[mj]
                    val = _metric_directed_for_pair(
                        spec, Xi, Yj, cache_i, seed=args.seed
                    )
                    out_matrix[i, j] = np.float32(val)

        # Флаги:
        # - pair_agg="directed": направленная величина m(X->Y)
        # - pair_agg="antisym": s(X,Y)=m(X->Y)-m(Y->X) (антисимметричная)
        # - pair_agg="sym":     sim(X,Y)=0.5*(m(X->Y)+m(Y->X)) (симметричная)
        # поэтому здесь ставим True для antisym и sym.
        meta = {
            "metric_name": spec.name,
            "is_paired": True,
            "is_symmetric": bool(spec.pair_agg in {"antisym", "sym"}),
            "pair_agg": spec.pair_agg,
            "metric_spec": asdict(spec),
            "metric_config": cfg.get("meta", {}) if isinstance(cfg, dict) else {},
        }

        np.savez_compressed(
            out_path,
            matrix=out_matrix.astype(np.float32),
            model_names=np.array(model_names, dtype=object),
            meta_json=json.dumps(meta, ensure_ascii=False),
        )
        print(f"Сохранено: {out_path}")

    print("\nГотово.")


if __name__ == "__main__":
    main()
