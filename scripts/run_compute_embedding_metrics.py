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

АГРЕГАЦИЯ РАНГА:
- По умолчанию используется формула RankMe (энтропийная, из статьи):
    rankme(s) = exp(-sum(p_k * log(p_k))), где p_k = s_k / sum(s)
  Значение интерпретируется как "эффективное число измерений", от 1 до D.
- Альтернатива: hard_rank — количество сингулярных значений выше абсолютного порога.
  Значение интерпретируемо напрямую: "матрица M имеет ранг N".
  Управляется через параметры rank_aggregation и hard_rank_threshold в конфиге метрики.
- Старые .npz-файлы (посчитанные с rankme) при добавлении hard_rank-конфигов не затрагиваются:
  новые конфиги записываются в отдельные файлы.

АРТЕФАКТЫ:
- Вместе с матрицей метрики всегда сохраняется файл
  artifacts/{metric_name}_artifacts.npz внутри out_dir.
- Артефакты содержат сырые данные по каждому центру для каждого направления (i->j):
    singular_values: (n_centers, d) — сингулярные значения матрицы M
    residuals:       (n_centers,)   — legacy-невязка ||(Xc-xc) @ M - (Yc-yc)||_F
    ranks:           (n_centers,)   — legacy hard-rank по относительному порогу
- Для новых конфигов и диагностики дополнительно сохраняются:
    relative_residuals: (n_centers,) — относительная ошибка по локальным отклонениям Y - yc
    metric_ranks:       (n_centers,) — ранг в той же агрегации, что и сама метрика
- Для расширенной диагностики новых методов также сохраняются:
    neighbor_sizes:  (n_centers,)   — сколько точек вошло в окрестность
    neighbor_distances: object      — расстояния до точек окрестности по центрам
    sigma_values:    (n_centers,)   — использованный sigma (если применимо)
    eps_values:      (n_centers,)   — использованный eps (если применимо)
    sample_weights:  object         — веса точек в окрестности (если применимо)
    inlier_masks:    object         — mask инлайеров после robust-solver (если применимо)
    inlier_counts:   (n_centers,)   — число инлайеров
    inlier_fracs:    (n_centers,)   — доля инлайеров
- Инкрементальность артефактов синхронна с инкрементальностью матрицы:
  если пара уже посчитана (не NaN в матрице), артефакты для неё тоже уже есть.
- Ключи в файле артефактов: "{model_i}_to_{model_j}/{поле}"
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict, replace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.linalg import svd
from scipy.spatial.distance import cdist, pdist
from tqdm import tqdm

try:
    import skdim
except ImportError:
    skdim = None

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
# 1) RankMe, hard_rank и метрика локального ранга отображения
# ============================================================


def rankme(s: np.ndarray) -> float:
    norm_1 = np.sum(np.abs(s))
    p_k = np.abs(s) / (norm_1 + 1e-10)
    entropy = -np.sum(p_k * np.log(p_k + 1e-10))
    return float(np.exp(entropy))


def hard_rank(s: np.ndarray, threshold: float = 1e-2) -> float:
    """
    Количество сингулярных значений, превышающих абсолютный порог threshold.

    В отличие от RankMe, значение напрямую интерпретируемо:
    "матрица отображения M имеет ранг N".

    Важно: для kNN с k < D система недоопределена, и ранг ограничен сверху
    самим k — это честное отражение того, что окрестность слишком мала.
    """
    return float(np.sum(s > threshold))


def _aggregate_rank(
    s: np.ndarray,
    rank_aggregation: str,
    hard_rank_threshold: float,
) -> float:
    """
    Выбирает способ агрегации ранга по сингулярным значениям матрицы M.

    rank_aggregation:
      "rankme"    — энтропийная формула из статьи RankMe (значение от 1 до D,
                    интерпретируется как "эффективное число измерений").
      "hard_rank" — количество сингулярных значений выше порога hard_rank_threshold
                    (значение целочисленное, напрямую интерпретируемо как ранг).
    """
    if rank_aggregation == "hard_rank":
        return hard_rank(s, threshold=hard_rank_threshold)
    return rankme(s)


def _fit_local_linear_map(
    Xc: np.ndarray,
    Yc: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Решает Xc * M ≈ Yc обычным или взвешенным МНК.
    """
    if sample_weights is None:
        M, *_ = np.linalg.lstsq(Xc, Yc, rcond=None)
        return M

    w = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    w = np.clip(w, 1e-8, None)
    sqrt_w = np.sqrt(w)[:, None]
    M, *_ = np.linalg.lstsq(Xc * sqrt_w, Yc * sqrt_w, rcond=None)
    return M


def _residual_rows(Xc: np.ndarray, Yc: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Возвращает построчные евклидовы нормы невязки.
    """
    return np.linalg.norm(Xc @ M - Yc, axis=1)


def _robust_inlier_mask(
    row_residuals: np.ndarray,
    threshold_scale: float,
    min_inliers: int,
) -> np.ndarray:
    """
    Выделяет инлайеры по робастному порогу median + c * MAD.
    """
    med = float(np.median(row_residuals))
    mad = float(np.median(np.abs(row_residuals - med)))
    robust_sigma = 1.4826 * mad
    thr = med + threshold_scale * robust_sigma

    if not np.isfinite(thr) or thr <= 0.0:
        thr = float(np.quantile(row_residuals, 0.75))
    if not np.isfinite(thr) or thr <= 0.0:
        thr = max(med, 1e-8)

    mask = row_residuals <= thr
    if int(np.sum(mask)) >= min_inliers:
        return mask

    order = np.argsort(row_residuals)
    mask = np.zeros_like(row_residuals, dtype=bool)
    mask[order[:min_inliers]] = True
    return mask


def _fit_local_linear_map_ransac(
    Xc: np.ndarray,
    Yc: np.ndarray,
    sample_weights: Optional[np.ndarray],
    rng: np.random.RandomState,
    n_iter: int,
    sample_frac: float,
    min_inliers: int,
    threshold_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Упрощённый RANSAC для локального линейного отображения.

    Схема:
      - несколько раз выбираем подмножество строк;
      - оцениваем модель на подмножестве;
      - выделяем инлайеры по робастному порогу;
      - выбираем лучшую модель по числу инлайеров и их среднему residual;
      - переобучаемся только на найденных инлайерах.
    """
    n_rows = Xc.shape[0]
    if n_rows == 0:
        return np.zeros((Xc.shape[1], Yc.shape[1]), dtype=np.float32), np.zeros(
            (0,), dtype=bool
        )

    if n_rows <= 2:
        M = _fit_local_linear_map(Xc, Yc, sample_weights=sample_weights)
        return M, np.ones(n_rows, dtype=bool)

    weights = None
    probs = None
    if sample_weights is not None:
        weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        weights = np.clip(weights, 1e-8, None)
        probs = weights / np.sum(weights)

    min_inliers_eff = min(max(2, min_inliers), n_rows)
    sample_size = int(np.ceil(sample_frac * n_rows))
    sample_size = max(2, sample_size, min_inliers_eff)
    sample_size = min(sample_size, n_rows)

    best_mask = None
    best_score = None

    for _ in range(max(1, n_iter)):
        subset_idx = rng.choice(
            n_rows,
            size=sample_size,
            replace=False,
            p=probs,
        )
        subset_weights = None if weights is None else weights[subset_idx]
        try:
            M_candidate = _fit_local_linear_map(
                Xc[subset_idx],
                Yc[subset_idx],
                sample_weights=subset_weights,
            )
        except np.linalg.LinAlgError:
            continue

        row_res = _residual_rows(Xc, Yc, M_candidate)
        mask = _robust_inlier_mask(
            row_res,
            threshold_scale=threshold_scale,
            min_inliers=min_inliers_eff,
        )

        if weights is None:
            err = float(np.mean(row_res[mask])) if np.any(mask) else float("inf")
        else:
            err = float(np.average(row_res[mask], weights=weights[mask]))

        score = (int(np.sum(mask)), -err)
        if best_score is None or score > best_score:
            best_score = score
            best_mask = mask

    if best_mask is None:
        M = _fit_local_linear_map(Xc, Yc, sample_weights=sample_weights)
        return M, np.ones(n_rows, dtype=bool)

    final_weights = None if weights is None else weights[best_mask]
    M = _fit_local_linear_map(
        Xc[best_mask],
        Yc[best_mask],
        sample_weights=final_weights,
    )
    return M, best_mask


@dataclass
class LocalSolveResult:
    rank_value: float  # значение агрегированного ранга (rankme или hard_rank)
    singular_values: np.ndarray
    raw_residual: float
    relative_residual: float
    inlier_mask: np.ndarray


_LOCAL_GEOMETRY_MODE = "centered_offsets_v1"


def _local_geometry_meta() -> Dict[str, Any]:
    return {
        "local_geometry_mode": _LOCAL_GEOMETRY_MODE,
        "local_geometry_description": (
            "Локальная affine-линеаризация через центрирование: "
            "(Xc - xc) @ M ≈ (Yc - yc). Эквивалентно добавлению локального сдвига."
        ),
    }


def _solve_local_linear_map_and_rank(
    Xc: np.ndarray,
    Yc: np.ndarray,
    X_center: np.ndarray,
    Y_center: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    solver: str = "lstsq",
    rng: Optional[np.random.RandomState] = None,
    ransac_n_iter: int = 48,
    ransac_sample_frac: float = 0.5,
    ransac_min_inliers: int = 4,
    ransac_threshold_scale: float = 2.5,
    rank_aggregation: str = "rankme",
    hard_rank_threshold: float = 1e-2,
) -> LocalSolveResult:
    """
    Решает локальную affine-линеаризацию через центрирование:
        (Xc - xc) * M ≈ (Yc - yc)
    и возвращает:
      - rank_value: агрегированный ранг по сингулярным значениям M
          * rank_aggregation="rankme"    -> RankMe (энтропийная, от 1 до D)
          * rank_aggregation="hard_rank" -> количество s_i > hard_rank_threshold
      - сингулярные значения M (для диагностики)
      - raw_residual: legacy-невязка ||(Xc-xc) @ M - (Yc-yc)||_F
      - relative_residual: относительная ошибка по локальным отклонениям Y:
            mean(|(Xc-xc) @ M - (Yc-yc)|) / (mean(|Yc-yc|) + eps)
        Смысл: "на сколько процентов линейное отображение искажает локальные отклонения Y от центра".
    """
    X_center = np.asarray(X_center, dtype=np.float64).reshape(1, -1)
    Y_center = np.asarray(Y_center, dtype=np.float64).reshape(1, -1)
    Xc_centered = np.asarray(Xc, dtype=np.float64) - X_center
    Yc_centered = np.asarray(Yc, dtype=np.float64) - Y_center
    inlier_mask = np.ones(Xc.shape[0], dtype=bool)

    if solver == "ransac":
        if rng is None:
            rng = np.random.RandomState(42)
        M, inlier_mask = _fit_local_linear_map_ransac(
            Xc_centered,
            Yc_centered,
            sample_weights=sample_weights,
            rng=rng,
            n_iter=ransac_n_iter,
            sample_frac=ransac_sample_frac,
            min_inliers=ransac_min_inliers,
            threshold_scale=ransac_threshold_scale,
        )
    else:
        M = _fit_local_linear_map(
            Xc_centered, Yc_centered, sample_weights=sample_weights
        )

    s = svd(M, full_matrices=False, compute_uv=False)

    # Legacy residual сохраняем в старом поле для обратной совместимости артефактов.
    # Дополнительно считаем интерпретируемую относительную ошибку по локальным отклонениям Y.
    X_eval = Xc_centered[inlier_mask]
    Y_eval = Yc_centered[inlier_mask]
    if sample_weights is None:
        err = X_eval @ M - Y_eval
        raw_residual = float(np.linalg.norm(err, "fro"))
        mean_abs_err = float(np.mean(np.abs(err)))
        mean_abs_y = float(np.mean(np.abs(Y_eval)))
    else:
        w_eval = np.asarray(sample_weights, dtype=np.float64).reshape(-1)[inlier_mask]
        w_eval = np.clip(w_eval, 1e-8, None)
        err = X_eval @ M - Y_eval
        sqrt_w = np.sqrt(w_eval)[:, None]
        raw_residual = float(np.linalg.norm(err * sqrt_w, "fro"))
        mean_abs_err = float(np.average(np.mean(np.abs(err), axis=1), weights=w_eval))
        mean_abs_y = float(np.average(np.mean(np.abs(Y_eval), axis=1), weights=w_eval))
    relative_residual = mean_abs_err / (mean_abs_y + 1e-8)

    return LocalSolveResult(
        rank_value=_aggregate_rank(s, rank_aggregation, hard_rank_threshold),
        singular_values=s,
        raw_residual=raw_residual,
        relative_residual=relative_residual,
        inlier_mask=inlier_mask,
    )


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


def _prepare_features_for_local_id(
    X: np.ndarray, spec: Optional["MetricSpec"] = None
) -> np.ndarray:
    # Local ID должен описывать локальную геометрию самого эмбеддинга,
    # а не вспомогательного пространства признаков, используемого конкретной метрикой.
    return np.asarray(_zscore_rows(X), dtype=np.float32)


def _estimate_local_intrinsic_dim_pw(
    X: np.ndarray,
    estimator_name: str,
    n_neighbors: int,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Точная схема по ссылке руководителя:
      skdim.id.<Estimator>().fit_transform_pw(data, n_neighbors=..., n_jobs=...)
    """
    if skdim is None:
        raise RuntimeError(
            "Для --compute_local_id_diagnostics требуется пакет scikit-dimension "
            "(модуль skdim). Установите его в окружение проекта."
        )

    if not hasattr(skdim.id, estimator_name):
        raise ValueError(f"Неизвестный skdim estimator: {estimator_name}")

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] < 3:
        return np.full((X.shape[0],), np.nan, dtype=np.float32)

    n_neighbors_eff = max(2, min(int(n_neighbors), X.shape[0] - 1))
    estimator = getattr(skdim.id, estimator_name)()
    dims = estimator.fit_transform_pw(
        X,
        n_neighbors=n_neighbors_eff,
        n_jobs=n_jobs,
    )
    return np.asarray(dims, dtype=np.float32).reshape(-1)


def _iter_metric_center_indices(
    cache: "NeighborCache",
    spec: "MetricSpec",
) -> Iterable[int]:
    if spec.kind in {"linear_knn", "rff_knn", "multiscale_knn", "local_id_diff"}:
        for center_idx in cache.center_indices:
            yield int(center_idx)
        return

    if spec.kind == "linear_eps":
        percentile_key = (
            spec.sigma_percentile
            if spec.sigma_percentile is not None
            else spec.eps_percentile
        )
        assert percentile_key is not None
        for center_idx, idxs in zip(cache.center_indices, cache.eps[percentile_key]):
            if np.asarray(idxs).size < 2:
                continue
            yield int(center_idx)
        return

    raise ValueError(f"Неизвестный spec.kind для local ID: {spec.kind}")


@dataclass
class LocalIDArtifacts:
    intrinsic_dims_x: List[float]
    intrinsic_dims_y: List[float]
    neighbor_sizes_x: List[int]
    neighbor_sizes_y: List[int]


# ============================================================
# 2) Разбор конфигов (устойчивый к текущей схеме именования)
# ============================================================


@dataclass(frozen=True)
class MetricSpec:
    name: str
    kind: str  # "linear_knn" | "linear_eps" | "multiscale_knn" | "rff_knn" | "local_id_diff"
    pair_agg: str  # "directed" | "antisym" | "sym"

    # параметры окрестности
    k: Optional[int] = None
    eps_percentile: Optional[int] = None
    sigma_percentile: Optional[int] = None
    eps_scale: float = 1.0
    k_list: Optional[Tuple[int, ...]] = None
    aggregator: str = "mean"
    weighting: str = "uniform"  # "uniform" | "gaussian"
    solver: str = "lstsq"  # "lstsq" | "ransac"

    # параметры глобального усреднения
    n_centers: int = 200

    # параметры local ID
    local_id_estimator: str = "MLE"
    local_id_n_neighbors: int = 100

    # параметры RFF
    rff_n_features: int = 256
    rff_gamma: float = 1.0
    rff_seed: int = 42

    # параметры RANSAC
    ransac_n_iter: int = 48
    ransac_sample_frac: float = 0.5
    ransac_min_inliers: int = 4
    ransac_threshold_scale: float = 2.5

    # агрегация ранга:
    #   "rankme"    — энтропийная формула (значение от 1 до D, обратно совместимо)
    #   "hard_rank" — количество сингулярных значений выше порога hard_rank_threshold
    rank_aggregation: str = "rankme"
    hard_rank_threshold: float = 1e-2


def _infer_metric_spec(name: str, cfg: Any, default_n_centers: int = 200) -> MetricSpec:
    """
    Восстанавливает параметры по meta-конфигурации и имени метрики.

    Предпочтительный путь:
      - читаем variant / k / eps_percentile / sigma_percentile / ... из meta.

    Fallback для legacy-данных:
      - если meta неполное, поддерживаем старые длинные и новые короткие имена.
    """
    lower_name = name.lower()
    if "antisym" in lower_name:
        pair_agg = "antisym"
    elif re.search(r"(?:^|_)sym(?:$|_)", lower_name):
        pair_agg = "sym"
    else:
        pair_agg = "directed"

    cfg_dict = cfg if isinstance(cfg, dict) else {}
    meta = cfg_dict.get("meta", cfg_dict) if isinstance(cfg_dict, dict) else {}

    # Пытаемся прочитать n_centers из словаря cfg, если он задан.
    n_centers = default_n_centers
    if isinstance(cfg_dict, dict):
        for key in ["n_centers", "n_samples", "N_SAMPLES", "num_centers"]:
            if key in cfg_dict:
                try:
                    n_centers = int(cfg_dict[key])
                except Exception:
                    pass
    if isinstance(meta, dict):
        for key in ["n_centers", "n_samples", "N_SAMPLES", "num_centers"]:
            if key in meta:
                try:
                    n_centers = int(meta[key])
                except Exception:
                    pass

    # Нестандартные параметры берём из meta, если они заданы.
    eps_scale = 1.0
    weighting = "uniform"
    solver = "lstsq"
    ransac_n_iter = 48
    ransac_sample_frac = 0.5
    ransac_min_inliers = 4
    ransac_threshold_scale = 2.5
    # Параметры агрегации ранга (новые, с обратной совместимостью по умолчанию).
    rank_aggregation = "rankme"
    hard_rank_threshold = 1e-2
    local_id_estimator = "MLE"
    local_id_n_neighbors = 100
    if isinstance(meta, dict):
        try:
            eps_scale = float(meta.get("eps_scale", eps_scale))
        except Exception:
            pass
        weighting = str(meta.get("weighting", weighting))
        solver = str(meta.get("solver", solver))
        try:
            ransac_n_iter = int(meta.get("ransac_n_iter", ransac_n_iter))
        except Exception:
            pass
        try:
            ransac_sample_frac = float(
                meta.get("ransac_sample_frac", ransac_sample_frac)
            )
        except Exception:
            pass
        try:
            ransac_min_inliers = int(
                meta.get("ransac_min_inliers", ransac_min_inliers)
            )
        except Exception:
            pass
        try:
            ransac_threshold_scale = float(
                meta.get("ransac_threshold_scale", ransac_threshold_scale)
            )
        except Exception:
            pass
        rank_aggregation = str(meta.get("rank_aggregation", rank_aggregation))
        try:
            hard_rank_threshold = float(
                meta.get("hard_rank_threshold", hard_rank_threshold)
            )
        except Exception:
            pass
        local_id_estimator = str(
            meta.get("estimator", meta.get("local_id_estimator", local_id_estimator))
        )
        try:
            local_id_n_neighbors = int(
                meta.get(
                    "n_neighbors",
                    meta.get("local_id_n_neighbors", local_id_n_neighbors),
                )
            )
        except Exception:
            pass

    # Новый канонический путь: строим спецификацию из meta, а имя используем как label.
    variant = str(meta.get("variant", "")) if isinstance(meta, dict) else ""
    if variant:
        if variant in {"local_id_diff", "local_id_diff_antisym", "local_id_diff_sym"}:
            return MetricSpec(
                name=name,
                kind="local_id_diff",
                pair_agg=pair_agg,
                n_centers=n_centers,
                local_id_estimator=local_id_estimator,
                local_id_n_neighbors=local_id_n_neighbors,
            )

        if variant in {"linear_knn", "linear_knn_antisym", "linear_knn_sym"}:
            k = int(meta.get("k", 10))
            return MetricSpec(
                name=name,
                kind="linear_knn",
                pair_agg=pair_agg,
                k=k,
                n_centers=n_centers,
                rank_aggregation=rank_aggregation,
                hard_rank_threshold=hard_rank_threshold,
            )

        if variant in {"linear_epsilon_antisym", "linear_epsilon_sym"}:
            return MetricSpec(
                name=name,
                kind="linear_eps",
                pair_agg=pair_agg,
                eps_percentile=int(meta.get("eps_percentile")),
                n_centers=n_centers,
                weighting=weighting,
                solver=solver,
                ransac_n_iter=ransac_n_iter,
                ransac_sample_frac=ransac_sample_frac,
                ransac_min_inliers=ransac_min_inliers,
                ransac_threshold_scale=ransac_threshold_scale,
                rank_aggregation=rank_aggregation,
                hard_rank_threshold=hard_rank_threshold,
            )

        if variant in {"weighted_epsilon_antisym", "weighted_epsilon_sym"}:
            return MetricSpec(
                name=name,
                kind="linear_eps",
                pair_agg=pair_agg,
                sigma_percentile=int(meta.get("sigma_percentile")),
                eps_scale=eps_scale,
                weighting=weighting,
                solver=solver,
                n_centers=n_centers,
                ransac_n_iter=ransac_n_iter,
                ransac_sample_frac=ransac_sample_frac,
                ransac_min_inliers=ransac_min_inliers,
                ransac_threshold_scale=ransac_threshold_scale,
                rank_aggregation=rank_aggregation,
                hard_rank_threshold=hard_rank_threshold,
            )

        if variant in {"weighted_epsilon_ransac_antisym", "weighted_epsilon_ransac_sym"}:
            return MetricSpec(
                name=name,
                kind="linear_eps",
                pair_agg=pair_agg,
                sigma_percentile=int(meta.get("sigma_percentile")),
                eps_scale=eps_scale,
                weighting=weighting,
                solver="ransac",
                n_centers=n_centers,
                ransac_n_iter=ransac_n_iter,
                ransac_sample_frac=ransac_sample_frac,
                ransac_min_inliers=ransac_min_inliers,
                ransac_threshold_scale=ransac_threshold_scale,
                rank_aggregation=rank_aggregation,
                hard_rank_threshold=hard_rank_threshold,
            )

        if variant in {"multiscale_knn", "multiscale_knn_sym"}:
            k_list = tuple(int(x) for x in meta.get("k_list", (5, 10, 20, 40)))
            agg = str(meta.get("aggregator", "mean"))
            return MetricSpec(
                name=name,
                kind="multiscale_knn",
                pair_agg=pair_agg,
                k_list=k_list,
                aggregator=agg,
                n_centers=n_centers,
                rank_aggregation=rank_aggregation,
                hard_rank_threshold=hard_rank_threshold,
            )

        if variant in {"rff_knn", "rff_knn_sym"}:
            k = int(meta.get("k", 10))
            rff_n_features = int(meta.get("n_features", 256))
            rff_gamma = float(meta.get("gamma", 1.0))
            rff_seed = int(meta.get("rff_seed", 42))
            return MetricSpec(
                name=name,
                kind="rff_knn",
                pair_agg=pair_agg,
                k=k,
                n_centers=n_centers,
                rff_n_features=rff_n_features,
                rff_gamma=rff_gamma,
                rff_seed=rff_seed,
                rank_aggregation=rank_aggregation,
                hard_rank_threshold=hard_rank_threshold,
            )

    lower = name.lower()

    # Короткие канонические имена.
    m = re.fullmatch(r"directed_k(\d+)", lower)
    if m:
        return MetricSpec(
            name=name,
            kind="linear_knn",
            pair_agg="directed",
            k=int(m.group(1)),
            n_centers=n_centers,
            rank_aggregation=rank_aggregation,
            hard_rank_threshold=hard_rank_threshold,
        )

    m = re.fullmatch(r"lin_k(\d+)(_antisym|_sym)?", lower)
    if m:
        pair_agg_short = "sym" if m.group(2) == "_sym" else "antisym"
        return MetricSpec(
            name=name,
            kind="linear_knn",
            pair_agg=pair_agg_short,
            k=int(m.group(1)),
            n_centers=n_centers,
            rank_aggregation=rank_aggregation,
            hard_rank_threshold=hard_rank_threshold,
        )

    m = re.fullmatch(r"lin_eps_(\d+)(_antisym|_sym)?", lower)
    if m:
        pair_agg_short = "sym" if m.group(2) == "_sym" else "antisym"
        return MetricSpec(
            name=name,
            kind="linear_eps",
            pair_agg=pair_agg_short,
            eps_percentile=int(m.group(1)),
            n_centers=n_centers,
            weighting=weighting,
            solver=solver,
            ransac_n_iter=ransac_n_iter,
            ransac_sample_frac=ransac_sample_frac,
            ransac_min_inliers=ransac_min_inliers,
            ransac_threshold_scale=ransac_threshold_scale,
            rank_aggregation=rank_aggregation,
            hard_rank_threshold=hard_rank_threshold,
        )

    m = re.fullmatch(r"w_eps_(\d+)(_rsc)?(_hr)?(_antisym|_sym)?", lower)
    if m:
        pair_agg_short = "sym" if m.group(4) == "_sym" else "antisym"
        solver_short = "ransac" if m.group(2) else solver
        rank_aggregation_short = "hard_rank" if m.group(3) else rank_aggregation
        return MetricSpec(
            name=name,
            kind="linear_eps",
            pair_agg=pair_agg_short,
            sigma_percentile=int(m.group(1)),
            eps_scale=eps_scale if eps_scale != 1.0 else 3.0,
            weighting="gaussian",
            solver=solver_short,
            n_centers=n_centers,
            ransac_n_iter=ransac_n_iter,
            ransac_sample_frac=ransac_sample_frac,
            ransac_min_inliers=ransac_min_inliers,
            ransac_threshold_scale=ransac_threshold_scale,
            rank_aggregation=rank_aggregation_short,
            hard_rank_threshold=hard_rank_threshold,
        )

    if lower in {"multiscale_mean", "multiscale_mean_antisym", "multiscale_mean_sym"}:
        return MetricSpec(
            name=name,
            kind="multiscale_knn",
            pair_agg="sym" if lower.endswith("_sym") else "antisym",
            k_list=(5, 10, 20, 40),
            aggregator="mean",
            n_centers=n_centers,
            rank_aggregation=rank_aggregation,
            hard_rank_threshold=hard_rank_threshold,
        )

    m = re.fullmatch(r"rff_k(\d+)(_antisym|_sym)?", lower)
    if m:
        return MetricSpec(
            name=name,
            kind="rff_knn",
            pair_agg="sym" if m.group(2) == "_sym" else "antisym",
            k=int(m.group(1)),
            n_centers=n_centers,
            rank_aggregation=rank_aggregation,
            hard_rank_threshold=hard_rank_threshold,
        )

    m = re.fullmatch(r"id_diff_([a-z0-9]+)_k(\d+)(_antisym|_sym)?", lower)
    if m:
        if m.group(3) == "_sym":
            pair_agg_short = "sym"
        elif m.group(3) == "_antisym":
            pair_agg_short = "antisym"
        else:
            pair_agg_short = pair_agg
        return MetricSpec(
            name=name,
            kind="local_id_diff",
            pair_agg=pair_agg_short,
            n_centers=n_centers,
            local_id_estimator=m.group(1).upper(),
            local_id_n_neighbors=int(m.group(2)),
        )

    # linear knn
    m = re.search(r"linear_knn_k(\d+)", lower)
    if m:
        k = int(m.group(1))
        return MetricSpec(
            name=name,
            kind="linear_knn",
            pair_agg=pair_agg,
            k=k,
            n_centers=n_centers,
            rank_aggregation=rank_aggregation,
            hard_rank_threshold=hard_rank_threshold,
        )

    # weighted epsilon via sigma percentile, optionally with RANSAC
    m = re.search(r"weighted_eps_sigma_percentile_(\d+)", lower)
    if m:
        p = int(m.group(1))
        if isinstance(meta, dict):
            weighting = str(meta.get("weighting", "gaussian"))
            solver = str(
                meta.get("solver", "ransac" if "ransac" in lower else "lstsq")
            )
            try:
                eps_scale = float(meta.get("eps_scale", 3.0))
            except Exception:
                eps_scale = 3.0
        else:
            weighting = "gaussian"
            solver = "ransac" if "ransac" in lower else "lstsq"
            eps_scale = 3.0
        return MetricSpec(
            name=name,
            kind="linear_eps",
            pair_agg=pair_agg,
            sigma_percentile=p,
            eps_scale=eps_scale,
            weighting=weighting,
            solver=solver,
            n_centers=n_centers,
            ransac_n_iter=ransac_n_iter,
            ransac_sample_frac=ransac_sample_frac,
            ransac_min_inliers=ransac_min_inliers,
            ransac_threshold_scale=ransac_threshold_scale,
            rank_aggregation=rank_aggregation,
            hard_rank_threshold=hard_rank_threshold,
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
            weighting=weighting,
            solver=solver,
            ransac_n_iter=ransac_n_iter,
            ransac_sample_frac=ransac_sample_frac,
            ransac_min_inliers=ransac_min_inliers,
            ransac_threshold_scale=ransac_threshold_scale,
            rank_aggregation=rank_aggregation,
            hard_rank_threshold=hard_rank_threshold,
        )

    # multiscale knn
    m = re.search(r"multiscale_knn_(mean|median|min|max)", lower)
    if m:
        agg = m.group(1)
        # Обычно k_list лежит в meta, но если его нет, используем значение по умолчанию.
        k_list = None
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
            rank_aggregation=rank_aggregation,
            hard_rank_threshold=hard_rank_threshold,
        )

    # rff knn
    m = re.search(r"rff_knn_k(\d+)", lower)
    if m:
        k = int(m.group(1))
        # Параметры RFF берём из meta, либо используем значения по умолчанию.
        rff_n_features = 256
        rff_gamma = 1.0
        rff_seed = 42
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
            rank_aggregation=rank_aggregation,
            hard_rank_threshold=hard_rank_threshold,
        )

    raise ValueError(f"Не удалось восстановить спецификацию метрики по имени: {name}")


# ============================================================
# 3) Кэш окрестностей для заданной модели (центры + соседи)
# ============================================================


@dataclass
class NeighborCache:
    center_indices: np.ndarray  # (C,) индексы центральных точек в исходной выборке
    centers: np.ndarray  # (C, D)
    knn: Dict[int, np.ndarray]  # k -> (C, k) индексы
    knn_distances: Dict[int, np.ndarray]  # k -> (C, k) расстояния
    eps: Dict[
        int, np.ndarray
    ]  # percentile -> список индексов, упакованный в object-массив
    eps_distances: Dict[int, np.ndarray]  # percentile -> object-массив расстояний
    X_norm: np.ndarray  # (N, D) нормализованные данные
    sigma_values: Dict[int, float]  # percentile -> скалярный sigma
    eps_values: Dict[int, float]  # percentile -> скалярный eps


def _zscore_rows(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sigma


def _build_neighbor_cache(
    X: np.ndarray,
    spec: MetricSpec,
    seed: int = 42,
    center_indices: Optional[np.ndarray] = None,
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
    if center_indices is None:
        C = min(spec.n_centers, N)
        centers_idx = rng.choice(N, size=C, replace=False)
    else:
        centers_idx = np.asarray(center_indices, dtype=np.int32).reshape(-1)
        if centers_idx.size == 0:
            raise ValueError("center_indices пусты")
        if np.any(centers_idx < 0) or np.any(centers_idx >= N):
            raise ValueError("center_indices выходят за допустимые границы")
    centers = Xn[centers_idx]

    knn: Dict[int, np.ndarray] = {}
    knn_distances: Dict[int, np.ndarray] = {}
    eps: Dict[int, np.ndarray] = {}
    eps_distances: Dict[int, np.ndarray] = {}
    sigma_values: Dict[int, float] = {}
    eps_values: Dict[int, float] = {}

    # Какие k нам нужны.
    ks: List[int] = []
    if spec.kind in {"linear_knn", "rff_knn"} and spec.k is not None:
        ks.append(spec.k)
    if spec.kind == "multiscale_knn" and spec.k_list is not None:
        ks.extend(list(spec.k_list))

    ks = sorted(set(ks))

    D: Optional[np.ndarray] = None

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
            knn_distances[k] = D[row, nn[:, :k]]

    # eps-окрестность:
    if spec.kind == "linear_eps":
        percentile = (
            spec.sigma_percentile
            if spec.sigma_percentile is not None
            else spec.eps_percentile
        )
    else:
        percentile = None

    if spec.kind == "linear_eps" and percentile is not None:
        # Оцениваем характерный масштаб по подвыборке попарных расстояний.
        sub_n = min(4000, N)
        sub_idx = rng.choice(N, size=sub_n, replace=False)
        d = pdist(Xn[sub_idx], metric="euclidean")
        sigma_val = float(np.percentile(d, percentile))
        eps_val = float(sigma_val * spec.eps_scale)
        sigma_values[percentile] = sigma_val
        eps_values[percentile] = eps_val

        if D is None:
            D = cdist(centers, Xn, metric="euclidean")

        for p in [percentile]:
            mask = D <= eps_val
            neigh = []
            neigh_distances = []
            for r in range(mask.shape[0]):
                idx_r = np.where(mask[r])[0].astype(np.int32)
                neigh.append(idx_r)
                neigh_distances.append(D[r, idx_r].astype(np.float32))
            # ВАЖНО: np.array(neigh, dtype=object) при одинаковых размерах
            # элементов создаёт 2D-массив dtype=object вместо 1D object-массива
            # из int-массивов, что ломает индексацию Xn[idxs].
            # Единственный надёжный способ — заполнять поэлементно.
            eps_arr = np.empty(len(neigh), dtype=object)
            for _i, _x in enumerate(neigh):
                eps_arr[_i] = _x
            eps[p] = eps_arr

            eps_dist_arr = np.empty(len(neigh_distances), dtype=object)
            for _i, _x in enumerate(neigh_distances):
                eps_dist_arr[_i] = _x
            eps_distances[p] = eps_dist_arr

    return NeighborCache(
        center_indices=centers_idx.astype(np.int32),
        centers=centers,
        knn=knn,
        knn_distances=knn_distances,
        eps=eps,
        eps_distances=eps_distances,
        X_norm=Xn,
        sigma_values=sigma_values,
        eps_values=eps_values,
    )


# ============================================================
# 4) Направленная метрика m(X->Y) для пары
# ============================================================


# Структура для хранения диагностических данных одного направления (i->j).
@dataclass
class DirectedArtifacts:
    singular_values: List[np.ndarray]  # список (d,) — по одному на центр
    residuals: List[float]  # legacy-невязка по каждому центру
    relative_residuals: List[float]  # относительная ошибка по каждому центру
    ranks: List[int]  # legacy hard-rank M по каждому центру
    metric_ranks: List[float]  # ранг в той же агрегации, что и итоговая метрика
    neighbor_sizes: List[int]  # число точек в окрестности по каждому центру
    neighbor_distances: List[np.ndarray]  # расстояния до точек окрестности
    sigma_values: List[float]  # sigma по каждому центру (если применимо)
    eps_values: List[float]  # eps по каждому центру (если применимо)
    sample_weights: List[np.ndarray]  # веса точек окрестности
    inlier_masks: List[np.ndarray]  # маска инлайеров в robust-solver
    inlier_counts: List[int]  # число инлайеров
    inlier_fracs: List[float]  # доля инлайеров


def _compute_local_intrinsic_dims_for_pair(
    spec: MetricSpec,
    cache_x: NeighborCache,
    dims_x_all: np.ndarray,
    dims_y_all: np.ndarray,
    n_neighbors_eff: int,
) -> LocalIDArtifacts:
    """
    Независимо оценивает локальную размерность в пространствах X и Y
    на согласованных центрах.
    """
    dims_x: List[float] = []
    dims_y: List[float] = []
    sizes_x: List[int] = []
    sizes_y: List[int] = []

    for center_idx in _iter_metric_center_indices(cache_x, spec):
        dims_x.append(float(dims_x_all[int(center_idx)]))
        dims_y.append(float(dims_y_all[int(center_idx)]))
        sizes_x.append(int(n_neighbors_eff))
        sizes_y.append(int(n_neighbors_eff))

    return LocalIDArtifacts(
        intrinsic_dims_x=dims_x,
        intrinsic_dims_y=dims_y,
        neighbor_sizes_x=sizes_x,
        neighbor_sizes_y=sizes_y,
    )


def _metric_directed_for_pair(
    spec: MetricSpec,
    X: np.ndarray,
    Y: np.ndarray,
    cache_X: NeighborCache,
    seed: int = 42,
    dims_x: Optional[np.ndarray] = None,
    dims_y: Optional[np.ndarray] = None,
) -> Tuple[float, DirectedArtifacts]:
    """
    Вычисляет направленную m(X->Y) как среднее по центрам:
      - выбираем окрестность в X вокруг каждого центра (kNN или eps)
      - берём соответствующие строки в Y (те же индексы)
      - решаем локальное линейное отображение и считаем ранг по сингулярным значениям

    Агрегация ранга определяется spec.rank_aggregation:
      "rankme"    — энтропийная формула RankMe
      "hard_rank" — количество сингулярных значений выше spec.hard_rank_threshold

    Дополнительно собирает DirectedArtifacts для каждого центра:
      - сингулярные значения M
      - legacy residual ||(Xc-xc) @ M - (Yc-yc)||_F
      - relative residual по локальным отклонениям Y - yc
      - legacy hard-rank M (по относительному порогу, для совместимости)
      - metric_rank: ранг в той же агрегации, что и итоговая метрика
    """
    rng = np.random.RandomState(seed)
    center_indices = cache_X.center_indices

    artifacts = DirectedArtifacts(
        singular_values=[],
        residuals=[],
        relative_residuals=[],
        ranks=[],
        metric_ranks=[],
        neighbor_sizes=[],
        neighbor_distances=[],
        sigma_values=[],
        eps_values=[],
        sample_weights=[],
        inlier_masks=[],
        inlier_counts=[],
        inlier_fracs=[],
    )

    if spec.kind == "local_id_diff":
        if dims_x is None or dims_y is None:
            raise ValueError("Для local_id_diff нужно передать dims_x и dims_y.")

        vals = []
        n_neighbors_eff = max(2, min(int(spec.local_id_n_neighbors), X.shape[0] - 1))
        for center_idx in center_indices:
            diff = float(dims_x[int(center_idx)] - dims_y[int(center_idx)])
            vals.append(diff)
            artifacts.singular_values.append(np.zeros((0,), dtype=np.float32))
            artifacts.residuals.append(float("nan"))
            artifacts.relative_residuals.append(float("nan"))
            artifacts.ranks.append(0)
            artifacts.metric_ranks.append(diff)
            artifacts.neighbor_sizes.append(int(n_neighbors_eff))
            artifacts.neighbor_distances.append(np.zeros((0,), dtype=np.float32))
            artifacts.sigma_values.append(float("nan"))
            artifacts.eps_values.append(float("nan"))
            artifacts.sample_weights.append(np.ones((0,), dtype=np.float32))
            artifacts.inlier_masks.append(np.zeros((0,), dtype=bool))
            artifacts.inlier_counts.append(0)
            artifacts.inlier_fracs.append(float("nan"))

        if not vals or not np.any(np.isfinite(vals)):
            return float("nan"), artifacts
        return float(np.nanmean(np.asarray(vals, dtype=np.float32))), artifacts

    vals = []
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

    # Порог для жёсткого ранга в артефактах (относительный, от максимального сингулярного значения).
    # Используется только для поля ranks в артефактах — не зависит от rank_aggregation.
    rank_tol_ratio = 1e-10

    def _accumulate(
        Xc: np.ndarray,
        Yc: np.ndarray,
        X_center: np.ndarray,
        Y_center: np.ndarray,
        neighbor_distances: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
        sigma_value: float = float("nan"),
        eps_value: float = float("nan"),
    ) -> Optional[float]:
        """Решает одну локальную задачу и накапливает артефакты."""
        solve_result = _solve_local_linear_map_and_rank(
            Xc,
            Yc,
            X_center=X_center,
            Y_center=Y_center,
            sample_weights=sample_weights,
            solver=spec.solver,
            rng=rng,
            ransac_n_iter=spec.ransac_n_iter,
            ransac_sample_frac=spec.ransac_sample_frac,
            ransac_min_inliers=spec.ransac_min_inliers,
            ransac_threshold_scale=spec.ransac_threshold_scale,
            rank_aggregation=spec.rank_aggregation,
            hard_rank_threshold=spec.hard_rank_threshold,
        )
        rank_val = solve_result.rank_value
        s = solve_result.singular_values
        raw_residual = solve_result.raw_residual
        relative_residual = solve_result.relative_residual
        inlier_mask = solve_result.inlier_mask

        # Жёсткий ранг в артефактах считается по относительному порогу —
        # независимо от rank_aggregation, чтобы диагностика была сопоставимой.
        tol = rank_tol_ratio * float(np.max(s)) if len(s) > 0 else 0.0
        hard_rank_val = int(np.sum(s > tol))

        artifacts.singular_values.append(s)
        artifacts.residuals.append(raw_residual)
        artifacts.relative_residuals.append(relative_residual)
        artifacts.ranks.append(hard_rank_val)
        artifacts.metric_ranks.append(float(rank_val))
        artifacts.neighbor_sizes.append(int(Xc.shape[0]))
        if neighbor_distances is None:
            artifacts.neighbor_distances.append(
                np.zeros((Xc.shape[0],), dtype=np.float32)
            )
        else:
            artifacts.neighbor_distances.append(
                np.asarray(neighbor_distances, dtype=np.float32)
            )
        artifacts.sigma_values.append(float(sigma_value))
        artifacts.eps_values.append(float(eps_value))
        if sample_weights is None:
            artifacts.sample_weights.append(np.ones((Xc.shape[0],), dtype=np.float32))
        else:
            artifacts.sample_weights.append(np.asarray(sample_weights, dtype=np.float32))
        artifacts.inlier_masks.append(np.asarray(inlier_mask, dtype=bool))
        inlier_count = int(np.sum(inlier_mask))
        artifacts.inlier_counts.append(inlier_count)
        artifacts.inlier_fracs.append(
            float(inlier_count / len(inlier_mask)) if len(inlier_mask) > 0 else float("nan")
        )
        return rank_val

    if spec.kind in {"linear_knn", "rff_knn"}:
        assert spec.k is not None
        nn = cache_X.knn[spec.k]
        nn_dist = cache_X.knn_distances[spec.k]
        for center_idx, idxs, dists in zip(center_indices, nn, nn_dist):
            Xc = Xn[idxs]
            Yc = Yn[idxs]
            vals.append(
                _accumulate(
                    Xc,
                    Yc,
                    X_center=Xn[int(center_idx)],
                    Y_center=Yn[int(center_idx)],
                    neighbor_distances=dists,
                )
            )

    elif spec.kind == "multiscale_knn":
        assert spec.k_list is not None
        per_scale = []
        for k in spec.k_list:
            nn = cache_X.knn[int(k)]
            nn_dist = cache_X.knn_distances[int(k)]
            tmp = []
            for center_idx, idxs, dists in zip(center_indices, nn, nn_dist):
                Xc = Xn[idxs]
                Yc = Yn[idxs]
                tmp.append(
                    _accumulate(
                        Xc,
                        Yc,
                        X_center=Xn[int(center_idx)],
                        Y_center=Yn[int(center_idx)],
                        neighbor_distances=dists,
                    )
                )
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
        percentile_key = (
            spec.sigma_percentile
            if spec.sigma_percentile is not None
            else spec.eps_percentile
        )
        assert percentile_key is not None
        neigh = cache_X.eps[percentile_key]
        neigh_distances = cache_X.eps_distances[percentile_key]
        sigma_val = cache_X.sigma_values.get(percentile_key, 1.0)
        sigma_val = max(float(sigma_val), 1e-8)
        eps_val = cache_X.eps_values.get(percentile_key, float("nan"))

        for center_idx, idxs, dists in zip(center_indices, neigh, neigh_distances):
            if idxs.size < 2:
                continue
            Xc = Xn[idxs]
            Yc = Yn[idxs]
            sample_weights = None
            if spec.weighting == "gaussian":
                d2 = np.square(np.asarray(dists, dtype=np.float64))
                sample_weights = np.exp(-d2 / (sigma_val**2)).astype(np.float32)
            vals.append(
                _accumulate(
                    Xc,
                    Yc,
                    X_center=Xn[int(center_idx)],
                    Y_center=Yn[int(center_idx)],
                    neighbor_distances=dists,
                    sample_weights=sample_weights,
                    sigma_value=sigma_val,
                    eps_value=eps_val,
                )
            )

    else:
        raise ValueError(f"Неизвестный spec.kind: {spec.kind}")

    if not vals:
        return float("nan"), artifacts
    return float(np.mean(vals)), artifacts


# ============================================================
# 5) Артефакты: загрузка и дозапись в единый .npz на метрику
# ============================================================


def _artifacts_path(out_dir: str, metric_name: str) -> str:
    return os.path.join(out_dir, "artifacts", f"{metric_name}_artifacts.npz")


def _local_id_artifacts_path(out_dir: str, metric_name: str) -> str:
    return os.path.join(out_dir, "artifacts", f"{metric_name}_local_id_artifacts.npz")


def _load_artifacts(path: str) -> Dict[str, Any]:
    """
    Загружает существующий файл артефактов в словарь {ключ -> массив}.
    Если файл не существует — возвращает пустой словарь.
    """
    if not os.path.exists(path):
        return {}
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _artifact_key_exists(artifacts: Dict[str, Any], model_i: str, model_j: str) -> bool:
    """Проверяет, есть ли уже артефакты для направления model_i -> model_j."""
    key = f"{model_i}_to_{model_j}/residuals"
    return key in artifacts


def _local_id_artifact_key_exists(
    artifacts: Dict[str, Any],
    model_i: str,
    model_j: str,
    center_indices: Optional[np.ndarray] = None,
) -> bool:
    key = f"{model_i}_to_{model_j}/intrinsic_dims_x"
    if key not in artifacts:
        return False
    if center_indices is None:
        return True
    centers_key = f"{model_i}_to_{model_j}/center_indices"
    if centers_key not in artifacts:
        return False
    saved = np.asarray(artifacts[centers_key], dtype=np.int32).reshape(-1)
    current = np.asarray(center_indices, dtype=np.int32).reshape(-1)
    return saved.shape == current.shape and np.array_equal(saved, current)


def _load_diagnostics_meta_from_artifacts(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    raw = artifacts.get("diagnostics_meta_json", None)
    if raw is None:
        return {}
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return raw if isinstance(raw, dict) else {}


def _artifacts_match_current_geometry(artifacts: Dict[str, Any]) -> bool:
    if not artifacts:
        return True
    meta = _load_diagnostics_meta_from_artifacts(artifacts)
    return meta.get("local_geometry_mode") == _LOCAL_GEOMETRY_MODE


def _build_local_id_meta(estimator_name: str, n_neighbors: int) -> Dict[str, Any]:
    meta = {
        "schema_version": 1,
        "local_id_method": "skdim_fit_transform_pw",
        "local_id_estimator": str(estimator_name),
        "local_id_n_neighbors": int(n_neighbors),
    }
    meta.update(_local_geometry_meta())
    return meta


def _local_id_artifacts_match_current_config(
    artifacts: Dict[str, Any], estimator_name: str, n_neighbors: int
) -> bool:
    if not artifacts:
        return True
    raw = artifacts.get("local_id_meta_json", None)
    if raw is None:
        return False
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        meta = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return False
    if not isinstance(meta, dict):
        return False
    return (
        meta.get("local_geometry_mode") == _LOCAL_GEOMETRY_MODE
        and meta.get("local_id_method") == "skdim_fit_transform_pw"
        and str(meta.get("local_id_estimator", "")) == str(estimator_name)
        and int(meta.get("local_id_n_neighbors", -1)) == int(n_neighbors)
    )


def _build_diagnostics_meta(spec: "MetricSpec") -> Dict[str, Any]:
    """
    Формирует словарь метаданных для диагностики, который сохраняется в файл артефактов.

    Скрипт run_diagnose_local_map.py читает эти данные и использует их для:
      - подписей осей и заголовков графиков;
      - подписей в текстовых отчётах;
      - определения смысла поля residuals.

    Если поменялась формула residual или агрегация ранга — достаточно обновить этот
    словарь здесь; в скрипт диагностики лезть не нужно.
    """
    if spec.kind == "local_id_diff":
        ranks_axis_label = "Локальная размерность X - локальная размерность Y"
        ranks_short_label = "Local ID diff"
        ranks_description = (
            "Разность локальных intrinsic dimensions, оценённых независимо в пространствах X и Y "
            f"методом {spec.local_id_estimator} с n_neighbors={spec.local_id_n_neighbors}."
        )
        residual_axis_label = "Residual (не применяется)"
        residual_short_label = "Residual (N/A)"
        residual_summary_label = "Residual (не применяется)"
        residual_description = (
            "Для метрики local_id_diff локальное линейное отображение не решается, "
            "поэтому residuals сохраняются как NaN только для совместимости с форматом артефактов."
        )
    else:
        rank_agg = spec.rank_aggregation
        if rank_agg == "hard_rank":
            ranks_axis_label = (
                f"Ранг отображения M (кол-во сингулярных значений > {spec.hard_rank_threshold:.0e})"
            )
            ranks_short_label = f"Hard rank (thr={spec.hard_rank_threshold:.0e})"
            ranks_description = (
                f"Количество сингулярных значений матрицы M, превышающих порог "
                f"{spec.hard_rank_threshold:.0e}. Значение напрямую интерпретируется как ранг."
            )
        else:
            ranks_axis_label = "Ранг отображения M (RankMe, эффективное число измерений)"
            ranks_short_label = "RankMe"
            ranks_description = (
                "Энтропийная оценка ранга (RankMe): exp(-sum(p_k * log(p_k))), "
                "p_k = s_k / sum(s). Значение от 1 до D."
            )

        residual_axis_label = (
            r"Относительная ошибка ${\rm mean}|(X_c-x_c) M - (Y_c-y_c)| \,/\, ({\rm mean}|Y_c-y_c|)$"
        )
        residual_short_label = "Rel. residual"
        residual_summary_label = (
            r"Средняя относительная ошибка ${\rm mean}|(X_c-x_c) M - (Y_c-y_c)| \,/\, ({\rm mean}|Y_c-y_c|)$"
        )
        residual_description = (
            "Относительная ошибка по локальным приращениям Y: "
            "mean(|(Xc - xc) @ M - (Yc - yc)|) / mean(|Yc - yc|). "
            "Показывает, на сколько процентов линейное отображение искажает отклонения новых координат от центра."
        )

    meta = {
        "schema_version": 4,
        "rank_aggregation": spec.rank_aggregation,
        "hard_rank_threshold": spec.hard_rank_threshold,
        "preferred_rank_field": "metric_ranks",
        "legacy_rank_field": "ranks",
        "ranks_axis_label": ranks_axis_label,
        "ranks_short_label": ranks_short_label,
        "ranks_description": ranks_description,
        "preferred_residual_field": "relative_residuals",
        "legacy_residual_field": "residuals",
        "residual_axis_label": residual_axis_label,
        "residual_short_label": residual_short_label,
        "residual_summary_label": residual_summary_label,
        "residual_description": residual_description,
    }
    meta.update(_local_geometry_meta())
    return meta


def _metric_from_saved_singular_values(
    singular_values: np.ndarray,
    rank_aggregation: str,
    hard_rank_threshold: float,
) -> Tuple[float, np.ndarray]:
    """
    Переагрегирует уже сохранённые сингулярные значения без пересчёта локальных M.
    Возвращает:
      - среднее значение метрики по центрам
      - per-center значения metric_ranks
    """
    per_center = []
    for sv in singular_values:
        arr = np.asarray(sv, dtype=np.float64).reshape(-1)
        per_center.append(_aggregate_rank(arr, rank_aggregation, hard_rank_threshold))
    if not per_center:
        return float("nan"), np.array([], dtype=np.float32)
    metric_ranks = np.asarray(per_center, dtype=np.float32)
    return float(np.mean(metric_ranks)), metric_ranks


def _maybe_reaggregate_direction_from_artifacts(
    artifacts: Dict[str, Any],
    model_i: str,
    model_j: str,
    spec: "MetricSpec",
) -> Optional[Tuple[float, np.ndarray]]:
    """
    Если для направления уже сохранены singular_values, переагрегирует значение метрики
    по ним. Это позволяет менять hard-rank threshold без нового решения локальных задач.
    """
    if not _artifacts_match_current_geometry(artifacts):
        return None
    prefix = f"{model_i}_to_{model_j}"
    if spec.kind == "local_id_diff":
        key = f"{prefix}/metric_ranks"
        if key not in artifacts:
            return None
        metric_ranks = np.asarray(artifacts[key], dtype=np.float32).reshape(-1)
        if metric_ranks.size == 0 or not np.any(np.isfinite(metric_ranks)):
            return float("nan"), metric_ranks
        return float(np.nanmean(metric_ranks)), metric_ranks
    key = f"{prefix}/singular_values"
    if key not in artifacts:
        return None
    return _metric_from_saved_singular_values(
        artifacts[key],
        rank_aggregation=spec.rank_aggregation,
        hard_rank_threshold=spec.hard_rank_threshold,
    )


def _save_artifacts(
    path: str,
    artifacts: Dict[str, Any],
    model_i: str,
    model_j: str,
    directed: "DirectedArtifacts",
    spec: Optional["MetricSpec"] = None,
) -> None:
    """
    Дописывает артефакты направления model_i -> model_j в общий файл метрики.
    Существующие ключи не трогает — только добавляет новые.

    Если передан spec, при первом вызове записывает в файл поле diagnostics_meta_json —
    словарь с описанием смысла полей residuals и ranks (подписи осей, формулы и т.п.).
    Скрипт run_diagnose_local_map.py читает эти данные и не держит подписи в хардкоде.
    """
    prefix = f"{model_i}_to_{model_j}"

    def _to_object_array(items: List[np.ndarray]) -> np.ndarray:
        arr = np.empty(len(items), dtype=object)
        for idx, item in enumerate(items):
            arr[idx] = item
        return arr

    # Сингулярные значения и соседние расстояния могут иметь разную длину по центрам,
    # поэтому храним их как object-массивы.
    sv_array = _to_object_array(directed.singular_values)
    dist_array = _to_object_array(directed.neighbor_distances)
    weights_array = _to_object_array(directed.sample_weights)
    inlier_masks_array = _to_object_array(directed.inlier_masks)

    artifacts[f"{prefix}/singular_values"] = sv_array
    artifacts[f"{prefix}/residuals"] = np.array(directed.residuals, dtype=np.float32)
    artifacts[f"{prefix}/relative_residuals"] = np.array(
        directed.relative_residuals, dtype=np.float32
    )
    artifacts[f"{prefix}/ranks"] = np.array(directed.ranks, dtype=np.int32)
    artifacts[f"{prefix}/metric_ranks"] = np.array(
        directed.metric_ranks, dtype=np.float32
    )
    artifacts[f"{prefix}/neighbor_sizes"] = np.array(
        directed.neighbor_sizes, dtype=np.int32
    )
    artifacts[f"{prefix}/neighbor_distances"] = dist_array
    artifacts[f"{prefix}/sigma_values"] = np.array(
        directed.sigma_values, dtype=np.float32
    )
    artifacts[f"{prefix}/eps_values"] = np.array(directed.eps_values, dtype=np.float32)
    artifacts[f"{prefix}/sample_weights"] = weights_array
    artifacts[f"{prefix}/inlier_masks"] = inlier_masks_array
    artifacts[f"{prefix}/inlier_counts"] = np.array(
        directed.inlier_counts, dtype=np.int32
    )
    artifacts[f"{prefix}/inlier_fracs"] = np.array(
        directed.inlier_fracs, dtype=np.float32
    )

    # Записываем diagnostics_meta_json один раз при первом вызове (когда ключа ещё нет).
    # Содержит описание полей residuals и ranks: формулы, подписи осей, единицы измерения.
    # run_diagnose_local_map.py читает эти данные — менять подписи достаточно здесь.
    if spec is not None and "diagnostics_meta_json" not in artifacts:
        diag_meta = _build_diagnostics_meta(spec)
        artifacts["diagnostics_meta_json"] = json.dumps(diag_meta, ensure_ascii=False)

    np.savez_compressed(path, **artifacts)


def _save_local_id_artifacts(
    path: str,
    artifacts: Dict[str, Any],
    model_i: str,
    model_j: str,
    local_id: LocalIDArtifacts,
    center_indices: np.ndarray,
    estimator_name: str,
    n_neighbors: int,
) -> None:
    prefix = f"{model_i}_to_{model_j}"
    artifacts[f"{prefix}/intrinsic_dims_x"] = np.asarray(
        local_id.intrinsic_dims_x, dtype=np.float32
    )
    artifacts[f"{prefix}/intrinsic_dims_y"] = np.asarray(
        local_id.intrinsic_dims_y, dtype=np.float32
    )
    artifacts[f"{prefix}/neighbor_sizes_x"] = np.asarray(
        local_id.neighbor_sizes_x, dtype=np.int32
    )
    artifacts[f"{prefix}/neighbor_sizes_y"] = np.asarray(
        local_id.neighbor_sizes_y, dtype=np.int32
    )
    artifacts[f"{prefix}/center_indices"] = np.asarray(center_indices, dtype=np.int32)
    artifacts["local_id_meta_json"] = json.dumps(
        _build_local_id_meta(estimator_name, n_neighbors), ensure_ascii=False
    )
    np.savez_compressed(path, **artifacts)


# ============================================================
# 6) Вспомогательные функции для инкрементального ввода-вывода (.npz)
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
    old_spec = meta_old.get("metric_spec", None)
    if old_spec is not None:
        old_spec_cmp = dict(old_spec) if isinstance(old_spec, dict) else old_spec
        new_spec_cmp = asdict(new_spec)
        if isinstance(old_spec_cmp, dict) and old_spec_cmp.get("kind") != "local_id_diff":
            old_spec_cmp.pop("local_id_estimator", None)
            old_spec_cmp.pop("local_id_n_neighbors", None)
            new_spec_cmp.pop("local_id_estimator", None)
            new_spec_cmp.pop("local_id_n_neighbors", None)
        if old_spec_cmp != new_spec_cmp:
            raise RuntimeError(
                "Инкрементальный режим: у существующего файла метрики другой metric_spec.\n"
                f"Существующий: {old_spec}\n"
                f"Новый:        {asdict(new_spec)}\n"
                "Расширение отменено, чтобы не смешивать несовместимые матрицы."
            )

    old_geometry_mode = meta_old.get("local_geometry_mode", None)
    if old_geometry_mode != _LOCAL_GEOMETRY_MODE:
        raise RuntimeError(
            "Инкрементальный режим: файл метрики был посчитан в другом режиме локальной геометрии.\n"
            f"Существующий режим: {old_geometry_mode!r}\n"
            f"Новый режим:        {_LOCAL_GEOMETRY_MODE!r}\n"
            "Расширение отменено, чтобы не смешивать матрицы без центрирования и с центрированием."
        )


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
# 7) Основной запуск
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
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help=(
            "Имена моделей через запятую для ограничения расчёта. "
            "Если задано, из embeddings_dir берутся только указанные модели. "
            "Например: --models resnet50,vit_b_16"
        ),
    )
    parser.add_argument(
        "--hard_rank_threshold_override",
        type=float,
        default=float("nan"),
        help=(
            "Если задано и метрика использует hard-rank, переопределяет absolute threshold. "
            "При наличии сохранённых singular_values значение метрики будет переагрегировано "
            "без нового решения локальных задач."
        ),
    )
    parser.add_argument(
        "--compute_local_id_diagnostics",
        action="store_true",
        help=(
            "Если флаг задан, независимо оценивает локальную intrinsic dimension "
            "в пространствах X и Y и сохраняет её в sidecar-файл для диагностики."
        ),
    )
    parser.add_argument(
        "--local_id_n_neighbors",
        "--local_id_min_neighbors",
        dest="local_id_n_neighbors",
        type=int,
        default=100,
        help=(
            "Число соседей для skdim.id.<Estimator>().fit_transform_pw(...) "
            "при оценке local intrinsic dimension."
        ),
    )
    parser.add_argument(
        "--local_id_estimator",
        type=str,
        default="MLE",
        help="Имя локального оценивателя из skdim.id, например MLE, TLE или MOM.",
    )
    args = parser.parse_args()

    if not args.out_dir:
        if not args.experiment_dir:
            raise ValueError("Нужно указать либо --out_dir, либо --experiment_dir.")
        args.out_dir = os.path.join(args.experiment_dir, "metric_matrices")

    os.makedirs(args.out_dir, exist_ok=True)

    model_names_current, model_to_path = _list_models(args.embeddings_dir)

    # Опциональный фильтр по --models: оставляем только запрошенные модели.
    if args.models:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
        missing = [m for m in requested if m not in model_to_path]
        if missing:
            raise ValueError(
                f"--models: модели не найдены в embeddings_dir: {missing}. "
                f"Доступные: {sorted(model_to_path.keys())}"
            )
        model_names_current = [m for m in model_names_current if m in set(requested)]
        model_to_path = {m: model_to_path[m] for m in model_names_current}

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

    # Local ID зависит только от самих эмбеддингов и глобальных CLI-параметров
    # estimator/n_neighbors, а не от конкретной eps/k-конфигурации метрики.
    # Поэтому кэш безопасно держать на уровне всего запуска и не пересчитывать
    # одни и те же per-point оценки для w_eps_1 / w_eps_2 / w_eps_3 заново.
    global_local_id_dims_by_model: Dict[Tuple[str, str, int], np.ndarray] = {}

    def get_local_id_dims_for_model_global(
        model_name: str,
        estimator_name: str,
        n_neighbors: int,
    ) -> np.ndarray:
        key = (model_name, str(estimator_name), int(n_neighbors))
        if key not in global_local_id_dims_by_model:
            X_local_id = _prepare_features_for_local_id(embeddings[model_name], spec=None)
            global_local_id_dims_by_model[key] = _estimate_local_intrinsic_dim_pw(
                X_local_id,
                estimator_name=estimator_name,
                n_neighbors=n_neighbors,
                n_jobs=1,
            )
        return global_local_id_dims_by_model[key]

    # Вычисление отдельно для каждой метрики.
    for name in chosen:
        cfg = cfgs[name]
        spec = _infer_metric_spec(
            name,
            cfg.get("meta", {}) if isinstance(cfg, dict) else cfg,
            default_n_centers=200,
        )
        if (
            spec.rank_aggregation == "hard_rank"
            and np.isfinite(args.hard_rank_threshold_override)
            and args.hard_rank_threshold_override > 0.0
        ):
            spec = replace(spec, hard_rank_threshold=float(args.hard_rank_threshold_override))

        out_path = os.path.join(args.out_dir, f"{name}.npz")
        artifacts_path = _artifacts_path(args.out_dir, name)
        local_id_artifacts_path = _local_id_artifacts_path(args.out_dir, name)
        os.makedirs(os.path.dirname(artifacts_path), exist_ok=True)

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

        # Загружаем существующие артефакты (или пустой словарь если файла нет).
        # Артефакты всегда синхронны с матрицей: если пара посчитана — артефакты есть.
        saved_artifacts = _load_artifacts(artifacts_path)
        if saved_artifacts and not _artifacts_match_current_geometry(saved_artifacts):
            print(
                "[WARN] Найдены артефакты, посчитанные без текущего локального центрирования. "
                "Они будут проигнорированы и перезаписаны."
            )
            saved_artifacts = {}

        saved_local_id_artifacts = (
            _load_artifacts(local_id_artifacts_path)
            if args.compute_local_id_diagnostics
            else {}
        )
        if (
            args.compute_local_id_diagnostics
            and saved_local_id_artifacts
            and not _local_id_artifacts_match_current_config(
                saved_local_id_artifacts,
                args.local_id_estimator,
                args.local_id_n_neighbors,
            )
        ):
            print(
                "[WARN] Найдены local-ID артефакты с несовместимой конфигурацией. "
                "Они будут проигнорированы и перезаписаны."
            )
            saved_local_id_artifacts = {}

        # ============================================================
        # Вычисление значений
        # - directed: out_matrix — направленная M (может быть несимметричной)
        # - antisym: out_matrix — антисимметричная A
        # - sym: out_matrix — симметричная sim
        # ============================================================

        def _directed_metric(model_i: str, model_j: str) -> Tuple[float, Optional[DirectedArtifacts]]:
            reused = _maybe_reaggregate_direction_from_artifacts(
                saved_artifacts, model_i, model_j, spec
            )
            if reused is not None:
                value, metric_ranks = reused
                saved_artifacts[f"{model_i}_to_{model_j}/metric_ranks"] = np.array(
                    metric_ranks, dtype=np.float32
                )
                return value, None

            Xi_local = embeddings[model_i]
            Yj_local = embeddings[model_j]
            cache_i_local = get_cache_for_model_i(model_i)
            dims_x_local = None
            dims_y_local = None
            if spec.kind == "local_id_diff":
                dims_x_local = get_local_id_dims_for_model_global(
                    model_i,
                    spec.local_id_estimator,
                    spec.local_id_n_neighbors,
                )
                dims_y_local = get_local_id_dims_for_model_global(
                    model_j,
                    spec.local_id_estimator,
                    spec.local_id_n_neighbors,
                )
            return _metric_directed_for_pair(
                spec,
                Xi_local,
                Yj_local,
                cache_i_local,
                seed=args.seed,
                dims_x=dims_x_local,
                dims_y=dims_y_local,
            )

        def _ensure_local_id_for_direction(model_i: str, model_j: str) -> None:
            if not args.compute_local_id_diagnostics:
                return
            cache_x_local = get_cache_for_model_i(model_i)
            if _local_id_artifact_key_exists(
                saved_local_id_artifacts,
                model_i,
                model_j,
                center_indices=cache_x_local.center_indices,
            ):
                return
            dims_x_all = get_local_id_dims_for_model_global(
                model_i,
                args.local_id_estimator,
                args.local_id_n_neighbors,
            )
            dims_y_all = get_local_id_dims_for_model_global(
                model_j,
                args.local_id_estimator,
                args.local_id_n_neighbors,
            )
            n_neighbors_eff_x = max(
                2, min(int(args.local_id_n_neighbors), embeddings[model_i].shape[0] - 1)
            )
            n_neighbors_eff_y = max(
                2, min(int(args.local_id_n_neighbors), embeddings[model_j].shape[0] - 1)
            )
            n_neighbors_eff = min(n_neighbors_eff_x, n_neighbors_eff_y)
            local_id = _compute_local_intrinsic_dims_for_pair(
                spec,
                cache_x_local,
                dims_x_all,
                dims_y_all,
                n_neighbors_eff=n_neighbors_eff,
            )
            _save_local_id_artifacts(
                local_id_artifacts_path,
                saved_local_id_artifacts,
                model_i,
                model_j,
                local_id,
                center_indices=cache_x_local.center_indices,
                estimator_name=args.local_id_estimator,
                n_neighbors=args.local_id_n_neighbors,
            )

        if spec.pair_agg == "antisym":
            # out_matrix хранит A. Мы никогда не трогаем старый блок. Вычисляем только NaN.
            # Для пары (i, j) нужны оба направленных значения: m(i->j) и m(j->i).
            # Тогда A[i,j] = m(i->j) - m(j->i), A[j,i] = -A[i,j].
            # Диагональ равна 0.
            for i, mi in enumerate(tqdm(model_names, desc="Модель i", unit="model")):
                # Если в строке нигде нет NaN, можно быстро пропустить шаг.
                if not np.isnan(out_matrix[i]).any():
                    continue

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

                    # m(i->j)
                    mij, artifacts_ij = _directed_metric(mi, mj)
                    _ensure_local_id_for_direction(mi, mj)

                    # m(j->i)
                    mji, artifacts_ji = _directed_metric(mj, mi)
                    _ensure_local_id_for_direction(mj, mi)

                    # Для local_id_diff directed-величина уже является signed difference
                    # на фиксированном наборе центров. Поэтому для антисимметричной матрицы
                    # усредняем две направленные оценки, а не вычитаем второй раз.
                    antisym_scale = 0.5 if spec.kind == "local_id_diff" else 1.0
                    aij = np.float32(antisym_scale * (mij - mji))
                    out_matrix[i, j] = aij
                    out_matrix[j, i] = np.float32(-aij)

                    # Сохраняем артефакты для обоих направлений.
                    if artifacts_ij is not None:
                        _save_artifacts(
                            artifacts_path, saved_artifacts, mi, mj, artifacts_ij, spec=spec
                        )
                    if artifacts_ji is not None:
                        _save_artifacts(
                            artifacts_path, saved_artifacts, mj, mi, artifacts_ji, spec=spec
                        )

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

                    # m(i->j)
                    mij, artifacts_ij = _directed_metric(mi, mj)
                    _ensure_local_id_for_direction(mi, mj)

                    # m(j->i)
                    mji, artifacts_ji = _directed_metric(mj, mi)
                    _ensure_local_id_for_direction(mj, mi)

                    sij = np.float32(0.5 * (mij + mji))
                    out_matrix[i, j] = sij
                    out_matrix[j, i] = sij

                    # Сохраняем артефакты для обоих направлений.
                    if artifacts_ij is not None:
                        _save_artifacts(
                            artifacts_path, saved_artifacts, mi, mj, artifacts_ij, spec=spec
                        )
                    if artifacts_ji is not None:
                        _save_artifacts(
                            artifacts_path, saved_artifacts, mj, mi, artifacts_ji, spec=spec
                        )

            # Обеспечиваем симметрию и нулевую диагональ
            np.fill_diagonal(out_matrix, 0.0)

        else:
            # out_matrix хранит направленные M. Старый блок не трогаем, только заполняем NaN.
            for i, mi in enumerate(tqdm(model_names, desc="Модель i", unit="model")):
                # Если в строке нигде нет NaN, можно быстро пропустить шаг.
                if not np.isnan(out_matrix[i]).any():
                    continue

                for j, mj in enumerate(model_names):
                    if not np.isnan(out_matrix[i, j]):
                        continue
                    val, artifacts_ij = _directed_metric(mi, mj)
                    _ensure_local_id_for_direction(mi, mj)
                    out_matrix[i, j] = np.float32(val)

                    # Сохраняем артефакты для направления i->j.
                    if artifacts_ij is not None:
                        _save_artifacts(
                            artifacts_path, saved_artifacts, mi, mj, artifacts_ij, spec=spec
                        )

        if args.compute_local_id_diagnostics:
            for mi in model_names:
                for mj in model_names:
                    if mi == mj:
                        continue
                    _ensure_local_id_for_direction(mi, mj)

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
            "rank_aggregation": spec.rank_aggregation,
            "hard_rank_threshold": spec.hard_rank_threshold,
            "metric_spec": asdict(spec),
            "metric_config": cfg.get("meta", {}) if isinstance(cfg, dict) else {},
        }
        meta.update(_local_geometry_meta())

        if saved_artifacts:
            saved_artifacts["diagnostics_meta_json"] = json.dumps(
                _build_diagnostics_meta(spec), ensure_ascii=False
            )
            np.savez_compressed(artifacts_path, **saved_artifacts)

        np.savez_compressed(
            out_path,
            matrix=out_matrix.astype(np.float32),
            model_names=np.array(model_names, dtype=object),
            meta_json=json.dumps(meta, ensure_ascii=False),
        )
        print(f"Сохранено: {out_path}")
        print(f"Артефакты: {artifacts_path}")

    print("\nГотово.")


if __name__ == "__main__":
    main()
