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

АРТЕФАКТЫ:
- Вместе с матрицей метрики всегда сохраняется файл {metric_name}_artifacts.npz.
- Артефакты содержат сырые данные по каждому центру для каждого направления (i->j):
    singular_values: (n_centers, d) — сингулярные значения матрицы M
    residuals:       (n_centers,)   — норма невязки ||Xc @ M - Yc||_F
    ranks:           (n_centers,)   — жёсткий ранг M по порогу
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
    rankme_value: float
    singular_values: np.ndarray
    residual: float
    inlier_mask: np.ndarray


def _solve_local_linear_map_and_rank(
    Xc: np.ndarray,
    Yc: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    solver: str = "lstsq",
    rng: Optional[np.random.RandomState] = None,
    ransac_n_iter: int = 48,
    ransac_sample_frac: float = 0.5,
    ransac_min_inliers: int = 4,
    ransac_threshold_scale: float = 2.5,
) -> LocalSolveResult:
    """
    Решает Xc * M ≈ Yc и возвращает:
      - RankMe по сингулярным значениям M
      - сингулярные значения M (для диагностики)
      - residual ||Xc @ M - Yc||_F (для диагностики)
    """
    inlier_mask = np.ones(Xc.shape[0], dtype=bool)

    if solver == "ransac":
        if rng is None:
            rng = np.random.RandomState(42)
        M, inlier_mask = _fit_local_linear_map_ransac(
            Xc,
            Yc,
            sample_weights=sample_weights,
            rng=rng,
            n_iter=ransac_n_iter,
            sample_frac=ransac_sample_frac,
            min_inliers=ransac_min_inliers,
            threshold_scale=ransac_threshold_scale,
        )
    else:
        M = _fit_local_linear_map(Xc, Yc, sample_weights=sample_weights)

    s = svd(M, full_matrices=False, compute_uv=False)

    # Невязка: насколько хорошо линейное приближение работает в данной точке.
    X_eval = Xc[inlier_mask]
    Y_eval = Yc[inlier_mask]
    if sample_weights is None:
        residual = float(np.linalg.norm(X_eval @ M - Y_eval, "fro"))
    else:
        w_eval = np.asarray(sample_weights, dtype=np.float64).reshape(-1)[inlier_mask]
        w_eval = np.clip(w_eval, 1e-8, None)
        sqrt_w = np.sqrt(w_eval)[:, None]
        residual = float(np.linalg.norm((X_eval @ M - Y_eval) * sqrt_w, "fro"))

    return LocalSolveResult(
        rankme_value=rankme(s),
        singular_values=s,
        residual=residual,
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
    sigma_percentile: Optional[int] = None
    eps_scale: float = 1.0
    k_list: Optional[Tuple[int, ...]] = None
    aggregator: str = "mean"
    weighting: str = "uniform"  # "uniform" | "gaussian"
    solver: str = "lstsq"  # "lstsq" | "ransac"

    # параметры глобального усреднения
    n_centers: int = 200

    # параметры RFF
    rff_n_features: int = 256
    rff_gamma: float = 1.0
    rff_seed: int = 42

    # параметры RANSAC
    ransac_n_iter: int = 48
    ransac_sample_frac: float = 0.5
    ransac_min_inliers: int = 4
    ransac_threshold_scale: float = 2.5


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

    # Новый канонический путь: строим спецификацию из meta, а имя используем как label.
    variant = str(meta.get("variant", "")) if isinstance(meta, dict) else ""
    if variant:
        if variant in {"linear_knn", "linear_knn_antisym", "linear_knn_sym"}:
            k = int(meta.get("k", 10))
            return MetricSpec(
                name=name,
                kind="linear_knn",
                pair_agg=pair_agg,
                k=k,
                n_centers=n_centers,
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
        )

    m = re.fullmatch(r"lin_k(\d+)(_sym)?", lower)
    if m:
        pair_agg_short = "sym" if m.group(2) else "antisym"
        return MetricSpec(
            name=name,
            kind="linear_knn",
            pair_agg=pair_agg_short,
            k=int(m.group(1)),
            n_centers=n_centers,
        )

    m = re.fullmatch(r"lin_eps_(\d+)(_sym)?", lower)
    if m:
        pair_agg_short = "sym" if m.group(2) else "antisym"
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
        )

    m = re.fullmatch(r"w_eps_(\d+)(_rsc)?(_sym)?", lower)
    if m:
        pair_agg_short = "sym" if m.group(3) else "antisym"
        solver_short = "ransac" if m.group(2) else solver
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
        )

    if lower in {"multiscale_mean", "multiscale_mean_sym"}:
        return MetricSpec(
            name=name,
            kind="multiscale_knn",
            pair_agg="sym" if lower.endswith("_sym") else "antisym",
            k_list=(5, 10, 20, 40),
            aggregator="mean",
            n_centers=n_centers,
        )

    m = re.fullmatch(r"rff_k(\d+)(_sym)?", lower)
    if m:
        return MetricSpec(
            name=name,
            kind="rff_knn",
            pair_agg="sym" if m.group(2) else "antisym",
            k=int(m.group(1)),
            n_centers=n_centers,
        )

    # linear knn
    m = re.search(r"linear_knn_k(\d+)", lower)
    if m:
        k = int(m.group(1))
        return MetricSpec(
            name=name, kind="linear_knn", pair_agg=pair_agg, k=k, n_centers=n_centers
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
        )

    raise ValueError(f"Не удалось восстановить спецификацию метрики по имени: {name}")


# ============================================================
# 3) Кэш окрестностей для заданной модели (центры + соседи)
# ============================================================


@dataclass
class NeighborCache:
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
            eps[p] = np.array(neigh, dtype=object)
            eps_distances[p] = np.array(neigh_distances, dtype=object)

    return NeighborCache(
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
    residuals: List[float]  # невязка по каждому центру
    ranks: List[int]  # жёсткий ранг M по каждому центру
    neighbor_sizes: List[int]  # число точек в окрестности по каждому центру
    neighbor_distances: List[np.ndarray]  # расстояния до точек окрестности
    sigma_values: List[float]  # sigma по каждому центру (если применимо)
    eps_values: List[float]  # eps по каждому центру (если применимо)
    sample_weights: List[np.ndarray]  # веса точек окрестности
    inlier_masks: List[np.ndarray]  # маска инлайеров в robust-solver
    inlier_counts: List[int]  # число инлайеров
    inlier_fracs: List[float]  # доля инлайеров


def _metric_directed_for_pair(
    spec: MetricSpec,
    X: np.ndarray,
    Y: np.ndarray,
    cache_X: NeighborCache,
    seed: int = 42,
) -> Tuple[float, DirectedArtifacts]:
    """
    Вычисляет направленную m(X->Y) как среднее по центрам:
      - выбираем окрестность в X вокруг каждого центра (kNN или eps)
      - берём соответствующие строки в Y (те же индексы)
      - решаем локальное линейное отображение и считаем RankMe по сингулярным значениям

    Дополнительно собирает DiagnosticData для каждого центра:
      - сингулярные значения M
      - residual ||Xc @ M - Yc||_F
      - жёсткий ранг M
    """
    rng = np.random.RandomState(seed)
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
    artifacts = DirectedArtifacts(
        singular_values=[],
        residuals=[],
        ranks=[],
        neighbor_sizes=[],
        neighbor_distances=[],
        sigma_values=[],
        eps_values=[],
        sample_weights=[],
        inlier_masks=[],
        inlier_counts=[],
        inlier_fracs=[],
    )

    # Порог для жёсткого ранга (относительный, от максимального сингулярного значения).
    rank_tol_ratio = 1e-10

    def _accumulate(
        Xc: np.ndarray,
        Yc: np.ndarray,
        neighbor_distances: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
        sigma_value: float = float("nan"),
        eps_value: float = float("nan"),
    ) -> Optional[float]:
        """Решает одну локальную задачу и накапливает артефакты."""
        solve_result = _solve_local_linear_map_and_rank(
            Xc,
            Yc,
            sample_weights=sample_weights,
            solver=spec.solver,
            rng=rng,
            ransac_n_iter=spec.ransac_n_iter,
            ransac_sample_frac=spec.ransac_sample_frac,
            ransac_min_inliers=spec.ransac_min_inliers,
            ransac_threshold_scale=spec.ransac_threshold_scale,
        )
        rankme_val = solve_result.rankme_value
        s = solve_result.singular_values
        residual = solve_result.residual
        inlier_mask = solve_result.inlier_mask
        tol = rank_tol_ratio * float(np.max(s)) if len(s) > 0 else 0.0
        hard_rank = int(np.sum(s > tol))
        artifacts.singular_values.append(s)
        artifacts.residuals.append(residual)
        artifacts.ranks.append(hard_rank)
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
        return rankme_val

    if spec.kind in {"linear_knn", "rff_knn"}:
        assert spec.k is not None
        nn = cache_X.knn[spec.k]
        nn_dist = cache_X.knn_distances[spec.k]
        for idxs, dists in zip(nn, nn_dist):
            Xc = Xn[idxs]
            Yc = Yn[idxs]
            vals.append(_accumulate(Xc, Yc, neighbor_distances=dists))

    elif spec.kind == "multiscale_knn":
        assert spec.k_list is not None
        per_scale = []
        for k in spec.k_list:
            nn = cache_X.knn[int(k)]
            nn_dist = cache_X.knn_distances[int(k)]
            tmp = []
            for idxs, dists in zip(nn, nn_dist):
                Xc = Xn[idxs]
                Yc = Yn[idxs]
                tmp.append(_accumulate(Xc, Yc, neighbor_distances=dists))
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

        for idxs, dists in zip(neigh, neigh_distances):
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
    return os.path.join(out_dir, f"{metric_name}_artifacts.npz")


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


def _save_artifacts(
    path: str,
    artifacts: Dict[str, Any],
    model_i: str,
    model_j: str,
    directed: DirectedArtifacts,
) -> None:
    """
    Дописывает артефакты направления model_i -> model_j в общий файл метрики.
    Существующие ключи не трогает — только добавляет новые.
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
    artifacts[f"{prefix}/ranks"] = np.array(directed.ranks, dtype=np.int32)
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
        artifacts_path = _artifacts_path(args.out_dir, name)

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
                    mij, artifacts_ij = _metric_directed_for_pair(
                        spec, Xi, Yj, cache_i, seed=args.seed
                    )

                    # m(j->i)
                    Xj = embeddings[mj]
                    cache_j = get_cache_for_model_i(mj)
                    mji, artifacts_ji = _metric_directed_for_pair(
                        spec, Xj, Xi, cache_j, seed=args.seed
                    )

                    aij = np.float32(mij - mji)
                    out_matrix[i, j] = aij
                    out_matrix[j, i] = np.float32(-aij)

                    # Сохраняем артефакты для обоих направлений.
                    _save_artifacts(
                        artifacts_path, saved_artifacts, mi, mj, artifacts_ij
                    )
                    _save_artifacts(
                        artifacts_path, saved_artifacts, mj, mi, artifacts_ji
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

                    # m(i->j)
                    mij, artifacts_ij = _metric_directed_for_pair(
                        spec, Xi, Yj, cache_i, seed=args.seed
                    )

                    # m(j->i)
                    Xj = embeddings[mj]
                    cache_j = get_cache_for_model_i(mj)
                    mji, artifacts_ji = _metric_directed_for_pair(
                        spec, Xj, Xi, cache_j, seed=args.seed
                    )

                    sij = np.float32(0.5 * (mij + mji))
                    out_matrix[i, j] = sij
                    out_matrix[j, i] = sij

                    # Сохраняем артефакты для обоих направлений.
                    _save_artifacts(
                        artifacts_path, saved_artifacts, mi, mj, artifacts_ij
                    )
                    _save_artifacts(
                        artifacts_path, saved_artifacts, mj, mi, artifacts_ji
                    )

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
                    val, artifacts_ij = _metric_directed_for_pair(
                        spec, Xi, Yj, cache_i, seed=args.seed
                    )
                    out_matrix[i, j] = np.float32(val)

                    # Сохраняем артефакты для направления i->j.
                    _save_artifacts(
                        artifacts_path, saved_artifacts, mi, mj, artifacts_ij
                    )

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
        print(f"Артефакты: {artifacts_path}")

    print("\nГотово.")


if __name__ == "__main__":
    main()
