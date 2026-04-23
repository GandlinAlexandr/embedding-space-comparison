"""
Единое место, где описаны конфигурации embedding-метрик для протокола на реальных данных.

Важно (текущее устройство проекта):
- Сами вычисления метрик реализованы в scripts/run_compute_embedding_metrics.py.
- Этот файл хранит ТОЛЬКО список конфигураций и метаданные (k, eps-percentile, multiscale, rff-параметры,
  sample_size и т.п.), чтобы:
    1) не плодить "ручные" конфиги в ноутбуках,
    2) воспроизводимо запускать одни и те же эксперименты,
    3) легко добавлять/выключать конфиги через CLI (--include/--exclude).

Имена метрик генерируются автоматически из параметров по шаблону:

    {kind}_{key_param}[_rsc][_hr][_antisym|_sym]

  где:
    kind       — тип окрестности: lin_k | lin_eps | w_eps | multiscale | rff_k
    key_param  — главный числовой параметр: k, eps_percentile, sigma_percentile,
                 aggregator (для multiscale)
    _rsc       — суффикс, если solver="ransac"
    _hr        — суффикс для hard-rank-конфигов
    _antisym   — суффикс для антисимметричных метрик
    _sym       — суффикс для симметричных метрик
    (без суффикса) — только directed

Примеры:
    lin_k10_antisym      — linear kNN, k=10, antisym
    lin_k10_sym          — linear kNN, k=10, sym
    lin_eps_10_antisym   — linear eps, eps_percentile=10, antisym
    w_eps_10_antisym     — weighted eps, sigma_percentile=10, lstsq, antisym
    w_eps_10_rsc_antisym — weighted eps, sigma_percentile=10, RANSAC, antisym
    w_eps_10_hr_antisym  — weighted eps, hard-rank, antisym
    w_eps_10_rsc_sym     — weighted eps, sigma_percentile=10, RANSAC, sym
    multiscale_mean_antisym — multiscale kNN, aggregator=mean, antisym
    rff_k10_antisym      — RFF kNN, k=10, antisym
    directed_k10         — directed (asymmetric), k=10

Чтобы добавить новый конфиг, достаточно вызвать одну из фабричных функций:
    _lin_knn(k, pair_agg)
    _lin_eps(eps_percentile, pair_agg)
    _w_eps(sigma_percentile, pair_agg, solver)
    _multiscale(k_list, aggregator, pair_agg)
    _rff(k, pair_agg)
    _id_diff_knn(k, estimator, pair_agg)
    _id_diff_eps(eps_percentile, estimator, pair_agg)
и добавить её в список METRIC_SPECS ниже.
"""

from __future__ import annotations

import os as _os
from typing import Dict, Any, List, Optional, Tuple


# ============================================================
# Базовые параметры (менять здесь, а не в каждом конфиге)
# ============================================================

_N_CENTERS = 200
_SAMPLE_SIZE = None

# k для kNN-метрик: менять здесь, список определяет абляцию.
_K_DEFAULT = 10
_K_KNN_ABLATION = [5, 10, 20, 40, 60, 80, 100]   # antisym и sym
_K_LIST_DEFAULT = [5, 10, 20, 40]        # для multiscale
_AGG_DEFAULT = "mean"

_SIGMA_PERCENTILES = [1, 2, 3, 5, 10, 20]
_EPS_SCALE = 1.5
_HARD_RANK_THRESHOLD = 1e-2

_RANSAC_N_ITER = 15
_RANSAC_SAMPLE_FRAC = 0.5
_RANSAC_MIN_INLIERS = 4
_RANSAC_THRESHOLD_SCALE = 2.5

_RFF_N_FEATURES = 256
_RFF_GAMMA = 1.0
_RFF_SEED = 42


# ============================================================
# Суффикс pair_agg в имени и variant
# ============================================================

def _agg_suffix(pair_agg: str) -> str:
    """Возвращает суффикс имени для pair_agg."""
    if pair_agg == "sym":
        return "_sym"
    if pair_agg == "antisym":
        return "_antisym"
    return ""  # directed — без суффикса


def _variant_suffix(pair_agg: str) -> str:
    """Возвращает суффикс variant для pair_agg."""
    if pair_agg == "sym":
        return "_sym"
    if pair_agg == "antisym":
        return "_antisym"
    return ""  # directed


def _rank_suffix(rank_aggregation: str) -> str:
    """Возвращает суффикс имени для способа агрегации ранга."""
    return "_hr" if rank_aggregation == "hard_rank" else ""


# ============================================================
# Фабричные функции: каждая возвращает (name, config_dict)
# ============================================================

def _lin_knn(k: int, pair_agg: str = "antisym") -> Tuple[str, Dict[str, Any]]:
    """linear kNN, окрестность из k соседей."""
    if pair_agg == "directed":
        name = f"directed_k{k}"
        variant = "linear_knn"
    else:
        name = f"lin_k{k}{_agg_suffix(pair_agg)}"
        variant = f"linear_knn{_variant_suffix(pair_agg)}"

    return name, {
        "sample_size": _SAMPLE_SIZE,
        "meta": {
            "family": "local_map_rank",
            "variant": variant,
            "k": k,
            "n_centers": _N_CENTERS,
        },
    }


def _lin_eps(eps_percentile: int, pair_agg: str = "antisym") -> Tuple[str, Dict[str, Any]]:
    """linear eps, порог окрестности = percentile попарных расстояний."""
    name = f"lin_eps_{eps_percentile}{_agg_suffix(pair_agg)}"
    variant = f"linear_epsilon{_variant_suffix(pair_agg)}"
    return name, {
        "sample_size": _SAMPLE_SIZE,
        "meta": {
            "family": "local_map_rank",
            "variant": variant,
            "eps_percentile": eps_percentile,
            "n_centers": _N_CENTERS,
        },
    }


def _w_eps(
    sigma_percentile: int,
    pair_agg: str = "antisym",
    solver: str = "lstsq",
    rank_aggregation: str = "rankme",
    hard_rank_threshold: float = _HARD_RANK_THRESHOLD,
) -> Tuple[str, Dict[str, Any]]:
    """
    Weighted eps: gaussian-веса exp(-d²/σ²), σ = percentile попарных расстояний,
    eps = eps_scale * σ. Опционально — RANSAC для робастного решения.
    """
    rsc_suffix = "_rsc" if solver == "ransac" else ""
    rank_suffix = _rank_suffix(rank_aggregation)
    name = f"w_eps_{sigma_percentile}{rsc_suffix}{rank_suffix}{_agg_suffix(pair_agg)}"

    if solver == "ransac":
        variant = f"weighted_epsilon_ransac{_variant_suffix(pair_agg)}"
    else:
        variant = f"weighted_epsilon{_variant_suffix(pair_agg)}"

    meta: Dict[str, Any] = {
        "family": "local_map_rank",
        "variant": variant,
        "sigma_percentile": sigma_percentile,
        "eps_scale": _EPS_SCALE,
        "weighting": "gaussian",
        "solver": solver,
        "n_centers": _N_CENTERS,
    }
    if rank_aggregation != "rankme":
        meta["rank_aggregation"] = rank_aggregation
        meta["hard_rank_threshold"] = hard_rank_threshold
    if solver == "ransac":
        meta.update({
            "ransac_n_iter": _RANSAC_N_ITER,
            "ransac_sample_frac": _RANSAC_SAMPLE_FRAC,
            "ransac_min_inliers": _RANSAC_MIN_INLIERS,
            "ransac_threshold_scale": _RANSAC_THRESHOLD_SCALE,
        })

    return name, {"sample_size": _SAMPLE_SIZE, "meta": meta}


def _multiscale(
    k_list: List[int] = None,
    aggregator: str = _AGG_DEFAULT,
    pair_agg: str = "antisym",
) -> Tuple[str, Dict[str, Any]]:
    """Multiscale kNN: усредняет по нескольким k."""
    if k_list is None:
        k_list = list(_K_LIST_DEFAULT)
    name = f"multiscale_{aggregator}{_agg_suffix(pair_agg)}"
    variant = f"multiscale_knn{_variant_suffix(pair_agg)}"
    return name, {
        "sample_size": _SAMPLE_SIZE,
        "meta": {
            "family": "local_map_rank",
            "variant": variant,
            "k_list": k_list,
            "aggregator": aggregator,
            "n_centers": _N_CENTERS,
        },
    }


def _rff(k: int, pair_agg: str = "antisym") -> Tuple[str, Dict[str, Any]]:
    """RFF (random Fourier features) + kNN."""
    name = f"rff_k{k}{_agg_suffix(pair_agg)}"
    variant = f"rff_knn{_variant_suffix(pair_agg)}"
    return name, {
        "sample_size": _SAMPLE_SIZE,
        "meta": {
            "family": "local_map_rank",
            "variant": variant,
            "k": k,
            "n_centers": _N_CENTERS,
            "n_features": _RFF_N_FEATURES,
            "gamma": _RFF_GAMMA,
            "rff_seed": _RFF_SEED,
        },
    }


def _id_diff_knn(
    k: int,
    estimator: str = "MLE",
    pair_agg: str = "antisym",
) -> Tuple[str, Dict[str, Any]]:
    """Разность local ID на наборах центров в стиле linear kNN."""
    name = f"id_diff_k{k}_{estimator.lower()}{_agg_suffix(pair_agg)}"
    variant = f"local_id_diff_knn{_variant_suffix(pair_agg)}"
    return name, {
        "sample_size": _SAMPLE_SIZE,
        "meta": {
            "family": "local_id_diff",
            "variant": variant,
            "k": k,
            "estimator": estimator,
            "n_centers": _N_CENTERS,
        },
    }


def _id_diff_eps(
    eps_percentile: int,
    estimator: str = "MLE",
    pair_agg: str = "antisym",
) -> Tuple[str, Dict[str, Any]]:
    """Разность local ID на тех же центрах, что проходят linear-eps фильтр."""
    name = f"id_diff_eps{eps_percentile}_{estimator.lower()}{_agg_suffix(pair_agg)}"
    variant = f"local_id_diff_epsilon{_variant_suffix(pair_agg)}"
    return name, {
        "sample_size": _SAMPLE_SIZE,
        "meta": {
            "family": "local_id_diff",
            "variant": variant,
            "eps_percentile": eps_percentile,
            "estimator": estimator,
            "n_centers": _N_CENTERS,
        },
    }


# ============================================================
# Реестр метрик
# ============================================================
# Добавить новую метрику = одна строка здесь.
# Порядок определяет порядок в отчётах и графиках.

def _build_metric_specs() -> List[Tuple[str, Dict[str, Any]]]:
    specs = []

    # --- Directed (для диагностики асимметрии) ---
    specs.append(_lin_knn(k=_K_DEFAULT, pair_agg="directed"))

    # --- local ID diff, antisym ---
    for k in _K_KNN_ABLATION:
        specs.append(
            _id_diff_knn(
                k=k,
                estimator="MLE",
                pair_agg="antisym",
            )
        )
    for q in [5, 10, 20]:
        specs.append(
            _id_diff_eps(
                eps_percentile=q,
                estimator="MLE",
                pair_agg="antisym",
            )
        )

    # --- linear kNN, antisym ---
    for k in _K_KNN_ABLATION:
        specs.append(_lin_knn(k=k, pair_agg="antisym"))

    # --- linear eps, antisym ---
    for q in [5, 10, 20]:
        specs.append(_lin_eps(eps_percentile=q, pair_agg="antisym"))

    # --- weighted eps (lstsq), antisym ---
    for q in _SIGMA_PERCENTILES:
        specs.append(_w_eps(sigma_percentile=q, pair_agg="antisym", solver="lstsq"))

    # --- weighted eps + RANSAC, antisym ---
    for q in _SIGMA_PERCENTILES:
        specs.append(_w_eps(sigma_percentile=q, pair_agg="antisym", solver="ransac"))

    # --- weighted eps, hard-rank, antisym ---
    for q in _SIGMA_PERCENTILES:
        specs.append(
            _w_eps(
                sigma_percentile=q,
                pair_agg="antisym",
                solver="lstsq",
                rank_aggregation="hard_rank",
            )
        )

    # --- weighted eps + RANSAC, hard-rank, antisym ---
    for q in _SIGMA_PERCENTILES:
        specs.append(
            _w_eps(
                sigma_percentile=q,
                pair_agg="antisym",
                solver="ransac",
                rank_aggregation="hard_rank",
            )
        )

    # --- multiscale, antisym ---
    specs.append(_multiscale(pair_agg="antisym"))

    # --- RFF, antisym ---
    specs.append(_rff(k=_K_DEFAULT, pair_agg="antisym"))

    # --- linear kNN, sym ---
    for k in _K_KNN_ABLATION:
        specs.append(_lin_knn(k=k, pair_agg="sym"))

    # --- linear eps, sym ---
    for q in [5, 10, 20]:
        specs.append(_lin_eps(eps_percentile=q, pair_agg="sym"))

    # --- weighted eps (lstsq), sym ---
    for q in _SIGMA_PERCENTILES:
        specs.append(_w_eps(sigma_percentile=q, pair_agg="sym", solver="lstsq"))

    # --- weighted eps + RANSAC, sym ---
    for q in _SIGMA_PERCENTILES:
        specs.append(_w_eps(sigma_percentile=q, pair_agg="sym", solver="ransac"))

    # --- weighted eps, hard-rank, sym ---
    for q in _SIGMA_PERCENTILES:
        specs.append(
            _w_eps(
                sigma_percentile=q,
                pair_agg="sym",
                solver="lstsq",
                rank_aggregation="hard_rank",
            )
        )

    # --- weighted eps + RANSAC, hard-rank, sym ---
    for q in _SIGMA_PERCENTILES:
        specs.append(
            _w_eps(
                sigma_percentile=q,
                pair_agg="sym",
                solver="ransac",
                rank_aggregation="hard_rank",
            )
        )

    # --- multiscale, sym ---
    specs.append(_multiscale(pair_agg="sym"))

    # --- RFF, sym ---
    specs.append(_rff(k=_K_DEFAULT, pair_agg="sym"))

    return specs


def get_embedding_metric_configs() -> Dict[str, Dict[str, Any]]:
    """
    Возвращает словарь конфигов метрик.

    Формат:
    {
        "metric_name": {
            "sample_size": int | None,
            "meta": dict
        },
        ...
    }

    Имена генерируются автоматически фабричными функциями на основе параметров.
    """
    configs: Dict[str, Dict[str, Any]] = {}
    for name, cfg in _build_metric_specs():
        if name in configs:
            raise ValueError(
                f"Дублирующееся имя метрики: '{name}'. "
                "Проверь параметры в _build_metric_specs()."
            )
        configs[name] = cfg
    return configs


# ============================================================
# Legacy-таблица: старые длинные имена -> короткие
# ============================================================
# Нужна только для чтения артефактов, посчитанных до перехода
# на короткие имена. В новых запусках не используется.

LEGACY_TO_SHORT_METRIC_NAMES: Dict[str, str] = {
    # Старые длинные имена -> новые короткие с _antisym/_sym
    "local_map_rank_linear_knn_k5_antisym": "lin_k5_antisym",
    "local_map_rank_linear_knn_k10_antisym": "lin_k10_antisym",
    "local_map_rank_linear_knn_k20_antisym": "lin_k20_antisym",
    "local_map_rank_linear_knn_k40_antisym": "lin_k40_antisym",
    "local_map_rank_linear_knn_k80_antisym": "lin_k80_antisym",
    "local_map_rank_linear_knn_k5_sym": "lin_k5_sym",
    "local_map_rank_linear_knn_k10_sym": "lin_k10_sym",
    "local_map_rank_linear_knn_k20_sym": "lin_k20_sym",
    "local_map_rank_linear_knn_k40_sym": "lin_k40_sym",
    "local_map_rank_linear_knn_k80_sym": "lin_k80_sym",
    "local_map_rank_linear_knn_k10": "directed_k10",
    "local_map_rank_linear_eps_percentile_5_antisym": "lin_eps_5_antisym",
    "local_map_rank_linear_eps_percentile_10_antisym": "lin_eps_10_antisym",
    "local_map_rank_linear_eps_percentile_20_antisym": "lin_eps_20_antisym",
    "local_map_rank_weighted_eps_sigma_percentile_5_antisym": "w_eps_5_antisym",
    "local_map_rank_weighted_eps_sigma_percentile_10_antisym": "w_eps_10_antisym",
    "local_map_rank_weighted_eps_sigma_percentile_20_antisym": "w_eps_20_antisym",
    "local_map_rank_weighted_eps_sigma_percentile_5_ransac_antisym": "w_eps_5_rsc_antisym",
    "local_map_rank_weighted_eps_sigma_percentile_10_ransac_antisym": "w_eps_10_rsc_antisym",
    "local_map_rank_weighted_eps_sigma_percentile_20_ransac_antisym": "w_eps_20_rsc_antisym",
    "local_map_rank_linear_eps_percentile_5_sym": "lin_eps_5_sym",
    "local_map_rank_linear_eps_percentile_10_sym": "lin_eps_10_sym",
    "local_map_rank_linear_eps_percentile_20_sym": "lin_eps_20_sym",
    "local_map_rank_weighted_eps_sigma_percentile_5_sym": "w_eps_5_sym",
    "local_map_rank_weighted_eps_sigma_percentile_10_sym": "w_eps_10_sym",
    "local_map_rank_weighted_eps_sigma_percentile_20_sym": "w_eps_20_sym",
    "local_map_rank_weighted_eps_sigma_percentile_5_ransac_sym": "w_eps_5_rsc_sym",
    "local_map_rank_weighted_eps_sigma_percentile_10_ransac_sym": "w_eps_10_rsc_sym",
    "local_map_rank_weighted_eps_sigma_percentile_20_ransac_sym": "w_eps_20_rsc_sym",
    "local_map_rank_multiscale_knn_mean_antisym": "multiscale_mean_antisym",
    "local_map_rank_multiscale_knn_mean_sym": "multiscale_mean_sym",
    "local_map_rank_rff_knn_k10_antisym": "rff_k10_antisym",
    "local_map_rank_rff_knn_k10_sym": "rff_k10_sym",
    # Прежние короткие имена без суффикса -> новые с _antisym
    # (для совместимости с уже посчитанными артефактами)
    "lin_k5": "lin_k5_antisym",
    "lin_k10": "lin_k10_antisym",
    "lin_k20": "lin_k20_antisym",
    "lin_k40": "lin_k40_antisym",
    "lin_k80": "lin_k80_antisym",
    "lin_eps_5": "lin_eps_5_antisym",
    "lin_eps_10": "lin_eps_10_antisym",
    "lin_eps_20": "lin_eps_20_antisym",
    "w_eps_5": "w_eps_5_antisym",
    "w_eps_10": "w_eps_10_antisym",
    "w_eps_20": "w_eps_20_antisym",
    "w_eps_5_rsc": "w_eps_5_rsc_antisym",
    "w_eps_10_rsc": "w_eps_10_rsc_antisym",
    "w_eps_20_rsc": "w_eps_20_rsc_antisym",
    "multiscale_mean": "multiscale_mean_antisym",
    "rff_k10": "rff_k10_antisym",
}


# ============================================================
# short_metric_name: имя для графиков
# ============================================================

def short_metric_name(name: str) -> str:
    """
    Возвращает каноническое короткое название метрики для использования в графиках.

    Принимает:
      - короткое имя метрики: "lin_k10"
      - legacy-имя метрики:   "local_map_rank_linear_knn_k10_antisym"
      - имя файла:            "lin_k10.npz"
      - путь к файлу:         "/data/metrics/lin_k10.npz"

    Если имя не распознано — возвращает его как есть (короткие имена
    уже являются каноническими и не нуждаются в дополнительном маппинге).
    """
    name = _os.path.basename(str(name))
    if name.lower().endswith(".npz"):
        name = name[:-4]

    # Legacy-имена переводим в короткие.
    if name in LEGACY_TO_SHORT_METRIC_NAMES:
        return LEGACY_TO_SHORT_METRIC_NAMES[name]

    # Короткие имена уже канонические.
    return name
