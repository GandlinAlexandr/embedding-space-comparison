"""
Единое место, где описаны конфигурации embedding-метрик для протокола на реальных данных.

Важно (текущее устройство проекта):
- Сами вычисления метрик реализованы в scripts/run_compute_embedding_metrics.py.
- Этот файл хранит ТОЛЬКО список конфигураций и метаданные (k, eps-percentile, multiscale, rff-параметры,
  sample_size и т.п.), чтобы:
    1) не плодить "ручные" конфиги в ноутбуках,
    2) воспроизводимо запускать одни и те же эксперименты,
    3) легко добавлять/выключать конфиги через CLI (--include/--exclude).
"""

from __future__ import annotations

from typing import Dict, Any


# Канонические короткие имена метрик.
# Старые длинные имена сохраняем только как legacy-алиасы для чтения старых артефактов
# и совместимости отображения.
LEGACY_TO_SHORT_METRIC_NAMES: Dict[str, str] = {
    "local_map_rank_linear_knn_k5_antisym": "lin_k5",
    "local_map_rank_linear_knn_k10_antisym": "lin_k10",
    "local_map_rank_linear_knn_k20_antisym": "lin_k20",
    "local_map_rank_linear_knn_k40_antisym": "lin_k40",
    "local_map_rank_linear_knn_k80_antisym": "lin_k80",
    "local_map_rank_linear_knn_k5_sym": "lin_k5_sym",
    "local_map_rank_linear_knn_k10_sym": "lin_k10_sym",
    "local_map_rank_linear_knn_k20_sym": "lin_k20_sym",
    "local_map_rank_linear_knn_k40_sym": "lin_k40_sym",
    "local_map_rank_linear_knn_k80_sym": "lin_k80_sym",
    "local_map_rank_linear_knn_k10": "directed_k10",
    "local_map_rank_linear_eps_percentile_5_antisym": "lin_eps_5",
    "local_map_rank_linear_eps_percentile_10_antisym": "lin_eps_10",
    "local_map_rank_linear_eps_percentile_20_antisym": "lin_eps_20",
    "local_map_rank_weighted_eps_sigma_percentile_5_antisym": "w_eps_5",
    "local_map_rank_weighted_eps_sigma_percentile_10_antisym": "w_eps_10",
    "local_map_rank_weighted_eps_sigma_percentile_20_antisym": "w_eps_20",
    "local_map_rank_weighted_eps_sigma_percentile_5_ransac_antisym": "w_eps_5_rsc",
    "local_map_rank_weighted_eps_sigma_percentile_10_ransac_antisym": "w_eps_10_rsc",
    "local_map_rank_weighted_eps_sigma_percentile_20_ransac_antisym": "w_eps_20_rsc",
    "local_map_rank_linear_eps_percentile_5_sym": "lin_eps_5_sym",
    "local_map_rank_linear_eps_percentile_10_sym": "lin_eps_10_sym",
    "local_map_rank_linear_eps_percentile_20_sym": "lin_eps_20_sym",
    "local_map_rank_weighted_eps_sigma_percentile_5_sym": "w_eps_5_sym",
    "local_map_rank_weighted_eps_sigma_percentile_10_sym": "w_eps_10_sym",
    "local_map_rank_weighted_eps_sigma_percentile_20_sym": "w_eps_20_sym",
    "local_map_rank_weighted_eps_sigma_percentile_5_ransac_sym": "w_eps_5_rsc_sym",
    "local_map_rank_weighted_eps_sigma_percentile_10_ransac_sym": "w_eps_10_rsc_sym",
    "local_map_rank_weighted_eps_sigma_percentile_20_ransac_sym": "w_eps_20_rsc_sym",
    "local_map_rank_multiscale_knn_mean_antisym": "multiscale_mean",
    "local_map_rank_multiscale_knn_mean_sym": "multiscale_mean_sym",
    "local_map_rank_rff_knn_k10_antisym": "rff_k10",
    "local_map_rank_rff_knn_k10_sym": "rff_k10_sym",
}


def get_embedding_metric_configs() -> Dict[str, Dict[str, Any]]:
    """Возвращает словарь конфигов метрик.

    Формат:
    {
        "metric_name": {
            "sample_size": int | None,   # общая подвыборка N (как в прошлом году)
            "meta": dict                 # произвольные метаданные
        },
        ...
    }

    Примечание:
    - Канонические имена метрик в проекте теперь короткие
      (пример: lin_k10, lin_eps_10, w_eps_10_rsc).
    - scripts/run_compute_embedding_metrics.py в первую очередь читает параметры из meta
      и поддерживает старые длинные имена как legacy-вариант.
    - meta используется как справочная информация и для передачи "нестандартных" параметров,
      которые нельзя/неудобно кодировать в имени.
    """

    # ---- Базовые параметры, которые можно менять централизованно ----

    # Сколько центров/точек усреднять в глобальной оценке (у нас это n_centers)
    N_CENTERS = 200

    # Для kNN
    K_DEFAULT = 10
    K_EXTRA_LARGE = [80]

    # Для multiscale
    K_LIST_DEFAULT = [5, 10, 20, 40]
    AGG_DEFAULT = "mean"

    # Для weighted-eps
    SIGMA_PERCENTILES = [5, 10, 20]
    EPS_SCALE = 3.0

    # Для RANSAC в локальной линейной задаче
    RANSAC_N_ITER = 48
    RANSAC_SAMPLE_FRAC = 0.5
    RANSAC_MIN_INLIERS = 4
    RANSAC_THRESHOLD_SCALE = 2.5

    # Для RFF
    RFF_N_FEATURES = 256
    RFF_GAMMA = 1.0
    RFF_SEED = 42

    # Подвыборка объектов (строк эмбеддингов) для ускорения.
    # Важно: скрипт применяет ОДИН общий индекс для всех моделей, чтобы пары были сопоставимы.
    SAMPLE_SIZE = 50000

    # -------------------------------------------------------------------

    configs: Dict[str, Dict[str, Any]] = {}

    # ================================================================
    # 1) Направленная метрика: m(X -> Y)
    # ================================================================
    configs["directed_k10"] = {
        "sample_size": SAMPLE_SIZE,
        "meta": {
            "family": "local_map_rank",
            "variant": "linear_knn",
            "k": K_DEFAULT,
            "n_centers": N_CENTERS,
            "notes": "Directed m(X->Y) using local linear map + RankMe(svd(M)).",
        },
    }

    # ================================================================
    # 2) Антисимметричный скор:
    #    s(X, Y) = m(X->Y) - m(Y->X)
    # ================================================================
    configs["lin_k10"] = {
        "sample_size": SAMPLE_SIZE,
        "meta": {
            "family": "local_map_rank",
            "variant": "linear_knn_antisym",
            "k": K_DEFAULT,
            "n_centers": N_CENTERS,
            "notes": "Anti-sym score for pairwise ranking; boundary=0.",
        },
    }

    # ================================================================
    # 3) Абляция по k — антисимметричный скор
    # ================================================================
    short_names_antisym = {5: "lin_k5", 20: "lin_k20", 40: "lin_k40"}
    for k in [5, 20, 40]:
        name = short_names_antisym[k]
        configs[name] = {
            "sample_size": SAMPLE_SIZE,
            "meta": {
                "family": "local_map_rank",
                "variant": "linear_knn_antisym",
                "k": k,
                "n_centers": N_CENTERS,
            },
        }

    # ================================================================
    # 3a) Дополнительные большие k — антисимметричный скор
    # ================================================================
    for k in K_EXTRA_LARGE:
        name = f"lin_k{k}"
        configs[name] = {
            "sample_size": SAMPLE_SIZE,
            "meta": {
                "family": "local_map_rank",
                "variant": "linear_knn_antisym",
                "k": k,
                "n_centers": N_CENTERS,
                "notes": "Large-k ablation for less trivial local rank.",
            },
        }

    # ================================================================
    # 4) Epsilon-окрестность (адаптивный eps через percentile расстояний) — антисимметричный скор
    # ================================================================
    for q in [5, 10, 20]:
        name = f"lin_eps_{q}"
        configs[name] = {
            "sample_size": SAMPLE_SIZE,
            "meta": {
                "family": "local_map_rank",
                "variant": "linear_epsilon_antisym",
                "eps_percentile": q,
                "n_centers": N_CENTERS,
                "notes": "eps computed as percentile(pdist(zscore(X))) on a subsample.",
            },
        }

    # ================================================================
    # 4a) Weighted-eps через sigma-percentile и eps = 3 * sigma — антисимметричный скор
    # ================================================================
    for q in SIGMA_PERCENTILES:
        name = f"w_eps_{q}"
        configs[name] = {
            "sample_size": SAMPLE_SIZE,
            "meta": {
                "family": "local_map_rank",
                "variant": "weighted_epsilon_antisym",
                "sigma_percentile": q,
                "eps_scale": EPS_SCALE,
                "weighting": "gaussian",
                "solver": "lstsq",
                "n_centers": N_CENTERS,
                "notes": "Gaussian weights exp(-d^2/sigma^2), eps = eps_scale * sigma.",
            },
        }

    # ================================================================
    # 4b) Weighted-eps + RANSAC — антисимметричный скор
    # ================================================================
    for q in SIGMA_PERCENTILES:
        name = f"w_eps_{q}_rsc"
        configs[name] = {
            "sample_size": SAMPLE_SIZE,
            "meta": {
                "family": "local_map_rank",
                "variant": "weighted_epsilon_ransac_antisym",
                "sigma_percentile": q,
                "eps_scale": EPS_SCALE,
                "weighting": "gaussian",
                "solver": "ransac",
                "ransac_n_iter": RANSAC_N_ITER,
                "ransac_sample_frac": RANSAC_SAMPLE_FRAC,
                "ransac_min_inliers": RANSAC_MIN_INLIERS,
                "ransac_threshold_scale": RANSAC_THRESHOLD_SCALE,
                "n_centers": N_CENTERS,
                "notes": "Gaussian weights inside eps-ball; robust fit via RANSAC.",
            },
        }

    # ================================================================
    # 5) Multiscale (по k_list) — антисимметричный скор
    # ================================================================
    configs["multiscale_mean"] = {
        "sample_size": SAMPLE_SIZE,
        "meta": {
            "family": "local_map_rank",
            "variant": "multiscale_knn",
            "k_list": K_LIST_DEFAULT,
            "aggregator": AGG_DEFAULT,
            "n_centers": N_CENTERS,
        },
    }

    # ================================================================
    # 6) RFF (нелинейное перепредставление) — антисимметричный скор
    # ================================================================
    configs["rff_k10"] = {
        "sample_size": SAMPLE_SIZE,
        "meta": {
            "family": "local_map_rank",
            "variant": "rff_knn",
            "k": K_DEFAULT,
            "n_centers": N_CENTERS,
            "n_features": RFF_N_FEATURES,
            "gamma": RFF_GAMMA,
            "rff_seed": RFF_SEED,
        },
    }

    # ================================================================
    # 2b) Симметризированная мера:
    #     sim(X, Y) = 0.5 * (m(X->Y) + m(Y->X))
    # ================================================================
    configs["lin_k10_sym"] = {
        "sample_size": SAMPLE_SIZE,
        "meta": {
            "family": "local_map_rank",
            "variant": "linear_knn_sym",
            "k": K_DEFAULT,
            "n_centers": N_CENTERS,
            "notes": "Symmetrized similarity: 0.5*(m(X->Y)+m(Y->X)).",
        },
    }

    # ================================================================
    # 3b) Абляция по k — симметризированная мера
    # ================================================================
    short_names_sym = {5: "lin_k5_sym", 20: "lin_k20_sym", 40: "lin_k40_sym"}
    for k in [5, 20, 40]:
        name = short_names_sym[k]
        configs[name] = {
            "sample_size": SAMPLE_SIZE,
            "meta": {
                "family": "local_map_rank",
                "variant": "linear_knn_sym",
                "k": k,
                "n_centers": N_CENTERS,
            },
        }

    # ================================================================
    # 3c) Дополнительные большие k — симметризированная мера
    # ================================================================
    for k in K_EXTRA_LARGE:
        name = f"lin_k{k}_sym"
        configs[name] = {
            "sample_size": SAMPLE_SIZE,
            "meta": {
                "family": "local_map_rank",
                "variant": "linear_knn_sym",
                "k": k,
                "n_centers": N_CENTERS,
                "notes": "Large-k ablation for less trivial local rank.",
            },
        }

    # ================================================================
    # 4c) Epsilon-окрестность — симметризированная мера
    # ================================================================
    for q in [5, 10, 20]:
        name = f"lin_eps_{q}_sym"
        configs[name] = {
            "sample_size": SAMPLE_SIZE,
            "meta": {
                "family": "local_map_rank",
                "variant": "linear_epsilon_sym",
                "eps_percentile": q,
                "n_centers": N_CENTERS,
            },
        }

    # ================================================================
    # 4d) Weighted-eps через sigma-percentile и eps = 3 * sigma — симметризированная мера
    # ================================================================
    for q in SIGMA_PERCENTILES:
        name = f"w_eps_{q}_sym"
        configs[name] = {
            "sample_size": SAMPLE_SIZE,
            "meta": {
                "family": "local_map_rank",
                "variant": "weighted_epsilon_sym",
                "sigma_percentile": q,
                "eps_scale": EPS_SCALE,
                "weighting": "gaussian",
                "solver": "lstsq",
                "n_centers": N_CENTERS,
            },
        }

    # ================================================================
    # 4e) Weighted-eps + RANSAC — симметризированная мера
    # ================================================================
    for q in SIGMA_PERCENTILES:
        name = f"w_eps_{q}_rsc_sym"
        configs[name] = {
            "sample_size": SAMPLE_SIZE,
            "meta": {
                "family": "local_map_rank",
                "variant": "weighted_epsilon_ransac_sym",
                "sigma_percentile": q,
                "eps_scale": EPS_SCALE,
                "weighting": "gaussian",
                "solver": "ransac",
                "ransac_n_iter": RANSAC_N_ITER,
                "ransac_sample_frac": RANSAC_SAMPLE_FRAC,
                "ransac_min_inliers": RANSAC_MIN_INLIERS,
                "ransac_threshold_scale": RANSAC_THRESHOLD_SCALE,
                "n_centers": N_CENTERS,
            },
        }

    # ================================================================
    # 5b) Multiscale (по k_list) — симметризированная мера
    # ================================================================
    configs["multiscale_mean_sym"] = {
        "sample_size": SAMPLE_SIZE,
        "meta": {
            "family": "local_map_rank",
            "variant": "multiscale_knn_sym",
            "k_list": K_LIST_DEFAULT,
            "aggregator": AGG_DEFAULT,
            "n_centers": N_CENTERS,
        },
    }

    # ================================================================
    # 6b) RFF — симметризированная мера
    # ================================================================
    configs["rff_k10_sym"] = {
        "sample_size": SAMPLE_SIZE,
        "meta": {
            "family": "local_map_rank",
            "variant": "rff_knn_sym",
            "k": K_DEFAULT,
            "n_centers": N_CENTERS,
            "n_features": RFF_N_FEATURES,
            "gamma": RFF_GAMMA,
            "rff_seed": RFF_SEED,
        },
    }

    return configs


# ============================================================
# Имена метрик для графиков
# ============================================================
# Единственное место в проекте, где задаются сокращения.
# Все скрипты визуализации импортируют отсюда.
#
# Функция принимает имя метрики или путь к файлу метрики
# (включая .npz) и возвращает каноническое короткое название.
#
# При добавлении новой метрики — добавить строку сюда.


import os as _os


def short_metric_name(name: str) -> str:
    """
    Возвращает каноническое короткое название метрики для использования в графиках.

    Принимает:
      - короткое имя метрики: "lin_k10"
      - legacy-имя метрики:   "local_map_rank_linear_knn_k10_antisym"
      - имя файла:            "lin_k10.npz"
      - путь к файлу:         "/data/metrics/lin_k10.npz"

    Если имя не распознано — возвращает его как есть.
    """
    # Берём только имя файла без пути и расширения.
    name = _os.path.basename(str(name))
    if name.lower().endswith(".npz"):
        name = name[:-4]

    if name in LEGACY_TO_SHORT_METRIC_NAMES:
        return LEGACY_TO_SHORT_METRIC_NAMES[name]

    return name
