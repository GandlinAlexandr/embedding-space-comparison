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
    - scripts/run_compute_embedding_metrics.py парсит параметры из ИМЕНИ метрики.
      (пример: local_map_rank_linear_knn_k10_antisym)
    - meta используется как справочная информация и для передачи "нестандартных" параметров,
      которые нельзя/неудобно кодировать в имени.
    """

    # ---- Базовые параметры, которые можно менять централизованно ----

    # Сколько центров/точек усреднять в глобальной оценке (у нас это n_centers)
    N_CENTERS = 200

    # Для kNN
    K_DEFAULT = 10

    # Для multiscale
    K_LIST_DEFAULT = [5, 10, 20, 40]
    AGG_DEFAULT = "mean"

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
    configs["local_map_rank_linear_knn_k10"] = {
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
    configs["local_map_rank_linear_knn_k10_antisym"] = {
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
    for k in [5, 20, 40]:
        name = f"local_map_rank_linear_knn_k{k}_antisym"
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
    # 4) Epsilon-окрестность (адаптивный eps через percentile расстояний) — антисимметричный скор
    # ================================================================
    for q in [5, 10, 20]:
        name = f"local_map_rank_linear_eps_percentile_{q}_antisym"
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
    # 5) Multiscale (по k_list) — антисимметричный скор
    # ================================================================
    configs["local_map_rank_multiscale_knn_mean_antisym"] = {
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
    configs["local_map_rank_rff_knn_k10_antisym"] = {
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
    configs["local_map_rank_linear_knn_k10_sym"] = {
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
    for k in [5, 20, 40]:
        name = f"local_map_rank_linear_knn_k{k}_sym"
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
    # 4b) Epsilon-окрестность — симметризированная мера
    # ================================================================
    for q in [5, 10, 20]:
        name = f"local_map_rank_linear_eps_percentile_{q}_sym"
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
    # 5b) Multiscale (по k_list) — симметризированная мера
    # ================================================================
    configs["local_map_rank_multiscale_knn_mean_sym"] = {
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
    configs["local_map_rank_rff_knn_k10_sym"] = {
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
# Короткие названия метрик для графиков
# ============================================================
# Единственное место в проекте, где задаются сокращения.
# Все скрипты визуализации импортируют отсюда.
#
# Функция принимает полное имя метрики или путь к файлу метрики
# (включая .npz) и возвращает короткое читаемое название.
#
# При добавлении новой метрики — добавить строку сюда.


import os as _os


def short_metric_name(name: str) -> str:
    """
    Возвращает короткое читаемое название метрики для использования в графиках.

    Принимает:
      - полное имя метрики:  "local_map_rank_linear_knn_k10_antisym"
      - имя файла:           "local_map_rank_linear_knn_k10_antisym.npz"
      - путь к файлу:        "/data/metrics/local_map_rank_linear_knn_k10_antisym.npz"

    Возвращает короткое название, например "lin_k10".
    Если имя не распознано — возвращает его как есть (без префикса и расширения).
    """
    # Берём только имя файла без пути и расширения.
    name = _os.path.basename(str(name))
    if name.lower().endswith(".npz"):
        name = name[:-4]

    # Словарь: полное имя метрики -> короткое название.
    # Порядок не важен — используется точное совпадение.
    _MAPPING = {
        # ---- Линейный kNN, антисимметричный ----
        "local_map_rank_linear_knn_k5_antisym": "lin_k5",
        "local_map_rank_linear_knn_k10_antisym": "lin_k10",
        "local_map_rank_linear_knn_k20_antisym": "lin_k20",
        "local_map_rank_linear_knn_k40_antisym": "lin_k40",
        # ---- Линейный kNN, симметричный ----
        "local_map_rank_linear_knn_k5_sym": "lin_k5_sym",
        "local_map_rank_linear_knn_k10_sym": "lin_k10_sym",
        "local_map_rank_linear_knn_k20_sym": "lin_k20_sym",
        "local_map_rank_linear_knn_k40_sym": "lin_k40_sym",
        # ---- Линейный kNN, направленный ----
        "local_map_rank_linear_knn_k10": "directed_k10",
        # ---- Epsilon, антисимметричный ----
        "local_map_rank_linear_eps_percentile_5_antisym": "lin_eps_5",
        "local_map_rank_linear_eps_percentile_10_antisym": "lin_eps_10",
        "local_map_rank_linear_eps_percentile_20_antisym": "lin_eps_20",
        # ---- Epsilon, симметричный ----
        "local_map_rank_linear_eps_percentile_5_sym": "lin_eps_5_sym",
        "local_map_rank_linear_eps_percentile_10_sym": "lin_eps_10_sym",
        "local_map_rank_linear_eps_percentile_20_sym": "lin_eps_20_sym",
        # ---- Multiscale, антисимметричный ----
        "local_map_rank_multiscale_knn_mean_antisym": "multiscale_mean",
        # ---- Multiscale, симметричный ----
        "local_map_rank_multiscale_knn_mean_sym": "multiscale_mean_sym",
        # ---- RFF, антисимметричный ----
        "local_map_rank_rff_knn_k10_antisym": "rff_k10",
        # ---- RFF, симметричный ----
        "local_map_rank_rff_knn_k10_sym": "rff_k10_sym",
    }

    if name in _MAPPING:
        return _MAPPING[name]

    # Если точного совпадения нет — убираем общий префикс как запасной вариант.
    return name.replace("local_map_rank_", "")
