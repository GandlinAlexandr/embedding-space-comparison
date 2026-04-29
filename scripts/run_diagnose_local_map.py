"""
run_diagnose_local_map.py

Диагностика метода локальных линейных отображений между эмбеддингами.

Читает артефакты, сохранённые run_compute_embedding_metrics.py
({metric_name}_artifacts.npz), и строит диагностические графики и статистику.

Проверяемые гипотезы (по заданию):
  1. Ошибка решения линейного уравнения (residual):
       насколько хорошо линейное приближение работает в каждой точке.
       Поддерживаются два режима:
         - legacy residual из main, который дополнительно нормируется в diagnose-скрипте;
         - relative_residuals из новых артефактов:
             mean(|Xc @ M - Yc|) / mean(|Yc|)
       то есть показывает, на сколько процентов отображение искажает координаты Y.
  2. Вырожденность отображений:
       для скольких центров оба направления (X->Y и Y->X) одновременно вырождены.
       По гипотезе таких точек должно быть мало.
  3. Стабильность ранга:
       гистограмма рангов по всем центрам, дисперсия ранга.
       Высокая дисперсия — сигнал шумного решения.

Режимы работы:
  - Сводный (--artifacts_dir): сравнение всех метрик между собой на одном графике.
  - Агрегированный (--artifacts_path): статистика и графики по всем парам одной метрики.
  - Детальный (--artifacts_path + --model_a + --model_b): графики для одной конкретной пары.

Запуск:
  # Сводный график по всем метрикам сразу:
  python -m scripts.run_diagnose_local_map \\
      --artifacts_dir metric_matrices/ \\
      --out_dir diagnostics/

  # Агрегированная диагностика по одной метрике:
  python -m scripts.run_diagnose_local_map \\
      --artifacts_path metric_matrices/directed_k10_artifacts.npz \\
      --out_dir diagnostics/

  # Детально по одной паре:
  python -m scripts.run_diagnose_local_map \\
      --artifacts_path metric_matrices/lin_k10_artifacts.npz \\
      --model_a resnet50 --model_b vit_b16 \\
      --out_dir diagnostics/
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ВАЖНО: запускаем как модуль: python -m scripts.run_diagnose_local_map
from configs.metric_configs import short_metric_name, get_embedding_metric_configs

VALID_PLOT_EXTS = ("png", "pdf", "svg")

# ============================================================
# 0) Вспомогательные функции вывода
# ============================================================


def _make_out_dirs(out_dir: str):
    """
    Создаёт две поддиректории внутри out_dir:
      plots/   — все графики
      reports/ — текстовые и JSON-отчёты (.txt, .json)
    Возвращает (plots_dir, reports_dir).
    """
    plots_dir = os.path.join(out_dir, "plots")
    reports_dir = os.path.join(out_dir, "reports")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    return plots_dir, reports_dir


def parse_plots_exts(raw: str) -> List[str]:
    items = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not items:
        raise ValueError("`--plots_ext` должен содержать хотя бы одно расширение.")

    invalid = [ext for ext in items if ext not in VALID_PLOT_EXTS]
    if invalid:
        raise ValueError(
            f"Неподдерживаемые расширения графиков: {invalid}. "
            f"Допустимые: {list(VALID_PLOT_EXTS)}"
        )

    unique: List[str] = []
    seen = set()
    for ext in items:
        if ext not in seen:
            unique.append(ext)
            seen.add(ext)
    return unique


def _save_figure_variants(
    fig: plt.Figure,
    out_path: str,
    plots_exts: Iterable[str],
    dpi: int = 150,
) -> List[str]:
    base, _ = os.path.splitext(out_path)
    saved_paths: List[str] = []
    for ext in plots_exts:
        save_path = f"{base}.{ext}"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        saved_paths.append(save_path)
    return saved_paths


def _load_downstream_scores(path: str, task_name: str = "") -> Dict[str, float]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and all(isinstance(v, (int, float)) for v in obj.values()):
        return {str(k): float(v) for k, v in obj.items()}

    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        scores = obj["scores"]
        if all(isinstance(v, (int, float)) for v in scores.values()):
            return {str(k): float(v) for k, v in scores.items()}

    if isinstance(obj, dict):
        out: Dict[str, float] = {}
        preferred_task = task_name or "main"
        for model_name, value in obj.items():
            if isinstance(value, dict):
                if preferred_task in value and isinstance(
                    value[preferred_task], (int, float)
                ):
                    out[str(model_name)] = float(value[preferred_task])
                elif len(value) == 1:
                    only_val = next(iter(value.values()))
                    if isinstance(only_val, (int, float)):
                        out[str(model_name)] = float(only_val)
        if out:
            return out

    raise ValueError(
        f"Не удалось прочитать downstream scores из {path}. "
        "Ожидался JSON вида {model: score}, {scores: {model: score}} "
        "или {model: {task: score}}."
    )


# ============================================================
# 1) Загрузка артефактов
# ============================================================

# Пороги по умолчанию — можно переопределить через --degenerate_threshold в CLI.
DEGENERATE_SV_THRESHOLD_DEFAULT = 1e-6
DEGENERATE_MAP_THRESHOLD_DEFAULT = 1e-6


def _load_artifacts(path: str) -> Dict[str, np.ndarray]:
    """Загружает файл артефактов в словарь {ключ -> массив}."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл артефактов не найден: {path}")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _local_id_artifacts_path(main_artifacts_path: str) -> str:
    if not main_artifacts_path.endswith("_artifacts.npz"):
        base, _ = os.path.splitext(main_artifacts_path)
        return f"{base}_local_id_artifacts.npz"
    return main_artifacts_path.replace(
        "_artifacts.npz", "_local_id_artifacts.npz"
    )


def _load_optional_local_id_artifacts(
    main_artifacts_path: str,
) -> Dict[str, np.ndarray]:
    sidecar_path = _local_id_artifacts_path(main_artifacts_path)
    if not os.path.exists(sidecar_path):
        return {}
    return _load_artifacts(sidecar_path)


def _parse_direction_key(key: str) -> Optional[Tuple[str, str]]:
    """
    Из ключа вида '{model_i}_to_{model_j}/residuals' извлекает (model_i, model_j).
    Возвращает None, если ключ не соответствует ожидаемому формату.
    """
    if "/residuals" not in key:
        return None
    direction = key.replace("/residuals", "")
    if "_to_" not in direction:
        return None
    # Разбиваем только по первому вхождению '_to_', чтобы не сломать имена моделей с '_to_'.
    idx = direction.index("_to_")
    model_i = direction[:idx]
    model_j = direction[idx + 4:]
    return model_i, model_j


def _list_directions(artifacts: Dict[str, np.ndarray]) -> List[Tuple[str, str]]:
    """Возвращает список всех направлений (model_i, model_j) в файле артефактов."""
    directions = []
    for key in artifacts:
        parsed = _parse_direction_key(key)
        if parsed is not None:
            directions.append(parsed)
    return directions


def _get_direction_data(
    artifacts: Dict[str, np.ndarray], model_i: str, model_j: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Возвращает (singular_values, residuals, ranks) для направления model_i -> model_j.
    singular_values — object-массив, каждый элемент — вектор сингулярных значений одного центра.

    Для новых артефактов предпочитаем:
      - relative_residuals вместо legacy residuals
      - metric_ranks вместо legacy ranks

    Для старых артефактов автоматически откатываемся к полям из main.
    """
    prefix = f"{model_i}_to_{model_j}"
    meta = _load_diagnostics_meta(artifacts)

    residual_field = str(meta.get("preferred_residual_field", ""))
    if not residual_field or f"{prefix}/{residual_field}" not in artifacts:
        residual_field = (
            "relative_residuals"
            if f"{prefix}/relative_residuals" in artifacts
            else "residuals"
        )

    rank_field = str(meta.get("preferred_rank_field", ""))
    if not rank_field or f"{prefix}/{rank_field}" not in artifacts:
        rank_field = "metric_ranks" if f"{prefix}/metric_ranks" in artifacts else "ranks"

    sv = artifacts[f"{prefix}/singular_values"]
    res = artifacts[f"{prefix}/{residual_field}"]
    ranks = artifacts[f"{prefix}/{rank_field}"]
    return sv, res, ranks


def _get_direction_hard_ranks(
    artifacts: Dict[str, np.ndarray], model_i: str, model_j: str
) -> np.ndarray:
    """Возвращает hard-rank отображения M для диагностики размерности системы."""
    prefix = f"{model_i}_to_{model_j}"
    key = f"{prefix}/ranks"
    if key in artifacts:
        return artifacts[key]
    _, _, metric_values = _get_direction_data(artifacts, model_i, model_j)
    return metric_values


def _uses_metric_values_as_rank_reference(artifacts: Dict[str, np.ndarray]) -> bool:
    """
    Для RankMe/hard-rank метрик значение метрики само является rank-like величиной,
    и график определённости должен оставаться N_sys - metric_value.

    Для weak/tail/adaptive-derived значений это уже не ранг локального отображения,
    поэтому там используем legacy hard-rank M как техническую опору системы.
    """
    meta = _load_diagnostics_meta(artifacts)
    rank_aggregation = str(meta.get("rank_aggregation", "rankme"))
    return rank_aggregation in {"", "rankme", "hard_rank"}


def _get_direction_rank_reference(
    artifacts: Dict[str, np.ndarray],
    model_i: str,
    model_j: str,
    metric_values: np.ndarray,
) -> Tuple[np.ndarray, str]:
    if _uses_metric_values_as_rank_reference(artifacts):
        return metric_values, _metric_value_short_label(_load_diagnostics_labels(artifacts))
    return _get_direction_hard_ranks(artifacts, model_i, model_j), _hard_rank_short_label()


def _direction_uses_relative_residuals(
    artifacts: Dict[str, np.ndarray], model_i: str, model_j: str
) -> bool:
    prefix = f"{model_i}_to_{model_j}"
    meta = _load_diagnostics_meta(artifacts)
    preferred = str(meta.get("preferred_residual_field", ""))
    if preferred and f"{prefix}/{preferred}" in artifacts:
        return preferred == "relative_residuals"
    return f"{prefix}/relative_residuals" in artifacts


def _direction_uses_metric_ranks(
    artifacts: Dict[str, np.ndarray], model_i: str, model_j: str
) -> bool:
    prefix = f"{model_i}_to_{model_j}"
    meta = _load_diagnostics_meta(artifacts)
    preferred = str(meta.get("preferred_rank_field", ""))
    if preferred and f"{prefix}/{preferred}" in artifacts:
        return preferred == "metric_ranks"
    return f"{prefix}/metric_ranks" in artifacts


@dataclass
class DirectionExtraData:
    neighbor_sizes: Optional[np.ndarray] = None
    neighbor_distances: Optional[np.ndarray] = None
    sigma_values: Optional[np.ndarray] = None
    eps_values: Optional[np.ndarray] = None
    sample_weights: Optional[np.ndarray] = None
    inlier_masks: Optional[np.ndarray] = None
    inlier_counts: Optional[np.ndarray] = None
    inlier_fracs: Optional[np.ndarray] = None


@dataclass
class DirectionLocalIDData:
    intrinsic_dims_x: Optional[np.ndarray] = None
    intrinsic_dims_y: Optional[np.ndarray] = None
    neighbor_sizes_x: Optional[np.ndarray] = None
    neighbor_sizes_y: Optional[np.ndarray] = None


def _get_optional_direction_array(
    artifacts: Dict[str, np.ndarray], model_i: str, model_j: str, field: str
) -> Optional[np.ndarray]:
    prefix = f"{model_i}_to_{model_j}"
    key = f"{prefix}/{field}"
    return artifacts.get(key, None)


def _get_direction_extra_data(
    artifacts: Dict[str, np.ndarray], model_i: str, model_j: str
) -> DirectionExtraData:
    return DirectionExtraData(
        neighbor_sizes=_get_optional_direction_array(
            artifacts, model_i, model_j, "neighbor_sizes"
        ),
        neighbor_distances=_get_optional_direction_array(
            artifacts, model_i, model_j, "neighbor_distances"
        ),
        sigma_values=_get_optional_direction_array(
            artifacts, model_i, model_j, "sigma_values"
        ),
        eps_values=_get_optional_direction_array(artifacts, model_i, model_j, "eps_values"),
        sample_weights=_get_optional_direction_array(
            artifacts, model_i, model_j, "sample_weights"
        ),
        inlier_masks=_get_optional_direction_array(
            artifacts, model_i, model_j, "inlier_masks"
        ),
        inlier_counts=_get_optional_direction_array(
            artifacts, model_i, model_j, "inlier_counts"
        ),
        inlier_fracs=_get_optional_direction_array(
            artifacts, model_i, model_j, "inlier_fracs"
        ),
    )


# ============================================================
# 2) Диагностические вычисления
# ============================================================


def _is_degenerate(
    sv: np.ndarray, threshold: float = DEGENERATE_MAP_THRESHOLD_DEFAULT
) -> bool:
    """Отображение считается вырожденным, если все сингулярные значения меньше порога."""
    if sv is None or len(sv) == 0:
        return True
    return bool(np.all(np.abs(sv) < threshold))


def _compute_direction_stats(
    sv_arr: np.ndarray,
    residuals: np.ndarray,
    ranks: np.ndarray,
    threshold: float = DEGENERATE_MAP_THRESHOLD_DEFAULT,
) -> Dict:
    """
    Вычисляет статистику для одного направления (i->j).
    residuals уже приведены к нужному виду вызывающей стороной:
      - либо relative_residuals из новых артефактов,
      - либо legacy residuals, дополнительно нормированные в diagnose-скрипте.
    """
    # Ранги
    ranks = ranks.astype(np.float32)
    rank_mean = float(np.mean(ranks))
    rank_std = float(np.std(ranks))
    rank_min = float(np.min(ranks))
    rank_max = float(np.max(ranks))

    # Residuals (уже относительные)
    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals))
    res_median = float(np.median(residuals))

    # Доля вырожденных отображений в данном направлении
    n_degenerate = sum(_is_degenerate(sv, threshold=threshold) for sv in sv_arr)
    frac_degenerate = n_degenerate / len(sv_arr) if len(sv_arr) > 0 else float("nan")

    return {
        "n_centers": len(ranks),
        "rank_mean": rank_mean,
        "rank_std": rank_std,
        "rank_min": rank_min,
        "rank_max": rank_max,
        "residual_mean": res_mean,
        "residual_std": res_std,
        "residual_median": res_median,
        "n_degenerate": n_degenerate,
        "frac_degenerate": frac_degenerate,
    }


# ============================================================
# Подписи для графиков и отчётов
# ============================================================
# Подписи для новых артефактов читаются из diagnostics_meta_json.
# Для старых артефактов сохраняем legacy-логику нормировки и подписей.

_FALLBACK_LABELS = {
    "ranks_axis_label": "Ранг отображения M",
    "ranks_short_label": "Ранг",
    "ranks_description": "Ранг матрицы отображения M.",
    "residual_axis_label": r"Residual $\|X_c M - Y_c\|_F$",
    "residual_short_label": "Residual",
    "residual_summary_label": r"Средний residual / normalized residual",
    "residual_description": "Legacy residual ||Xc @ M - Yc||_F.",
}


def _metric_value_axis_label(labels: Dict[str, str]) -> str:
    return labels.get("ranks_axis_label", _FALLBACK_LABELS["ranks_axis_label"])


def _metric_value_short_label(labels: Dict[str, str]) -> str:
    return labels.get("ranks_short_label", _FALLBACK_LABELS["ranks_short_label"])


def _hard_rank_axis_label() -> str:
    return "Hard rank отображения M"


def _hard_rank_short_label() -> str:
    return "hard rank"


def _load_diagnostics_meta(artifacts: Dict[str, np.ndarray]) -> Dict[str, object]:
    raw = artifacts.get("diagnostics_meta_json", None)
    if raw is None:
        return {}

    # numpy может сохранить строку как 0-мерный массив.
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    try:
        meta = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return {}

    if not isinstance(meta, dict):
        return {}
    return meta


def _load_diagnostics_labels(artifacts: Dict[str, np.ndarray]) -> Dict[str, str]:
    """
    Читает подписи осей и описания полей из diagnostics_meta_json в файле артефактов.
    Для старых артефактов без этого поля возвращает безопасные fallback-подписи.
    """
    meta = _load_diagnostics_meta(artifacts)
    if not meta:
        return dict(_FALLBACK_LABELS)

    # Подставляем только известные ключи; остальное берём из fallback.
    labels = dict(_FALLBACK_LABELS)
    for key in _FALLBACK_LABELS:
        if key in meta and isinstance(meta[key], str):
            labels[key] = meta[key]
    return labels


def _normalized_residual_label(normalized: bool) -> str:
    if normalized:
        return r"Normalized residual $\|X_c M - Y_c\|_F / \sqrt{N_{\mathrm{eff}}}$"
    return r"Residual $\|X_c M - Y_c\|_F$"


def _normalized_residual_short_label(normalized: bool) -> str:
    return "Normalized residual" if normalized else "Residual"


def _summary_residual_axis_label(all_normalized: bool) -> str:
    if all_normalized:
        return r"Средний normalized residual $\|X_c M - Y_c\|_F / \sqrt{N_{\mathrm{eff}}}$"
    return r"Средний residual / normalized residual"


def _compute_effective_counts(extra: DirectionExtraData) -> Optional[np.ndarray]:
    """
    Возвращает эффективное число точек N_eff для нормировки residual.

    Логика:
      - если есть sample_weights, нормируем на sqrt(sum(weights)) или
        sqrt(sum(weights[inliers])) для robust-solver;
      - иначе, если есть inlier_counts, используем sqrt(inlier_counts);
      - иначе, если есть neighbor_sizes, используем sqrt(neighbor_sizes).
    """
    if extra.sample_weights is not None and len(extra.sample_weights) > 0:
        eff = []
        has_masks = extra.inlier_masks is not None and len(extra.inlier_masks) == len(
            extra.sample_weights
        )
        for idx, weights in enumerate(extra.sample_weights):
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            w = np.clip(w, 0.0, None)
            if has_masks:
                mask = np.asarray(extra.inlier_masks[idx], dtype=bool).reshape(-1)
                if len(mask) == len(w):
                    w = w[mask]
            total_w = float(np.sum(w))
            eff.append(total_w if total_w > 0.0 else float("nan"))
        return np.asarray(eff, dtype=np.float64)

    if extra.inlier_counts is not None:
        return np.asarray(extra.inlier_counts, dtype=np.float64)

    if extra.neighbor_sizes is not None:
        return np.asarray(extra.neighbor_sizes, dtype=np.float64)

    return None


def _compute_system_point_counts(extra: DirectionExtraData) -> Optional[np.ndarray]:
    """
    Возвращает число точек, определяющих локальную систему.

    Для диагностики определённости системы считаем именно количество точек:
      - для robust-методов используем число инлайеров;
      - иначе размер окрестности;
      - в крайнем случае длину вектора весов.
    """
    if extra.inlier_counts is not None:
        return np.asarray(extra.inlier_counts, dtype=np.float64)

    if extra.neighbor_sizes is not None:
        return np.asarray(extra.neighbor_sizes, dtype=np.float64)

    if extra.sample_weights is not None and len(extra.sample_weights) > 0:
        counts = []
        for weights in extra.sample_weights:
            w = np.asarray(weights).reshape(-1)
            counts.append(float(len(w)))
        return np.asarray(counts, dtype=np.float64)

    return None


def _normalize_residuals(
    residuals: np.ndarray,
    extra: DirectionExtraData,
) -> Tuple[np.ndarray, bool]:
    """
    Нормирует legacy residual до RMS-подобной ошибки на эффективную точку.

    Для новых артефактов нужно использовать relative_residuals вместо этого шага.
    """
    res = np.asarray(residuals, dtype=np.float64)
    eff_counts = _compute_effective_counts(extra)
    if eff_counts is None or len(eff_counts) != len(res):
        return res, False

    denom = np.sqrt(np.clip(eff_counts, 0.0, None))
    out = np.full_like(res, np.nan, dtype=np.float64)
    good = np.isfinite(denom) & (denom > 0.0)
    out[good] = res[good] / denom[good]

    if not np.any(good):
        return res, False
    return out, True


def _safe_mean(arr: Optional[np.ndarray]) -> float:
    if arr is None:
        return float("nan")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _safe_std(arr: Optional[np.ndarray]) -> float:
    if arr is None:
        return float("nan")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.std(finite))


def _compute_extra_stats(extra: DirectionExtraData) -> Dict:
    """
    Сводит дополнительные артефакты в компактные числа для отчётов и таблиц.
    """
    neighbor_distance_mean = float("nan")
    neighbor_distance_std = float("nan")
    if extra.neighbor_distances is not None and len(extra.neighbor_distances) > 0:
        per_center_means = []
        for d in extra.neighbor_distances:
            arr = np.asarray(d, dtype=np.float64).reshape(-1)
            finite = arr[np.isfinite(arr)]
            if finite.size > 0:
                per_center_means.append(float(np.mean(finite)))
        if per_center_means:
            neighbor_distance_mean = float(np.mean(per_center_means))
            neighbor_distance_std = float(np.std(per_center_means))

    return {
        "neighbor_size_mean": _safe_mean(extra.neighbor_sizes),
        "neighbor_size_std": _safe_std(extra.neighbor_sizes),
        "neighbor_distance_mean": neighbor_distance_mean,
        "neighbor_distance_std": neighbor_distance_std,
        "sigma_mean": _safe_mean(extra.sigma_values),
        "eps_mean": _safe_mean(extra.eps_values),
        "inlier_count_mean": _safe_mean(extra.inlier_counts),
        "inlier_frac_mean": _safe_mean(extra.inlier_fracs),
        "inlier_frac_std": _safe_std(extra.inlier_fracs),
    }


def _compute_determinedness_stats(
    ranks: np.ndarray,
    extra: DirectionExtraData,
) -> Dict[str, float]:
    """
    Сводит в компактные числа определённость локальной системы:
      - сколько точек участвует в системе;
      - насколько система переопределена: N_sys - rank;
      - доля центров, где N_sys > rank.
    """
    point_counts = _compute_system_point_counts(extra)
    if point_counts is None:
        return {
            "point_count_mean": float("nan"),
            "point_count_std": float("nan"),
            "margin_mean": float("nan"),
            "margin_std": float("nan"),
            "overdetermined_frac": float("nan"),
        }

    point_counts = np.asarray(point_counts, dtype=np.float64).reshape(-1)
    ranks = np.asarray(ranks, dtype=np.float64).reshape(-1)
    if len(point_counts) != len(ranks) or len(ranks) == 0:
        return {
            "point_count_mean": float("nan"),
            "point_count_std": float("nan"),
            "margin_mean": float("nan"),
            "margin_std": float("nan"),
            "overdetermined_frac": float("nan"),
        }

    finite = np.isfinite(point_counts) & np.isfinite(ranks)
    if not np.any(finite):
        return {
            "point_count_mean": float("nan"),
            "point_count_std": float("nan"),
            "margin_mean": float("nan"),
            "margin_std": float("nan"),
            "overdetermined_frac": float("nan"),
        }

    point_counts = point_counts[finite]
    ranks = ranks[finite]
    margins = point_counts - ranks
    return {
        "point_count_mean": float(np.mean(point_counts)),
        "point_count_std": float(np.std(point_counts)),
        "margin_mean": float(np.mean(margins)),
        "margin_std": float(np.std(margins)),
        "overdetermined_frac": float(np.mean(margins > 0.0)),
    }


def _compute_local_id_stats(
    ranks: np.ndarray,
    extra: DirectionExtraData,
    local_id: DirectionLocalIDData,
) -> Dict[str, float]:
    if local_id.intrinsic_dims_x is None or local_id.intrinsic_dims_y is None:
        return {
            "id_x_mean": float("nan"),
            "id_y_mean": float("nan"),
            "id_min_mean": float("nan"),
            "rank_minus_id_min_mean": float("nan"),
            "system_minus_id_min_mean": float("nan"),
        }

    id_x = np.asarray(local_id.intrinsic_dims_x, dtype=np.float64).reshape(-1)
    id_y = np.asarray(local_id.intrinsic_dims_y, dtype=np.float64).reshape(-1)
    ranks = np.asarray(ranks, dtype=np.float64).reshape(-1)
    if len(id_x) != len(ranks) or len(id_y) != len(ranks) or len(ranks) == 0:
        return {
            "id_x_mean": float("nan"),
            "id_y_mean": float("nan"),
            "id_min_mean": float("nan"),
            "rank_minus_id_min_mean": float("nan"),
            "system_minus_id_min_mean": float("nan"),
        }

    id_min = np.minimum(id_x, id_y)
    system_counts = _compute_system_point_counts(extra)
    if system_counts is not None:
        system_counts = np.asarray(system_counts, dtype=np.float64).reshape(-1)
        if len(system_counts) != len(ranks):
            system_counts = None

    finite = np.isfinite(id_x) & np.isfinite(id_y) & np.isfinite(ranks)
    if not np.any(finite):
        return {
            "id_x_mean": float("nan"),
            "id_y_mean": float("nan"),
            "id_min_mean": float("nan"),
            "rank_minus_id_min_mean": float("nan"),
            "system_minus_id_min_mean": float("nan"),
        }

    id_x = id_x[finite]
    id_y = id_y[finite]
    id_min = id_min[finite]
    ranks = ranks[finite]
    if system_counts is not None:
        system_counts = system_counts[finite]

    return {
        "id_x_mean": float(np.mean(id_x)),
        "id_y_mean": float(np.mean(id_y)),
        "id_min_mean": float(np.mean(id_min)),
        "rank_minus_id_min_mean": float(np.mean(ranks - id_min)),
        "system_minus_id_min_mean": (
            float(np.mean(system_counts - id_min))
            if system_counts is not None
            else float("nan")
        ),
    }


def _compute_both_degenerate_fraction(
    artifacts: Dict[str, np.ndarray],
    directions: List[Tuple[str, str]],
    threshold: float = DEGENERATE_MAP_THRESHOLD_DEFAULT,
) -> Dict:
    """
    Для каждой неупорядоченной пары {i, j} проверяет, вырождены ли оба направления
    (i->j и j->i) одновременно в одном и том же центре.

    По гипотезе: таких центров должно быть мало —
    хотя бы одно из двух отображений должно быть невырожденным.
    """
    # Строим словарь direction -> sv_arr для быстрого доступа.
    sv_map: Dict[Tuple[str, str], np.ndarray] = {}
    for mi, mj in directions:
        sv, _, _ = _get_direction_data(artifacts, mi, mj)
        sv_map[(mi, mj)] = sv

    # Находим все уникальные неупорядоченные пары.
    pairs_seen = set()
    results = []

    for mi, mj in directions:
        pair = tuple(sorted([mi, mj]))
        if pair in pairs_seen:
            continue
        if mi == mj:
            # Диагональные пары (i→i) не имеют смысла — пропускаем.
            continue
        if (mj, mi) not in sv_map:
            # Обратного направления нет — пропускаем.
            continue
        pairs_seen.add(pair)

        sv_ij = sv_map[(mi, mj)]
        sv_ji = sv_map[(mj, mi)]
        n = min(len(sv_ij), len(sv_ji))

        if n == 0:
            continue

        both_deg = sum(
            _is_degenerate(sv_ij[c], threshold=threshold)
            and _is_degenerate(sv_ji[c], threshold=threshold)
            for c in range(n)
        )
        results.append(
            {
                "model_i": mi,
                "model_j": mj,
                "n_centers": n,
                "both_degenerate": both_deg,
                "frac_both_degenerate": both_deg / n,
            }
        )

    total_centers = sum(r["n_centers"] for r in results)
    total_both_deg = sum(r["both_degenerate"] for r in results)
    frac_overall = total_both_deg / total_centers if total_centers > 0 else float("nan")

    return {
        "per_pair": results,
        "total_centers": total_centers,
        "total_both_degenerate": total_both_deg,
        "frac_both_degenerate_overall": frac_overall,
    }


# ============================================================
# 3) Визуализация — одна пара
# ============================================================


def _plot_single_pair(
    artifacts: Dict[str, np.ndarray],
    local_id_artifacts: Optional[Dict[str, np.ndarray]],
    model_i: str,
    model_j: str,
    plots_dir: str,
    metric_name: str,
    threshold: float = DEGENERATE_MAP_THRESHOLD_DEFAULT,
    plots_exts: Iterable[str] = ("png",),
    labels: Optional[Dict[str, str]] = None,
) -> None:
    """
    Строит детальные графики для пары (model_i, model_j):
      - гистограммы рангов для обоих направлений
      - распределение residuals для обоих направлений
      - сингулярные значения (медиана ± std по центрам)
    """
    if labels is None:
        labels = _load_diagnostics_labels(artifacts)

    sv_ij, res_ij, ranks_ij = _get_direction_data(artifacts, model_i, model_j)
    sv_ji, res_ji, ranks_ji = _get_direction_data(artifacts, model_j, model_i)
    rank_ref_ij, rank_ref_label = _get_direction_rank_reference(
        artifacts, model_i, model_j, ranks_ij
    )
    rank_ref_ji, _ = _get_direction_rank_reference(
        artifacts, model_j, model_i, ranks_ji
    )
    extra_ij = _get_direction_extra_data(artifacts, model_i, model_j)
    extra_ji = _get_direction_extra_data(artifacts, model_j, model_i)
    local_id_ij = _get_direction_local_id_data(
        local_id_artifacts or {}, model_i, model_j
    )
    local_id_ji = _get_direction_local_id_data(
        local_id_artifacts or {}, model_j, model_i
    )
    use_metric_ranks = bool(
        _direction_uses_metric_ranks(artifacts, model_i, model_j)
        and _direction_uses_metric_ranks(artifacts, model_j, model_i)
    )
    use_relative_residuals = bool(
        _direction_uses_relative_residuals(artifacts, model_i, model_j)
        and _direction_uses_relative_residuals(artifacts, model_j, model_i)
    )

    if use_relative_residuals:
        plot_res_ij = np.asarray(res_ij, dtype=np.float64)
        plot_res_ji = np.asarray(res_ji, dtype=np.float64)
        residual_xlabel = labels["residual_axis_label"]
        residual_short_label = labels["residual_short_label"]
        residual_title = "Распределение относительной ошибки по направлениям"
        residual_box_title = "Относительная ошибка по направлениям"
    else:
        plot_res_ij, normed_ij = _normalize_residuals(res_ij, extra_ij)
        plot_res_ji, normed_ji = _normalize_residuals(res_ji, extra_ji)
        residuals_normalized = bool(normed_ij and normed_ji)
        residual_xlabel = _normalized_residual_label(residuals_normalized)
        residual_short_label = _normalized_residual_short_label(residuals_normalized)
        residual_title = (
            "Распределение нормированной ошибки по направлениям"
            if residuals_normalized
            else "Распределение ошибки по направлениям"
        )
        residual_box_title = (
            "Нормированная ошибка по направлениям"
            if residuals_normalized
            else "Ошибка по направлениям"
        )

    label_ij = f"{model_i} → {model_j}"
    label_ji = f"{model_j} → {model_i}"

    n_centers = len(ranks_ij)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Диагностика локального отображения\n"
        f"Метрика: {short_metric_name(metric_name)} | Пара: {model_i} / {model_j} | "
        f"Центров на пару: {n_centers} | Порог вырожденности: {threshold:.2e}",
        fontsize=12,
    )

    # --- 1. Гистограмма рангов ---
    ax = axes[0, 0]
    ax.hist(ranks_ij, bins="auto", alpha=0.6, label=label_ij, color="steelblue")
    ax.hist(ranks_ji, bins="auto", alpha=0.6, label=label_ji, color="coral")
    ax.set_xlabel(_metric_value_axis_label(labels) if use_metric_ranks else _hard_rank_axis_label())
    ax.set_ylabel("Количество центров")
    ax.set_title("Гистограмма значений метрики по направлениям")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 2. Распределение residuals ---
    ax = axes[0, 1]
    ax.hist(plot_res_ij, bins=30, alpha=0.6, label=label_ij, color="steelblue")
    ax.hist(plot_res_ji, bins=30, alpha=0.6, label=label_ji, color="coral")
    ax.set_xlabel(residual_xlabel)
    ax.set_ylabel("Количество центров")
    ax.set_title(residual_title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 3. Residuals: boxplot ---
    ax = axes[0, 2]
    ax.boxplot(
        [plot_res_ij, plot_res_ji],
        labels=[label_ij, label_ji],
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
    )
    ax.set_ylabel(residual_short_label)
    ax.set_title(residual_box_title)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", labelsize=8)

    # --- 4. Медиана сингулярных значений по центрам (i->j) ---
    ax = axes[1, 0]
    _plot_singular_values_summary(
        ax, sv_ij, label_ij, color="steelblue", sv_threshold=threshold
    )
    ax.set_title(f"Сингулярные значения M\n{label_ij}")

    # --- 5. Медиана сингулярных значений по центрам (j->i) ---
    ax = axes[1, 1]
    _plot_singular_values_summary(
        ax, sv_ji, label_ji, color="coral", sv_threshold=threshold
    )
    ax.set_title(f"Сингулярные значения M\n{label_ji}")

    # --- 6. Таблица сводной статистики ---
    ax = axes[1, 2]
    ax.axis("off")
    stats_ij = _compute_direction_stats(
        sv_ij, plot_res_ij, ranks_ij, threshold=threshold
    )
    stats_ji = _compute_direction_stats(
        sv_ji, plot_res_ji, ranks_ji, threshold=threshold
    )
    extra_stats_ij = _compute_extra_stats(extra_ij)
    extra_stats_ji = _compute_extra_stats(extra_ji)
    det_stats_ij = _compute_determinedness_stats(rank_ref_ij, extra_ij)
    det_stats_ji = _compute_determinedness_stats(rank_ref_ji, extra_ji)
    local_id_stats_ij = _compute_local_id_stats(rank_ref_ij, extra_ij, local_id_ij)
    local_id_stats_ji = _compute_local_id_stats(rank_ref_ji, extra_ji, local_id_ji)
    res_short = residual_short_label
    table_data = [
        ["", label_ij[:20], label_ji[:20]],
        ["Центров", stats_ij["n_centers"], stats_ji["n_centers"]],
        [
            f"{_metric_value_short_label(labels)} (mean)",
            f"{stats_ij['rank_mean']:.2f}",
            f"{stats_ji['rank_mean']:.2f}",
        ],
        [f"{_metric_value_short_label(labels)} (std)", f"{stats_ij['rank_std']:.2f}", f"{stats_ji['rank_std']:.2f}"],
        [
            f"{_metric_value_short_label(labels)} (min/max)",
            f"{stats_ij['rank_min']:.2f}/{stats_ij['rank_max']:.2f}",
            f"{stats_ji['rank_min']:.2f}/{stats_ji['rank_max']:.2f}",
        ],
        [
            f"{res_short} (mean)",
            f"{stats_ij['residual_mean']:.2e}",
            f"{stats_ji['residual_mean']:.2e}",
        ],
        [
            f"{res_short} (std)",
            f"{stats_ij['residual_std']:.2e}",
            f"{stats_ji['residual_std']:.2e}",
        ],
        [
            "Выр-х центров",
            f"{stats_ij['n_degenerate']} ({stats_ij['frac_degenerate']:.1%})",
            f"{stats_ji['n_degenerate']} ({stats_ji['frac_degenerate']:.1%})",
        ],
    ]
    if np.isfinite(extra_stats_ij["neighbor_size_mean"]) or np.isfinite(
        extra_stats_ji["neighbor_size_mean"]
    ):
        table_data.append(
            [
                "Размер окр. (mean)",
                f"{extra_stats_ij['neighbor_size_mean']:.2f}",
                f"{extra_stats_ji['neighbor_size_mean']:.2f}",
            ]
        )
    if np.isfinite(det_stats_ij["point_count_mean"]) or np.isfinite(
        det_stats_ji["point_count_mean"]
    ):
        table_data.append(
            [
                "N_sys (mean)",
                f"{det_stats_ij['point_count_mean']:.2f}",
                f"{det_stats_ji['point_count_mean']:.2f}",
            ]
        )
    if np.isfinite(det_stats_ij["margin_mean"]) or np.isfinite(
        det_stats_ji["margin_mean"]
    ):
        table_data.append(
            [
                f"N_sys-{rank_ref_label}",
                f"{det_stats_ij['margin_mean']:.2f}",
                f"{det_stats_ji['margin_mean']:.2f}",
            ]
        )
    if np.isfinite(det_stats_ij["overdetermined_frac"]) or np.isfinite(
        det_stats_ji["overdetermined_frac"]
    ):
        table_data.append(
            [
                "N_sys>rank",
                f"{det_stats_ij['overdetermined_frac']:.1%}",
                f"{det_stats_ji['overdetermined_frac']:.1%}",
            ]
        )
    if np.isfinite(local_id_stats_ij["id_x_mean"]) or np.isfinite(
        local_id_stats_ji["id_x_mean"]
    ):
        table_data.append(
            [
                "ID_x (mean)",
                f"{local_id_stats_ij['id_x_mean']:.2f}",
                f"{local_id_stats_ji['id_x_mean']:.2f}",
            ]
        )
    if np.isfinite(local_id_stats_ij["id_y_mean"]) or np.isfinite(
        local_id_stats_ji["id_y_mean"]
    ):
        table_data.append(
            [
                "ID_y (mean)",
                f"{local_id_stats_ij['id_y_mean']:.2f}",
                f"{local_id_stats_ji['id_y_mean']:.2f}",
            ]
        )
    if np.isfinite(local_id_stats_ij["rank_minus_id_min_mean"]) or np.isfinite(
        local_id_stats_ji["rank_minus_id_min_mean"]
    ):
        table_data.append(
            [
                f"{rank_ref_label}-min(ID)",
                f"{local_id_stats_ij['rank_minus_id_min_mean']:.2f}",
                f"{local_id_stats_ji['rank_minus_id_min_mean']:.2f}",
            ]
        )
    if np.isfinite(extra_stats_ij["neighbor_distance_mean"]) or np.isfinite(
        extra_stats_ji["neighbor_distance_mean"]
    ):
        table_data.append(
            [
                "Dist (mean)",
                f"{extra_stats_ij['neighbor_distance_mean']:.2e}",
                f"{extra_stats_ji['neighbor_distance_mean']:.2e}",
            ]
        )
    if np.isfinite(extra_stats_ij["sigma_mean"]) or np.isfinite(extra_stats_ji["sigma_mean"]):
        table_data.append(
            [
                "Sigma (mean)",
                f"{extra_stats_ij['sigma_mean']:.2e}",
                f"{extra_stats_ji['sigma_mean']:.2e}",
            ]
        )
    if np.isfinite(extra_stats_ij["eps_mean"]) or np.isfinite(extra_stats_ji["eps_mean"]):
        table_data.append(
            [
                "Eps (mean)",
                f"{extra_stats_ij['eps_mean']:.2e}",
                f"{extra_stats_ji['eps_mean']:.2e}",
            ]
        )
    if np.isfinite(extra_stats_ij["inlier_frac_mean"]) or np.isfinite(
        extra_stats_ji["inlier_frac_mean"]
    ):
        table_data.append(
            [
                "Inlier frac (mean)",
                f"{extra_stats_ij['inlier_frac_mean']:.2%}",
                f"{extra_stats_ji['inlier_frac_mean']:.2%}",
            ]
        )
    tbl = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    ax.set_title("Сводная статистика", fontsize=10)

    plt.tight_layout()
    fname = f"{metric_name}_pair_{model_i}_vs_{model_j}.png"
    fpath = os.path.join(plots_dir, fname)
    saved_paths = _save_figure_variants(fig, fpath, plots_exts, dpi=150)
    plt.close(fig)
    for save_path in saved_paths:
        print(f"  Сохранён график: {save_path}")


def _get_direction_local_id_data(
    local_id_artifacts: Dict[str, np.ndarray], model_i: str, model_j: str
) -> DirectionLocalIDData:
    if not local_id_artifacts:
        return DirectionLocalIDData()
    return DirectionLocalIDData(
        intrinsic_dims_x=_get_optional_direction_array(
            local_id_artifacts, model_i, model_j, "intrinsic_dims_x"
        ),
        intrinsic_dims_y=_get_optional_direction_array(
            local_id_artifacts, model_i, model_j, "intrinsic_dims_y"
        ),
        neighbor_sizes_x=_get_optional_direction_array(
            local_id_artifacts, model_i, model_j, "neighbor_sizes_x"
        ),
        neighbor_sizes_y=_get_optional_direction_array(
            local_id_artifacts, model_i, model_j, "neighbor_sizes_y"
        ),
    )


def _plot_singular_values_summary(
    ax: plt.Axes,
    sv_arr: np.ndarray,
    label: str,
    color: str = "steelblue",
    sv_threshold: float = DEGENERATE_SV_THRESHOLD_DEFAULT,
) -> None:
    """
    Строит медиану сингулярных значений по всем центрам с полосой ±std.
    Ось X — номер сингулярного значения, ось Y — его величина.
    """
    # Определяем максимальную длину вектора sv среди центров.
    max_len = max(
        (len(sv) for sv in sv_arr if sv is not None and len(sv) > 0), default=0
    )
    if max_len == 0:
        ax.text(
            0.5, 0.5, "Нет данных", ha="center", va="center", transform=ax.transAxes
        )
        return

    # Собираем матрицу (n_centers, max_len), заполняя нулями если sv короче.
    sv_matrix = np.zeros((len(sv_arr), max_len), dtype=np.float32)
    for c, sv in enumerate(sv_arr):
        if sv is not None and len(sv) > 0:
            l = min(len(sv), max_len)
            sv_matrix[c, :l] = sv[:l]

    median_sv = np.median(sv_matrix, axis=0)
    std_sv = np.std(sv_matrix, axis=0)
    x = np.arange(1, max_len + 1)

    ax.plot(x, median_sv, "o-", color=color, markersize=4, label="медиана")
    ax.fill_between(
        x, median_sv - std_sv, median_sv + std_sv, alpha=0.25, color=color, label="±std"
    )
    ax.axhline(
        sv_threshold,
        color="red",
        linestyle="--",
        linewidth=0.8,
        label=f"порог={sv_threshold:.0e}",
    )
    ax.set_xlabel("Номер сингулярного значения")
    ax.set_ylabel("Значение (лог. шкала)")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


# ============================================================
# 4) Визуализация — агрегированная по всем парам
# ============================================================


def _plot_aggregated(
    artifacts: Dict[str, np.ndarray],
    directions: List[Tuple[str, str]],
    both_deg_stats: Dict,
    plots_dir: str,
    metric_name: str,
    threshold: float = DEGENERATE_MAP_THRESHOLD_DEFAULT,
    metric_spec_str: str = "",
    plots_exts: Iterable[str] = ("png",),
    labels: Optional[Dict[str, str]] = None,
    downstream_scores: Optional[Dict[str, float]] = None,
) -> None:
    """
    Строит агрегированные графики по всем парам:
      - общая гистограмма рангов
      - общее распределение residuals
      - кривая значения метрики по направлениям
      - определённость системы: N_sys - rank по направлениям
    """
    if labels is None:
        labels = _load_diagnostics_labels(artifacts)
    all_ranks = []
    all_residuals = []
    point_count_means = []
    margin_means = []
    rank_means = []
    rank_stds = []
    rank_reference_means = []
    direction_labels = []
    direction_is_forward = []
    normalized_flags = []
    direction_stats = {}

    use_metric_ranks = bool(
        _direction_uses_metric_ranks(artifacts, directions[0][0], directions[0][1])
    )
    use_relative_residuals = bool(
        _direction_uses_relative_residuals(artifacts, directions[0][0], directions[0][1])
    )

    for mi, mj in directions:
        sv, res, ranks = _get_direction_data(artifacts, mi, mj)
        rank_ref, rank_ref_label = _get_direction_rank_reference(
            artifacts, mi, mj, ranks
        )
        extra = _get_direction_extra_data(artifacts, mi, mj)
        if _direction_uses_relative_residuals(artifacts, mi, mj):
            work_res = np.asarray(res, dtype=np.float64)
            is_normalized = True
        else:
            work_res, is_normalized = _normalize_residuals(res, extra)
        stats = _compute_direction_stats(sv, work_res, ranks, threshold=threshold)
        det_stats = _compute_determinedness_stats(rank_ref, extra)
        all_ranks.extend(ranks.tolist())
        all_residuals.extend(work_res.tolist())
        point_count_means.append(det_stats["point_count_mean"])
        margin_means.append(det_stats["margin_mean"])
        rank_means.append(stats["rank_mean"])
        rank_stds.append(stats["rank_std"])
        rank_reference_means.append(_safe_mean(rank_ref))
        direction_labels.append(f"{mi[:10]}→{mj[:10]}")
        direction_is_forward.append(mi <= mj)
        normalized_flags.append(is_normalized)
        direction_stats[(mi, mj)] = {
            "rank_mean": stats["rank_mean"],
            "residual_mean": stats["residual_mean"],
        }

    all_ranks = np.array(all_ranks)
    all_residuals = np.array(all_residuals)
    residuals_normalized = bool(all(normalized_flags)) if normalized_flags else False

    # n_centers — реальное количество центров первого направления, читается из артефактов.
    _, _, _ranks_first = _get_direction_data(
        artifacts, directions[0][0], directions[0][1]
    )
    n_centers_per_direction = len(_ranks_first)

    fig, axes_full = plt.subplots(3, 2, figsize=(14, 15))
    axes = axes_full[:2, :]
    diff_ax = axes_full[2, 0]
    unused_ax = axes_full[2, 1]
    fig.suptitle(
        f"Агрегированная диагностика локального отображения\n"
        f"Метрика: {short_metric_name(metric_name)}"
        + (f" | {metric_spec_str}" if metric_spec_str else "")
        + f" | Центров на пару: {n_centers_per_direction} | Пар моделей: {len(directions)} | "
        f"Порог вырожденности: {threshold:.2e}",
        fontsize=12,
    )

    # --- 1. Гистограмма рангов по всем парам и центрам ---
    ax = axes[0, 0]
    ax.hist(all_ranks, bins="auto", color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(
        np.mean(all_ranks),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"среднее={np.mean(all_ranks):.2f}",
    )
    ax.axvline(
        np.median(all_ranks),
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"медиана={np.median(all_ranks):.2f}",
    )
    ax.set_xlabel(_metric_value_axis_label(labels) if use_metric_ranks else _hard_rank_axis_label())
    ax.set_ylabel("Количество центров (все пары)")
    ax.set_title(
        f"Гистограмма значений метрики (X→Y и Y→X)\nstd={np.std(all_ranks):.3f}, медиана={np.median(all_ranks):.2f}"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 2. Распределение residuals по всем парам ---
    ax = axes[0, 1]
    ax.hist(all_residuals, bins=40, color="coral", edgecolor="white", linewidth=0.5)
    ax.axvline(
        np.mean(all_residuals),
        color="darkred",
        linestyle="--",
        linewidth=1.5,
        label=f"среднее={np.mean(all_residuals):.2e}",
    )
    ax.axvline(
        np.median(all_residuals),
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"медиана={np.median(all_residuals):.2e}",
    )
    ax.set_xlabel(
        labels["residual_axis_label"]
        if use_relative_residuals
        else _normalized_residual_label(residuals_normalized)
    )
    ax.set_ylabel("Количество центров (все пары)")
    ax.set_title(
        "Распределение относительной ошибки (X→Y и Y→X)"
        if use_relative_residuals
        else (
            "Распределение нормированной ошибки (X→Y и Y→X)"
            if residuals_normalized
            else "Распределение ошибки (X→Y и Y→X)"
        )
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 3. Значение метрики по направлениям ---
    ax = axes[1, 0]
    x = np.arange(len(direction_labels))
    rank_means_arr = np.asarray(rank_means, dtype=np.float64)
    rank_stds_arr = np.asarray(rank_stds, dtype=np.float64)
    direction_is_forward_arr = np.asarray(direction_is_forward, dtype=bool)
    finite_metric = np.isfinite(rank_means_arr)
    ax.set_xticks(x)
    ax.set_xticklabels(direction_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(f"Среднее {_metric_value_short_label(labels)} по центрам")
    ax.set_title("Значение метрики по направлениям (X→Y и Y→X)")
    ax.grid(True, alpha=0.3, axis="y")
    if np.any(finite_metric):
        for is_forward, color, ecolor, marker, name in [
            (True, "steelblue", "lightsteelblue", "o", "прямое направление"),
            (False, "darkorange", "moccasin", "s", "обратное направление"),
        ]:
            mask = finite_metric & (direction_is_forward_arr == is_forward)
            if not np.any(mask):
                continue
            ax.errorbar(
                x[mask],
                rank_means_arr[mask],
                yerr=rank_stds_arr[mask],
                fmt=f"{marker}-",
                color=color,
                ecolor=ecolor,
                elinewidth=1.2,
                capsize=3,
                markersize=4,
                linewidth=1.5,
                label=f"{name}: mean({_metric_value_short_label(labels)}) ± std",
            )
        finite_upper = rank_means_arr[finite_metric] + np.nan_to_num(
            rank_stds_arr[finite_metric], nan=0.0
        )
        y_max = float(np.nanmax(finite_upper)) if finite_upper.size > 0 else 1.0
        y_min = float(np.nanmin(rank_means_arr[finite_metric]))
        y_pad = max(1e-6, 0.08 * max(abs(y_max), abs(y_min), 1.0))
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.legend(fontsize=8, loc="best")
    else:
        ax.text(
            0.5,
            0.5,
            "Нет данных о значении метрики",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )

    # --- 4. Определённость системы: число точек против ранга ---
    ax = axes[1, 1]
    finite_margin = np.isfinite(margin_means)
    finite_points = np.isfinite(point_count_means)
    finite_rank = np.isfinite(rank_reference_means)
    if np.any(finite_margin) or np.any(finite_points):
        x2 = np.arange(len(direction_labels))
        bars2 = ax.bar(x2, margin_means, color="seagreen", alpha=0.8, label=f"mean(N_sys - {rank_ref_label})")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xticks(x2)
        ax.set_xticklabels(direction_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(f"Средний запас N_sys - {rank_ref_label}")
        ax.set_title(
            "Определённость локальной системы по направлениям\n"
            "(N_sys > rank: переопределена, N_sys <= rank: недоопределена/на границе)"
        )
        ax.grid(True, alpha=0.3, axis="y")
        if np.any(finite_margin):
            max_abs_margin = max(1.0, float(np.nanmax(np.abs(margin_means))) * 1.2)
            ax.set_ylim(-max_abs_margin, max_abs_margin)
        ax2 = ax.twinx()
        if np.any(finite_points):
            ax2.plot(
                x2,
                point_count_means,
                "D--",
                color="darkred",
                markersize=5,
                label="mean(N_sys)",
            )
        if np.any(finite_rank):
            ax2.plot(
                x2,
                rank_reference_means,
                "o-.",
                color="navy",
                markersize=4,
                label=f"mean({rank_ref_label})",
            )
        ax2.set_ylabel("Среднее значение")
        lines1, lbls1 = ax.get_legend_handles_labels()
        lines2, lbls2 = ax2.get_legend_handles_labels()
        legend = ax2.legend(
            lines1 + lines2,
            lbls1 + lbls2,
            fontsize=8,
            loc="best",
            framealpha=0.85,
            facecolor="white",
            edgecolor="0.8",
        )
        legend.set_zorder(1000)
    else:
        ax.text(
            0.5,
            0.5,
            "Нет данных о размере окрестности\nили числе инлайеров",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_title("Определённость локальной системы")

    # --- 5. Разница между прямым и обратным отображением ---
    seen_pairs = set()
    pair_labels = []
    rank_mean_diffs = []
    residual_mean_diffs = []
    accuracy_diffs = []
    for mi, mj in directions:
        if mi == mj:
            continue
        pair_key = tuple(sorted((mi, mj)))
        if pair_key in seen_pairs or (mj, mi) not in direction_stats:
            continue
        seen_pairs.add(pair_key)

        s_ij = direction_stats[(mi, mj)]
        s_ji = direction_stats[(mj, mi)]
        rank_mean_diffs.append(abs(s_ij["rank_mean"] - s_ji["rank_mean"]))
        residual_mean_diffs.append(
            abs(s_ij["residual_mean"] - s_ji["residual_mean"])
        )
        if (
            downstream_scores
            and mi in downstream_scores
            and mj in downstream_scores
        ):
            accuracy_diffs.append(abs(downstream_scores[mi] - downstream_scores[mj]))
        else:
            accuracy_diffs.append(np.nan)
        pair_labels.append(f"{mi[:10]}↔{mj[:10]}")

    x_diff = np.arange(len(pair_labels))
    if pair_labels:
        rank_diff_arr = np.asarray(rank_mean_diffs, dtype=np.float64)
        residual_diff_arr = np.asarray(residual_mean_diffs, dtype=np.float64)
        diff_ax.bar(
            x_diff,
            rank_diff_arr,
            color="steelblue",
            alpha=0.8,
            label=f"|Δ mean({_metric_value_short_label(labels)})|",
        )
        diff_ax.set_xticks(x_diff)
        diff_ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=7)
        diff_ax.set_ylabel(f"|Δ mean({_metric_value_short_label(labels)})|")
        diff_ax.set_title(
            "Разница между прямым и обратным отображением по парам моделей"
        )
        diff_ax.grid(True, alpha=0.3, axis="y")
        if np.any(np.isfinite(rank_diff_arr)):
            y_max = float(np.nanmax(rank_diff_arr))
            diff_ax.set_ylim(0.0, max(1e-6, y_max * 1.15))

        diff_ax2 = diff_ax.twinx()
        diff_ax2.plot(
            x_diff,
            residual_diff_arr,
            "o-",
            color="darkorange",
            markersize=4,
            linewidth=1.3,
            label=f"|Δ mean({labels['residual_short_label']})|",
        )
        diff_ax2.set_ylabel(f"|Δ mean({labels['residual_short_label']})|")

        lines1, lbls1 = diff_ax.get_legend_handles_labels()
        lines2, lbls2 = diff_ax2.get_legend_handles_labels()
        diff_ax2.legend(
            lines1 + lines2,
            lbls1 + lbls2,
            fontsize=8,
            loc="best",
            framealpha=0.85,
            facecolor="white",
            edgecolor="0.8",
        )

        accuracy_diff_arr = np.asarray(accuracy_diffs, dtype=np.float64)
        if np.any(np.isfinite(accuracy_diff_arr)):
            unused_ax.plot(
                x_diff,
                accuracy_diff_arr,
                "s-",
                color="darkgreen",
                markersize=4,
                linewidth=1.4,
                label="|Δ accuracy|",
            )
            unused_ax.set_xticks(x_diff)
            unused_ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=7)
            unused_ax.set_ylabel("|Δ accuracy|")
            unused_ax.set_title("Разница accuracy по парам моделей")
            unused_ax.grid(True, alpha=0.3, axis="y")
            unused_ax.legend(fontsize=8, loc="best")
            y_max = float(np.nanmax(accuracy_diff_arr))
            unused_ax.set_ylim(0.0, max(1e-6, y_max * 1.15))
        else:
            unused_ax.axis("off")
            unused_ax.text(
                0.5,
                0.5,
                "Нет downstream accuracy",
                ha="center",
                va="center",
                transform=unused_ax.transAxes,
                fontsize=11,
            )
    else:
        diff_ax.text(
            0.5,
            0.5,
            "Нет пар с обоими направлениями",
            ha="center",
            va="center",
            transform=diff_ax.transAxes,
            fontsize=11,
        )
        diff_ax.set_title("Разница между прямым и обратным отображением")
        unused_ax.axis("off")

    plt.tight_layout()
    fname = f"{metric_name}_aggregated_diagnostics.png"
    fpath = os.path.join(plots_dir, fname)
    saved_paths = _save_figure_variants(fig, fpath, plots_exts, dpi=150)
    plt.close(fig)
    for save_path in saved_paths:
        print(f"  Сохранён агрегированный график: {save_path}")


# ============================================================
# 5) Сохранение текстового отчёта
# ============================================================


def _save_report(
    directions: List[Tuple[str, str]],
    artifacts: Dict[str, np.ndarray],
    local_id_artifacts: Optional[Dict[str, np.ndarray]],
    both_deg_stats: Dict,
    reports_dir: str,
    metric_name: str,
) -> None:
    """Сохраняет текстовый отчёт с ключевыми числами диагностики."""
    labels = _load_diagnostics_labels(artifacts)
    use_relative_residuals = bool(
        _direction_uses_relative_residuals(artifacts, directions[0][0], directions[0][1])
    )
    res_description = labels["residual_description"]
    lines = []
    lines.append(f"Диагностика метрики: {metric_name}")
    lines.append("=" * 60)
    lines.append(f"Всего направлений: {len(directions)}")
    lines.append("")

    all_rank_stds = []
    extra_rows = []
    normalized_flags = []
    for mi, mj in sorted(directions):
        sv, res, ranks = _get_direction_data(artifacts, mi, mj)
        rank_ref, rank_ref_label = _get_direction_rank_reference(
            artifacts, mi, mj, ranks
        )
        extra_data = _get_direction_extra_data(artifacts, mi, mj)
        local_id_data = _get_direction_local_id_data(
            local_id_artifacts or {}, mi, mj
        )
        if _direction_uses_relative_residuals(artifacts, mi, mj):
            work_res = np.asarray(res, dtype=np.float64)
            is_normalized = True
        else:
            work_res, is_normalized = _normalize_residuals(res, extra_data)
        s = _compute_direction_stats(sv, work_res, ranks)
        extra = _compute_extra_stats(_get_direction_extra_data(artifacts, mi, mj))
        det = _compute_determinedness_stats(rank_ref, extra_data)
        lid = _compute_local_id_stats(rank_ref, extra_data, local_id_data)
        all_rank_stds.append(s["rank_std"])
        normalized_flags.append(is_normalized)
        direction_str = f"{mi}→{mj}"
        extra_rows.append(
            {
                "direction": direction_str,
                "id_x_mean": lid["id_x_mean"],
                "id_y_mean": lid["id_y_mean"],
                "rank_minus_id_min_mean": lid["rank_minus_id_min_mean"],
                "system_minus_id_min_mean": lid["system_minus_id_min_mean"],
                "point_count_mean": det["point_count_mean"],
                "margin_mean": det["margin_mean"],
                "overdetermined_frac": det["overdetermined_frac"],
                "neighbor_size_mean": extra["neighbor_size_mean"],
                "neighbor_distance_mean": extra["neighbor_distance_mean"],
                "sigma_mean": extra["sigma_mean"],
                "eps_mean": extra["eps_mean"],
                "inlier_frac_mean": extra["inlier_frac_mean"],
                "inlier_frac_std": extra["inlier_frac_std"],
            }
        )

    if use_relative_residuals:
        res_short = labels["residual_short_label"]
    else:
        res_short = _normalized_residual_short_label(bool(all(normalized_flags)))

    lines.append("--- Статистика по направлениям ---")
    metric_short = _metric_value_short_label(labels)
    header = (
        f"{'Направление':<35} {f'{metric_short}(mean)':<12} {f'{metric_short}(std)':<12} "
        f"{f'{res_short}(mean)':<14} {f'{res_short}(std)':<14} {'Выр-х,%':<10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for mi, mj in sorted(directions):
        sv, res, ranks = _get_direction_data(artifacts, mi, mj)
        extra_data = _get_direction_extra_data(artifacts, mi, mj)
        if _direction_uses_relative_residuals(artifacts, mi, mj):
            work_res = np.asarray(res, dtype=np.float64)
        else:
            work_res, _ = _normalize_residuals(res, extra_data)
        s = _compute_direction_stats(sv, work_res, ranks)
        direction_str = f"{mi}→{mj}"
        lines.append(
            f"{direction_str:<35} {s['rank_mean']:<12.3f} {s['rank_std']:<12.3f} "
            f"{s['residual_mean']:<14.4f} {s['residual_std']:<14.4f} "
            f"{s['frac_degenerate']:<10.1%}"
        )

    lines.append("")
    if use_relative_residuals:
        lines.append(f"{res_short} = {res_description}")
    elif all(normalized_flags):
        lines.append(
            "ResN = нормированная RMS-подобная ошибка ||X_c M - Y_c||_F / sqrt(N_eff)."
        )
    else:
        lines.append(
            "Res = raw residual для старых артефактов без данных, необходимых для нормировки."
        )
    lines.append("")
    lines.append(
        f"Средняя std значения метрики по всем направлениям: {np.mean(all_rank_stds):.4f}"
    )
    lines.append("")

    have_extra = any(
        np.isfinite(row["id_x_mean"])
        or np.isfinite(row["id_y_mean"])
        or np.isfinite(row["rank_minus_id_min_mean"])
        or np.isfinite(row["system_minus_id_min_mean"])
        or np.isfinite(row["point_count_mean"])
        or np.isfinite(row["margin_mean"])
        or np.isfinite(row["neighbor_size_mean"])
        or np.isfinite(row["neighbor_distance_mean"])
        or np.isfinite(row["sigma_mean"])
        or np.isfinite(row["eps_mean"])
        or np.isfinite(row["inlier_frac_mean"])
        for row in extra_rows
    )
    if have_extra:
        lines.append("--- Дополнительные артефакты новых методов ---")
        extra_header = (
            f"{'Направление':<35} {'ID_x':<10} {'ID_y':<10} {f'{rank_ref_label}-ID':<10} "
            f"{'N-ID':<10} {'N_sys(mean)':<12} {f'N-{rank_ref_label}':<12} "
            f"{'Overdet,%':<12} {'Nhood(mean)':<12} {'Dist(mean)':<12} "
            f"{'Sigma(mean)':<12} {'Eps(mean)':<12} {'Inlier(mean)':<12}"
        )
        lines.append(extra_header)
        lines.append("-" * len(extra_header))
        for row in extra_rows:
            def _fmt(val: float, fmt: str) -> str:
                return fmt.format(val) if np.isfinite(val) else "n/a"

            lines.append(
                f"{row['direction']:<35} "
                f"{_fmt(row['id_x_mean'], '{:.2f}'):<10} "
                f"{_fmt(row['id_y_mean'], '{:.2f}'):<10} "
                f"{_fmt(row['rank_minus_id_min_mean'], '{:.2f}'):<10} "
                f"{_fmt(row['system_minus_id_min_mean'], '{:.2f}'):<10} "
                f"{_fmt(row['point_count_mean'], '{:.2f}'):<12} "
                f"{_fmt(row['margin_mean'], '{:.2f}'):<12} "
                f"{_fmt(row['overdetermined_frac'], '{:.1%}'):<12} "
                f"{_fmt(row['neighbor_size_mean'], '{:.2f}'):<12} "
                f"{_fmt(row['neighbor_distance_mean'], '{:.3e}'):<12} "
                f"{_fmt(row['sigma_mean'], '{:.3e}'):<12} "
                f"{_fmt(row['eps_mean'], '{:.3e}'):<12} "
                f"{_fmt(row['inlier_frac_mean'], '{:.2%}'):<12}"
            )
        lines.append("")

    lines.append("--- Одновременная вырожденность X→Y и Y→X в одном центре ---")
    overall = both_deg_stats["frac_both_degenerate_overall"]
    lines.append(f"Всего центров (по всем парам): {both_deg_stats['total_centers']}")
    lines.append(
        f"Центров, где оба вырождены: {both_deg_stats['total_both_degenerate']} ({overall:.2%})"
    )
    lines.append("")

    for r in sorted(
        both_deg_stats["per_pair"], key=lambda x: -x["frac_both_degenerate"]
    ):
        lines.append(
            f"  {r['model_i']} / {r['model_j']}: "
            f"{r['both_degenerate']}/{r['n_centers']} ({r['frac_both_degenerate']:.2%})"
        )

    report_path = os.path.join(reports_dir, f"{metric_name}_diagnostics_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Сохранён отчёт: {report_path}")

    # Также сохраняем JSON для удобной машинной обработки.
    json_report = {
        "metric_name": metric_name,
        "n_directions": len(directions),
        "both_degenerate": both_deg_stats,
        "mean_rank_std": float(np.mean(all_rank_stds)),
        "uses_relative_residuals": use_relative_residuals,
        "residuals_normalized": bool(all(normalized_flags)) if normalized_flags else False,
        "direction_extras": extra_rows,
    }
    json_path = os.path.join(reports_dir, f"{metric_name}_diagnostics_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2)
    print(f"  Сохранён JSON-отчёт: {json_path}")


# ============================================================
# 6) Визуализация — сводный график по всем метрикам
# ============================================================


def _plot_summary(
    metrics_data: List[Dict],
    plots_dir: str,
    threshold: float,
    plots_exts: Iterable[str] = ("png",),
) -> None:
    """
    Строит сводный график сравнения всех метрик по трём вопросам руководителя:
      1. Ошибка решения (relative или normalized residual, в зависимости от артефактов)
      2. Стабильность ранга (std ранга по всем центрам и парам)
      3. Определённость системы: N_sys - rank
      4. Распределение значений самой метрики по центрам

    metrics_data — список словарей, по одному на метрику:
      {
        "metric_name": str,
        "rank_means":  np.ndarray,
        "rank_stds":   np.ndarray,
        "res_means":   np.ndarray,
        "res_stds":    np.ndarray,
        "system_point_means": np.ndarray,
        "determined_margin_means": np.ndarray,
        "overdetermined_fracs": np.ndarray,
        "n_centers":   int,
        "n_directions": int,
        "labels": dict,   # подписи из diagnostics_meta_json
      }
    """
    if not metrics_data:
        print("  [WARN] Нет данных для сводного графика.")
        return

    labels = [d["metric_name"] for d in metrics_data]
    # Укорачиваем длинные имена для читаемости осей.
    short_labels = [short_metric_name(l) for l in labels]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(5, 1, figsize=(max(12, len(labels) * 1.2), 22))
    n_centers = metrics_data[0]["n_centers"]
    fig.suptitle(
        f"Сводная диагностика локальных отображений\n"
        f"Центров на пару: {n_centers} | Метрик: {len(labels)} | "
        f"Порог вырожденности: {threshold:.0e}",
        fontsize=13,
    )

    # --- 1. Стабильность значения метрики: среднее std по направлениям ---
    ax = axes[0]
    mean_rank_stds = [float(np.mean(d["rank_stds"])) for d in metrics_data]
    mean_rank_means = [float(np.mean(d["rank_means"])) for d in metrics_data]
    bars = ax.bar(
        x, mean_rank_stds, color="steelblue", alpha=0.8, label="mean(std значения)"
    )
    # Поверх баров — среднее значение метрики как точки.
    ax2 = ax.twinx()
    ax2.plot(
        x, mean_rank_means, "D--", color="darkred", markersize=6, label="mean(value)"
    )
    ax2.set_ylabel("Среднее значение метрики", color="darkred")
    ax2.tick_params(axis="y", labelcolor="darkred")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Среднее std значения метрики по направлениям")
    ax.set_title("Стабильность значения метрики\n(чем меньше std — тем стабильнее)")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, mean_rank_stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(mean_rank_stds) * 0.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    lines1, lbls1 = ax.get_legend_handles_labels()
    lines2, lbls2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbls1 + lbls2, fontsize=8, loc="upper right")

    # --- 2. Ошибка решения ---
    ax = axes[1]
    mean_res = [float(np.mean(d["res_means"])) for d in metrics_data]
    std_res = [float(np.mean(d["res_stds"])) for d in metrics_data]
    all_relative_residuals = all(bool(d.get("uses_relative_residuals", False)) for d in metrics_data)
    all_residuals_normalized = all(
        bool(d.get("residuals_normalized", False)) for d in metrics_data
    )
    if all_relative_residuals:
        residual_labels = {
            d.get("labels", _FALLBACK_LABELS)["residual_summary_label"]
            for d in metrics_data
        }
        res_summary_label = (
            next(iter(residual_labels))
            if len(residual_labels) == 1
            else "Средняя относительная ошибка (формула зависит от метрики/геометрии)"
        )
    else:
        res_summary_label = _summary_residual_axis_label(all_residuals_normalized)
    bars = ax.bar(
        x,
        mean_res,
        color="coral",
        alpha=0.8,
        yerr=std_res,
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "darkred"},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(res_summary_label)
    ax.set_title(
        (
            "Относительная ошибка решения линейного уравнения\n"
            "(чем меньше — тем лучше линейное приближение; сопоставима между метриками)"
            if all_relative_residuals
            else "Нормированная ошибка решения линейного уравнения\n"
            "(чем меньше — тем лучше линейное приближение)"
            if all_residuals_normalized
            else "Ошибка решения линейного уравнения\n"
            "(часть артефактов не содержит данных для нормировки)"
        )
    )
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, mean_res):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(mean_res) * 0.01,
            f"{val:.2e}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # --- 3. Определённость системы ---
    ax = axes[2]
    mean_margins = [_safe_mean(d["determined_margin_means"]) for d in metrics_data]
    mean_point_counts = [_safe_mean(d["system_point_means"]) for d in metrics_data]
    rank_reference_labels = {
        str(d.get("rank_reference_label", _metric_value_short_label(d.get("labels", _FALLBACK_LABELS))))
        for d in metrics_data
    }
    rank_reference_label = (
        next(iter(rank_reference_labels))
        if len(rank_reference_labels) == 1
        else "rank reference"
    )
    mean_rank_values = [
        _safe_mean(d.get("rank_reference_means", d.get("rank_means")))
        for d in metrics_data
    ]
    bars = ax.bar(x, mean_margins, color="seagreen", alpha=0.8, label=f"mean(N_sys - {rank_reference_label})")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"Средний запас N_sys - {rank_reference_label}")
    ax.set_title(
        "Определённость локальной системы\n"
        "(N_sys > rank: переопределена, N_sys <= rank: недоопределена/на границе)"
    )
    ax.grid(True, alpha=0.3, axis="y")
    finite_margins = np.isfinite(mean_margins)
    if np.any(finite_margins):
        max_abs_margin = max(1.0, float(np.nanmax(np.abs(mean_margins))) * 1.2)
        ax.set_ylim(-max_abs_margin, max_abs_margin)
    ax2 = ax.twinx()
    if np.any(np.isfinite(mean_point_counts)):
        ax2.plot(
            x,
            mean_point_counts,
            "D--",
            color="darkred",
            markersize=6,
            label="mean(N_sys)",
        )
    if np.any(np.isfinite(mean_rank_values)):
        ax2.plot(
            x,
            mean_rank_values,
            "o-.",
            color="navy",
            markersize=5,
            label=f"mean({rank_reference_label})",
        )
    ax2.set_ylabel("Среднее значение")
    for bar, val in zip(bars, mean_margins):
        if np.isfinite(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.01 * ax.get_ylim()[1] if val >= 0 else -0.06 * ax.get_ylim()[1]),
                f"{val:.1f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=7,
            )
    lines1, lbls1 = ax.get_legend_handles_labels()
    lines2, lbls2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbls1 + lbls2, fontsize=8, loc="upper right")

    # --- 4. Спектр сингулярных значений по всем метрикам на одном поле ---
    ax = axes[3]
    cmap = plt.get_cmap("tab10")
    for idx, d in enumerate(metrics_data):
        sp_med = d.get("spectrum_median", np.array([]))
        sp_std = d.get("spectrum_std", np.array([]))
        if len(sp_med) == 0:
            continue
        color = cmap(idx % 10)
        short_name = short_metric_name(d["metric_name"])
        # Нормируем ось X как долю от длины спектра — сравниваем метрики с разной размерностью.
        x_norm = np.linspace(0, 1, len(sp_med))
        ax.plot(x_norm, sp_med, color=color, linewidth=1.5, label=short_name)
        ax.fill_between(
            x_norm, sp_med - sp_std, sp_med + sp_std, color=color, alpha=0.12
        )
    ax.set_xlabel("Нормированный индекс сингулярного значения (0 = макс, 1 = мин)")
    ax.set_ylabel("Сингулярное значение (медиана по центрам и парам)")
    ax.set_title(
        "Спектр сингулярных значений матрицы M\n"
        "(медиана ± std по всем центрам, X→Y и Y→X)"
    )
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- 5. Распределение значений метрики по центрам ---
    ax = axes[4]
    metric_value_arrays = []
    metric_value_positions = []
    metric_value_labels = []
    for idx, d in enumerate(metrics_data):
        vals = np.asarray(d.get("all_metric_values", np.array([])), dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        metric_value_arrays.append(vals)
        metric_value_positions.append(idx + 1)
        metric_value_labels.append(short_metric_name(d["metric_name"]))

    if metric_value_arrays:
        box = ax.boxplot(
            metric_value_arrays,
            positions=metric_value_positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="darkred", linewidth=1.4),
        )
        for patch in box["boxes"]:
            patch.set_facecolor("lightsteelblue")
            patch.set_alpha(0.85)
        ax.set_xticks(metric_value_positions)
        ax.set_xticklabels(metric_value_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Значение метрики по направлению для данного центра")
        ax.set_title(
            "Распределение значений метрики по центрам и направлениям\n"
            "(boxplot по всем центрам, X→Y и Y→X)"
        )
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Нет доступных per-center значений метрики для summary.",
            ha="center",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    fpath = os.path.join(plots_dir, "summary_diagnostics.png")
    saved_paths = _save_figure_variants(fig, fpath, plots_exts, dpi=150)
    plt.close(fig)
    for save_path in saved_paths:
        print(f"  Сохранён сводный график: {save_path}")


def _plot_summary_local_id(
    metrics_data: List[Dict],
    plots_dir: str,
    plots_exts: Iterable[str] = ("png",),
) -> None:
    if not metrics_data:
        return

    has_local_id = any(
        np.any(np.isfinite(d.get("local_id_min_means", np.array([]))))
        for d in metrics_data
    )
    if not has_local_id:
        return

    labels = [short_metric_name(d["metric_name"]) for d in metrics_data]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(labels) * 1.2), 14))
    fig.suptitle("Сводная диагностика локальной размерности", fontsize=13)

    mean_id_x = [_safe_mean(d.get("local_id_x_means")) for d in metrics_data]
    mean_id_y = [_safe_mean(d.get("local_id_y_means")) for d in metrics_data]
    mean_id_min = [_safe_mean(d.get("local_id_min_means")) for d in metrics_data]
    rank_reference_labels = {
        str(d.get("rank_reference_label", _metric_value_short_label(d.get("labels", _FALLBACK_LABELS))))
        for d in metrics_data
    }
    rank_reference_label = (
        next(iter(rank_reference_labels))
        if len(rank_reference_labels) == 1
        else "rank reference"
    )
    mean_rank = [
        _safe_mean(d.get("rank_reference_means", d.get("rank_means")))
        for d in metrics_data
    ]
    mean_rank_gap = [
        _safe_mean(d.get("rank_minus_id_min_means")) for d in metrics_data
    ]
    mean_system_gap = [
        _safe_mean(d.get("system_minus_id_min_means")) for d in metrics_data
    ]

    ax = axes[0]
    width = 0.25
    ax.bar(x - width, mean_id_x, width=width, color="steelblue", alpha=0.8, label="mean(ID_x)")
    ax.bar(x, mean_id_y, width=width, color="coral", alpha=0.8, label="mean(ID_y)")
    ax.bar(
        x + width,
        mean_id_min,
        width=width,
        color="seagreen",
        alpha=0.8,
        label="mean(min(ID_x, ID_y))",
    )
    ax.plot(x, mean_rank, "D--", color="darkred", markersize=6, label=f"mean({rank_reference_label})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Среднее значение")
    ax.set_title(f"Локальная размерность и {rank_reference_label}")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    bars = ax.bar(x, mean_rank_gap, color="mediumpurple", alpha=0.8)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"mean({rank_reference_label} - min(ID))")
    ax.set_title(f"Насколько {rank_reference_label} превышает локальную размерность")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, mean_rank_gap):
        if np.isfinite(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.01 * max(1.0, abs(bar.get_height()))),
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax = axes[2]
    bars = ax.bar(x, mean_system_gap, color="darkseagreen", alpha=0.8)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean(N_sys - min(ID))")
    ax.set_title("Определённость системы относительно локальной размерности")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, mean_system_gap):
        if np.isfinite(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.01 * max(1.0, abs(bar.get_height()))),
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    plt.tight_layout()
    fpath = os.path.join(plots_dir, "summary_local_id_diagnostics.png")
    saved_paths = _save_figure_variants(fig, fpath, plots_exts, dpi=150)
    plt.close(fig)
    for save_path in saved_paths:
        print(f"  Сохранён сводный local-ID график: {save_path}")


def _collect_metric_data(
    artifacts: Dict[str, np.ndarray],
    local_id_artifacts: Optional[Dict[str, np.ndarray]],
    directions: List[Tuple[str, str]],
    metric_name: str,
    both_deg_stats: Dict,
    threshold: float,
) -> Dict:
    """
    Собирает сводные данные по одной метрике для последующей передачи в _plot_summary.
    """
    rank_means, rank_stds = [], []
    rank_reference_means = []
    rank_reference_label = _hard_rank_short_label()
    res_means, res_stds = [], []
    system_point_means, determined_margin_means, overdetermined_fracs = [], [], []
    all_metric_values = []
    local_id_x_means, local_id_y_means = [], []
    local_id_min_means, rank_minus_id_min_means = [], []
    system_minus_id_min_means = []
    residuals_normalized_flags = []
    relative_residual_flags = []

    # Для спектра: собираем все сингулярные значения по всем центрам и всем направлениям.
    all_sv_lists: List[np.ndarray] = []

    for mi, mj in directions:
        sv, res, ranks = _get_direction_data(artifacts, mi, mj)
        rank_ref, rank_reference_label = _get_direction_rank_reference(
            artifacts, mi, mj, ranks
        )
        extra = _get_direction_extra_data(artifacts, mi, mj)
        local_id = _get_direction_local_id_data(local_id_artifacts or {}, mi, mj)
        if _direction_uses_relative_residuals(artifacts, mi, mj):
            work_res = np.asarray(res, dtype=np.float64)
            is_normalized = True
            is_relative = True
        else:
            work_res, is_normalized = _normalize_residuals(res, extra)
            is_relative = False
        s = _compute_direction_stats(sv, work_res, ranks, threshold=threshold)
        det_stats = _compute_determinedness_stats(rank_ref, extra)
        lid_stats = _compute_local_id_stats(rank_ref, extra, local_id)
        all_metric_values.extend(np.asarray(ranks, dtype=np.float64).reshape(-1).tolist())
        rank_means.append(s["rank_mean"])
        rank_stds.append(s["rank_std"])
        rank_reference_means.append(_safe_mean(rank_ref))
        res_means.append(s["residual_mean"])
        res_stds.append(s["residual_std"])
        system_point_means.append(det_stats["point_count_mean"])
        determined_margin_means.append(det_stats["margin_mean"])
        overdetermined_fracs.append(det_stats["overdetermined_frac"])
        local_id_x_means.append(lid_stats["id_x_mean"])
        local_id_y_means.append(lid_stats["id_y_mean"])
        local_id_min_means.append(lid_stats["id_min_mean"])
        rank_minus_id_min_means.append(lid_stats["rank_minus_id_min_mean"])
        system_minus_id_min_means.append(lid_stats["system_minus_id_min_mean"])
        residuals_normalized_flags.append(is_normalized)
        relative_residual_flags.append(is_relative)
        # Накапливаем сингулярные значения каждого центра.
        for sv_center in sv:
            if sv_center is not None and len(sv_center) > 0:
                all_sv_lists.append(np.array(sv_center, dtype=np.float32))

    # Агрегируем спектр: медиана и std по позиции сингулярного значения.
    # Обрезаем все векторы до минимальной длины чтобы выровнять размерности.
    if all_sv_lists:
        min_len = min(len(s) for s in all_sv_lists)
        sv_matrix = np.stack([s[:min_len] for s in all_sv_lists], axis=0)
        spectrum_median = np.median(sv_matrix, axis=0)
        spectrum_std = np.std(sv_matrix, axis=0)
    else:
        spectrum_median = np.array([])
        spectrum_std = np.array([])

    _, _, _ranks_first = _get_direction_data(
        artifacts, directions[0][0], directions[0][1]
    )
    n_centers = len(_ranks_first)

    return {
        "metric_name": metric_name,
        "rank_means": np.array(rank_means),
        "rank_stds": np.array(rank_stds),
        "rank_reference_means": np.array(rank_reference_means),
        "rank_reference_label": rank_reference_label,
        "all_metric_values": np.array(all_metric_values, dtype=np.float64),
        "res_means": np.array(res_means),
        "res_stds": np.array(res_stds),
        "system_point_means": np.array(system_point_means),
        "determined_margin_means": np.array(determined_margin_means),
        "overdetermined_fracs": np.array(overdetermined_fracs),
        "local_id_x_means": np.array(local_id_x_means),
        "local_id_y_means": np.array(local_id_y_means),
        "local_id_min_means": np.array(local_id_min_means),
        "rank_minus_id_min_means": np.array(rank_minus_id_min_means),
        "system_minus_id_min_means": np.array(system_minus_id_min_means),
        "residuals_normalized": bool(all(residuals_normalized_flags))
        if residuals_normalized_flags
        else False,
        "uses_relative_residuals": bool(all(relative_residual_flags))
        if relative_residual_flags
        else False,
        "frac_both_degenerate": both_deg_stats["frac_both_degenerate_overall"],
        "n_centers": n_centers,
        "n_directions": len(directions),
        "spectrum_median": spectrum_median,  # (min_len,) - медиана сингулярных значений
        "spectrum_std": spectrum_std,  # (min_len,) - std сингулярных значений
        "labels": _load_diagnostics_labels(artifacts),  # подписи из diagnostics_meta_json
    }


def _load_metric_spec(metric_name: str) -> str:
    """
    Читает параметры метрики из configs/metric_configs.py и возвращает
    читаемую строку с ключевыми параметрами для отображения в заголовке.
    Если метрика не найдена — возвращает пустую строку.
    """
    try:
        cfgs = get_embedding_metric_configs()
        if metric_name not in cfgs:
            return ""
        meta = cfgs[metric_name].get("meta", {})
        parts = []

        if "k_list" in meta:
            parts.append(f"k_list={meta['k_list']}")
        if "aggregator" in meta:
            parts.append(f"agg={meta['aggregator']}")
        if "k" in meta and "k_list" not in meta:
            parts.append(f"k={meta['k']}")
        if "eps_percentile" in meta:
            parts.append(f"eps_percentile={meta['eps_percentile']}")
        if "sigma_percentile" in meta:
            parts.append(f"sigma_percentile={meta['sigma_percentile']}")
        if "eps_scale" in meta:
            parts.append(f"eps={meta['eps_scale']}*sigma")
        if "weighting" in meta:
            parts.append(f"weighting={meta['weighting']}")
        if "solver" in meta:
            parts.append(f"solver={meta['solver']}")
        if "n_centers" in meta:
            parts.append(f"n_centers={meta['n_centers']}")
        if "rank_aggregation" in meta:
            parts.append(f"rank_agg={meta['rank_aggregation']}")

        return ", ".join(parts)
    except Exception:
        return ""


def _is_local_id_diff_metric(metric_name: str) -> bool:
    try:
        cfgs = get_embedding_metric_configs()
        if metric_name in cfgs:
            meta = cfgs[metric_name].get("meta", {})
            return str(meta.get("family", "")) == "local_id_diff"
    except Exception:
        pass
    return short_metric_name(metric_name).startswith("id_diff_")


# ============================================================
# 7) Основной запуск
# ============================================================


def _run_single_metric(
    artifacts_path: str,
    out_dir: str,
    threshold: float,
    model_a: str = "",
    model_b: str = "",
    collect_for_summary: bool = False,
    plots_exts: Iterable[str] = ("png",),
    downstream_scores: Optional[Dict[str, float]] = None,
) -> Optional[Dict]:
    """
    Запускает полную диагностику для одного файла артефактов.
    Если collect_for_summary=True — возвращает сводные данные для _plot_summary,
    иначе возвращает None.
    """
    plots_dir, reports_dir = _make_out_dirs(out_dir)
    metric_name = os.path.basename(artifacts_path).replace("_artifacts.npz", "")
    metric_spec_str = _load_metric_spec(metric_name)

    print(f"\nМетрика: {metric_name}")
    print(f"Загрузка артефактов: {artifacts_path}")

    if _is_local_id_diff_metric(metric_name):
        warn = (
            "Графики диагностики для local_id_diff временно отключены: "
            "текущий diagnose-скрипт ориентирован на rank/residual-сравнения map-метрик "
            "и не подходит для графиков сравнения размерностей новой метрики."
        )
        print(f"  [WARN] {warn}")
        os.makedirs(reports_dir, exist_ok=True)
        stub_path = os.path.join(reports_dir, f"{metric_name}_diagnostics_skipped.txt")
        with open(stub_path, "w", encoding="utf-8") as f:
            f.write(warn + "\n")
        print(f"  [INFO] Сохранена заглушка: {stub_path}")
        return None

    artifacts = _load_artifacts(artifacts_path)
    local_id_artifacts = _load_optional_local_id_artifacts(artifacts_path)

    # Читаем подписи из артефактов один раз и передаём во все функции рисования.
    # Для старых артефактов без diagnostics_meta_json используются значения из _FALLBACK_LABELS.
    labels = _load_diagnostics_labels(artifacts)

    directions = _list_directions(artifacts)
    if not directions:
        print(
            f"  [WARN] Не найдено ни одного направления в {artifacts_path}, пропускаем."
        )
        return None

    print(f"Пар моделей: {len(directions)}")

    both_deg_stats = _compute_both_degenerate_fraction(
        artifacts, directions, threshold=threshold
    )
    print(
        f"Одновременно вырожденных центров: "
        f"{both_deg_stats['total_both_degenerate']} / {both_deg_stats['total_centers']} "
        f"({both_deg_stats['frac_both_degenerate_overall']:.2%})"
    )

    _plot_aggregated(
        artifacts,
        directions,
        both_deg_stats,
        plots_dir,
        metric_name,
        threshold=threshold,
        metric_spec_str=metric_spec_str,
        plots_exts=plots_exts,
        labels=labels,
        downstream_scores=downstream_scores,
    )
    _save_report(
        directions,
        artifacts,
        local_id_artifacts,
        both_deg_stats,
        reports_dir,
        metric_name,
    )

    if model_a and model_b:
        missing = []
        for a, b in [(model_a, model_b), (model_b, model_a)]:
            if (a, b) not in directions:
                missing.append(f"{a} → {b}")
        if missing:
            print(
                f"  [WARN] Не найдены направления: {missing}. Детальный анализ пропущен."
            )
        else:
            _plot_single_pair(
                artifacts,
                local_id_artifacts,
                model_a,
                model_b,
                plots_dir,
                metric_name,
                threshold=threshold,
                plots_exts=plots_exts,
                labels=labels,
            )

    if collect_for_summary:
        return _collect_metric_data(
            artifacts,
            local_id_artifacts,
            directions,
            metric_name,
            both_deg_stats,
            threshold,
        )
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Диагностика метода локальных линейных отображений между эмбеддингами."
    )
    # Режим 1: сводный — папка со всеми артефактами.
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default="",
        help=(
            "Папка с файлами артефактов (*_artifacts.npz). "
            "Если задано — строит сводный график по всем метрикам сразу "
            "и агрегированный по каждой. Взаимоисключающий с --artifacts_path."
        ),
    )
    # Режим 2/3: один файл артефактов.
    parser.add_argument(
        "--artifacts_path",
        type=str,
        default="",
        help="Путь к одному файлу артефактов {metric_name}_artifacts.npz.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="diagnostics",
        help="Куда сохранять графики и отчёты.",
    )
    parser.add_argument(
        "--plots_ext",
        type=str,
        default="png",
        help="Одно или несколько расширений графиков через запятую, например png или svg,png.",
    )
    parser.add_argument(
        "--model_a",
        type=str,
        default="",
        help="Первая модель для детального анализа одной пары (только с --artifacts_path).",
    )
    parser.add_argument(
        "--model_b",
        type=str,
        default="",
        help="Вторая модель для детального анализа одной пары (только с --artifacts_path).",
    )
    parser.add_argument(
        "--degenerate_threshold",
        type=float,
        default=DEGENERATE_MAP_THRESHOLD_DEFAULT,
        help=(
            "Порог для определения вырожденного отображения: "
            "отображение считается вырожденным, если все его сингулярные значения < порога. "
            f"По умолчанию: {DEGENERATE_MAP_THRESHOLD_DEFAULT:.0e}."
        ),
    )
    args = parser.parse_args()
    args.plots_ext = parse_plots_exts(args.plots_ext)
    downstream_scores = _load_downstream_scores(
        os.path.join("data", "downstream", "cifar10_linear_probe.json"),
        task_name="cifar10_linear_probe",
    )

    if not args.artifacts_dir and not args.artifacts_path:
        parser.error("Нужно указать либо --artifacts_dir, либо --artifacts_path.")
    if args.artifacts_dir and args.artifacts_path:
        parser.error("--artifacts_dir и --artifacts_path взаимоисключающие.")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Порог вырожденности: {args.degenerate_threshold:.2e}")

    # ============================================================
    # Режим 1: сводный — обходим все *_artifacts.npz в папке
    # ============================================================
    if args.artifacts_dir:
        artifact_files = sorted(
            [
                os.path.join(args.artifacts_dir, fn)
                for fn in os.listdir(args.artifacts_dir)
                if fn.endswith("_artifacts.npz")
                and not fn.endswith("_local_id_artifacts.npz")
            ]
        )
        if not artifact_files:
            raise RuntimeError(
                f"В папке {args.artifacts_dir} не найдено файлов *_artifacts.npz"
            )

        print(f"Найдено файлов артефактов: {len(artifact_files)}")
        for p in artifact_files:
            print(f"  {os.path.basename(p)}")

        # Для каждой метрики — агрегированный график + сбор данных для сводного.
        metrics_data = []
        for ap in artifact_files:
            data = _run_single_metric(
                artifacts_path=ap,
                out_dir=args.out_dir,
                threshold=args.degenerate_threshold,
                collect_for_summary=True,
                plots_exts=args.plots_ext,
                downstream_scores=downstream_scores,
            )
            if data is not None:
                metrics_data.append(data)

        # Сводный график по всем метрикам.
        print("\nПостроение сводного графика...")
        plots_dir, _ = _make_out_dirs(args.out_dir)
        _plot_summary(
            metrics_data,
            plots_dir,
            threshold=args.degenerate_threshold,
            plots_exts=args.plots_ext,
        )
        _plot_summary_local_id(
            metrics_data,
            plots_dir,
            plots_exts=args.plots_ext,
        )

    # ============================================================
    # Режим 2/3: один файл — агрегированный или детальный
    # ============================================================
    else:
        _run_single_metric(
            artifacts_path=args.artifacts_path,
            out_dir=args.out_dir,
            threshold=args.degenerate_threshold,
            model_a=args.model_a,
            model_b=args.model_b,
            collect_for_summary=False,
            plots_exts=args.plots_ext,
            downstream_scores=downstream_scores,
        )

    print("\nГотово.")


if __name__ == "__main__":
    main()
