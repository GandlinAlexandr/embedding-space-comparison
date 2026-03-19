"""
run_diagnose_local_map.py

Диагностика метода локальных линейных отображений между эмбеддингами.

Читает артефакты, сохранённые run_compute_embedding_metrics.py
({metric_name}_artifacts.npz), и строит диагностические графики и статистику.

Проверяемые гипотезы (по заданию):
  1. Ошибка решения линейного уравнения (residual):
       насколько хорошо линейное приближение работает в каждой точке.
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
      --artifacts_path metric_matrices/local_map_rank_linear_knn_k10_artifacts.npz \\
      --out_dir diagnostics/

  # Детально по одной паре:
  python -m scripts.run_diagnose_local_map \\
      --artifacts_path metric_matrices/local_map_rank_linear_knn_k10_artifacts.npz \\
      --model_a resnet50 --model_b vit_b16 \\
      --out_dir diagnostics/
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ВАЖНО: запускаем как модуль: python -m scripts.run_diagnose_local_map
from configs.metric_configs import short_metric_name

# ============================================================
# 0) Вспомогательные функции вывода
# ============================================================


def _make_out_dirs(out_dir: str):
    """
    Создаёт две поддиректории внутри out_dir:
      plots/   — все графики (.png)
      reports/ — текстовые и JSON-отчёты (.txt, .json)
    Возвращает (plots_dir, reports_dir).
    """
    plots_dir = os.path.join(out_dir, "plots")
    reports_dir = os.path.join(out_dir, "reports")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    return plots_dir, reports_dir


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
    model_j = direction[idx + 4 :]
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
    """
    prefix = f"{model_i}_to_{model_j}"
    sv = artifacts[f"{prefix}/singular_values"]
    res = artifacts[f"{prefix}/residuals"]
    ranks = artifacts[f"{prefix}/ranks"]
    return sv, res, ranks


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
    """
    # Ранги
    ranks = ranks.astype(np.float32)
    rank_mean = float(np.mean(ranks))
    rank_std = float(np.std(ranks))
    rank_min = int(np.min(ranks))
    rank_max = int(np.max(ranks))

    # Residuals
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
    model_i: str,
    model_j: str,
    plots_dir: str,
    metric_name: str,
    threshold: float = DEGENERATE_MAP_THRESHOLD_DEFAULT,
) -> None:
    """
    Строит детальные графики для пары (model_i, model_j):
      - гистограммы рангов для обоих направлений
      - распределение residuals для обоих направлений
      - сингулярные значения (медиана ± std по центрам)
    """
    sv_ij, res_ij, ranks_ij = _get_direction_data(artifacts, model_i, model_j)
    sv_ji, res_ji, ranks_ji = _get_direction_data(artifacts, model_j, model_i)

    label_ij = f"{model_i} → {model_j}"
    label_ji = f"{model_j} → {model_i}"

    n_centers = len(ranks_ij)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Диагностика локального отображения\n"
        f"Метрика: {metric_name} | Пара: {model_i} / {model_j} | Центров: {n_centers} | Порог вырожденности: {threshold:.2e}",
        fontsize=12,
    )

    # --- 1. Гистограмма рангов ---
    ax = axes[0, 0]
    all_ranks = np.concatenate([ranks_ij, ranks_ji])
    bins = np.arange(all_ranks.min(), all_ranks.max() + 2) - 0.5
    ax.hist(ranks_ij, bins=bins, alpha=0.6, label=label_ij, color="steelblue")
    ax.hist(ranks_ji, bins=bins, alpha=0.6, label=label_ji, color="coral")
    ax.set_xlabel("Ранг отображения M")
    ax.set_ylabel("Количество центров")
    ax.set_title("Гистограмма рангов")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 2. Распределение residuals ---
    ax = axes[0, 1]
    ax.hist(res_ij, bins=30, alpha=0.6, label=label_ij, color="steelblue")
    ax.hist(res_ji, bins=30, alpha=0.6, label=label_ji, color="coral")
    ax.set_xlabel("Residual ||Xc @ M - Yc||_F")
    ax.set_ylabel("Количество центров")
    ax.set_title("Распределение ошибок решения")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 3. Residuals: boxplot ---
    ax = axes[0, 2]
    ax.boxplot(
        [res_ij, res_ji],
        labels=[label_ij, label_ji],
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
    )
    ax.set_ylabel("Residual")
    ax.set_title("Ошибки решения: boxplot")
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
    stats_ij = _compute_direction_stats(sv_ij, res_ij, ranks_ij, threshold=threshold)
    stats_ji = _compute_direction_stats(sv_ji, res_ji, ranks_ji, threshold=threshold)
    table_data = [
        ["", label_ij[:20], label_ji[:20]],
        ["Центров", stats_ij["n_centers"], stats_ji["n_centers"]],
        [
            "Ранг (среднее)",
            f"{stats_ij['rank_mean']:.2f}",
            f"{stats_ji['rank_mean']:.2f}",
        ],
        ["Ранг (std)", f"{stats_ij['rank_std']:.2f}", f"{stats_ji['rank_std']:.2f}"],
        [
            "Ранг (min/max)",
            f"{stats_ij['rank_min']}/{stats_ij['rank_max']}",
            f"{stats_ji['rank_min']}/{stats_ji['rank_max']}",
        ],
        [
            "Residual (mean)",
            f"{stats_ij['residual_mean']:.4f}",
            f"{stats_ji['residual_mean']:.4f}",
        ],
        [
            "Residual (std)",
            f"{stats_ij['residual_std']:.4f}",
            f"{stats_ji['residual_std']:.4f}",
        ],
        [
            "Выр-х центров",
            f"{stats_ij['n_degenerate']} ({stats_ij['frac_degenerate']:.1%})",
            f"{stats_ji['n_degenerate']} ({stats_ji['frac_degenerate']:.1%})",
        ],
    ]
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
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Сохранён график: {fpath}")


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
    ax.set_ylabel("Значение")
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
) -> None:
    """
    Строит агрегированные графики по всем парам:
      - общая гистограмма рангов
      - общее распределение residuals
      - доля вырожденных отображений по парам
      - доля одновременно вырожденных центров (оба направления) по парам
    """
    all_ranks = []
    all_residuals = []
    frac_deg_per_direction = []
    direction_labels = []

    for mi, mj in directions:
        sv, res, ranks = _get_direction_data(artifacts, mi, mj)
        stats = _compute_direction_stats(sv, res, ranks, threshold=threshold)
        all_ranks.extend(ranks.tolist())
        all_residuals.extend(res.tolist())
        frac_deg_per_direction.append(stats["frac_degenerate"])
        direction_labels.append(f"{mi[:10]}→{mj[:10]}")

    all_ranks = np.array(all_ranks)
    all_residuals = np.array(all_residuals)

    # n_centers — реальное количество центров первого направления, читается из артефактов.
    _, _, _ranks_first = _get_direction_data(
        artifacts, directions[0][0], directions[0][1]
    )
    n_centers_per_direction = len(_ranks_first)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Агрегированная диагностика локального отображения\n"
        f"Метрика: {metric_name} | Центров на направление: {n_centers_per_direction} | Направлений: {len(directions)} | Порог вырожденности: {threshold:.2e}",
        fontsize=12,
    )

    # --- 1. Гистограмма рангов по всем парам и центрам ---
    ax = axes[0, 0]
    bins = np.arange(all_ranks.min(), all_ranks.max() + 2) - 0.5
    ax.hist(all_ranks, bins=bins, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(
        np.mean(all_ranks),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"среднее={np.mean(all_ranks):.2f}",
    )
    ax.set_xlabel("Ранг отображения M")
    ax.set_ylabel("Количество центров (все пары)")
    ax.set_title(f"Гистограмма рангов\nstd={np.std(all_ranks):.3f}")
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
        label=f"среднее={np.mean(all_residuals):.4f}",
    )
    ax.axvline(
        np.median(all_residuals),
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"медиана={np.median(all_residuals):.4f}",
    )
    ax.set_xlabel("Residual ||Xc @ M - Yc||_F")
    ax.set_ylabel("Количество центров (все пары)")
    ax.set_title("Распределение ошибок решения")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 3. Доля вырожденных отображений по направлениям ---
    ax = axes[1, 0]
    x = np.arange(len(frac_deg_per_direction))
    bars = ax.bar(x, frac_deg_per_direction, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(direction_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Доля вырожденных центров")
    ax.set_title("Вырожденность по направлениям")
    ax.set_ylim(0, max(max(frac_deg_per_direction) * 1.2, 0.05))
    ax.grid(True, alpha=0.3, axis="y")
    # Подписываем значения на барах
    for bar, val in zip(bars, frac_deg_per_direction):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.1%}",
                ha="center",
                va="bottom",
                fontsize=6,
            )

    # --- 4. Доля одновременно вырожденных центров (оба направления) по парам ---
    ax = axes[1, 1]
    per_pair = both_deg_stats["per_pair"]
    if per_pair:
        pair_labels = [f"{r['model_i'][:8]}/{r['model_j'][:8]}" for r in per_pair]
        pair_fracs = [r["frac_both_degenerate"] for r in per_pair]
        x2 = np.arange(len(pair_labels))
        bars2 = ax.bar(x2, pair_fracs, color="coral", alpha=0.8)
        ax.set_xticks(x2)
        ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Доля центров, где оба вырождены")
        overall = both_deg_stats["frac_both_degenerate_overall"]
        ax.set_title(
            f"Одновременная вырожденность (X→Y и Y→X)\n"
            f"Итого: {both_deg_stats['total_both_degenerate']} / {both_deg_stats['total_centers']} "
            f"({overall:.1%})"
        )
        ax.set_ylim(0, max(max(pair_fracs) * 1.2, 0.05))
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars2, pair_fracs):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )
    else:
        ax.text(
            0.5,
            0.5,
            "Недостаточно пар\nдля анализа вырожденности",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_title("Одновременная вырожденность")

    plt.tight_layout()
    fname = f"{metric_name}_aggregated_diagnostics.png"
    fpath = os.path.join(plots_dir, fname)
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Сохранён агрегированный график: {fpath}")


# ============================================================
# 5) Сохранение текстового отчёта
# ============================================================


def _save_report(
    directions: List[Tuple[str, str]],
    artifacts: Dict[str, np.ndarray],
    both_deg_stats: Dict,
    reports_dir: str,
    metric_name: str,
) -> None:
    """Сохраняет текстовый отчёт с ключевыми числами диагностики."""
    lines = []
    lines.append(f"Диагностика метрики: {metric_name}")
    lines.append("=" * 60)
    lines.append(f"Всего направлений: {len(directions)}")
    lines.append("")

    lines.append("--- Статистика по направлениям ---")
    header = f"{'Направление':<35} {'Ранг(mean)':<12} {'Ранг(std)':<12} {'Res(mean)':<12} {'Res(std)':<12} {'Выр-х,%':<10}"
    lines.append(header)
    lines.append("-" * len(header))

    all_rank_stds = []
    for mi, mj in sorted(directions):
        sv, res, ranks = _get_direction_data(artifacts, mi, mj)
        s = _compute_direction_stats(sv, res, ranks)
        all_rank_stds.append(s["rank_std"])
        direction_str = f"{mi}→{mj}"
        lines.append(
            f"{direction_str:<35} {s['rank_mean']:<12.3f} {s['rank_std']:<12.3f} "
            f"{s['residual_mean']:<12.4f} {s['residual_std']:<12.4f} "
            f"{s['frac_degenerate']:<10.1%}"
        )

    lines.append("")
    lines.append(
        f"Средняя std ранга по всем направлениям: {np.mean(all_rank_stds):.4f}"
    )
    lines.append("")

    lines.append("--- Одновременная вырожденность (X→Y и Y→X) ---")
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
) -> None:
    """
    Строит сводный график сравнения всех метрик по трём вопросам руководителя:
      1. Ошибка решения (residual mean ± std по всем центрам и парам)
      2. Стабильность ранга (std ранга по всем центрам и парам)
      3. Доля одновременно вырожденных центров (оба направления)

    metrics_data — список словарей, по одному на метрику:
      {
        "metric_name": str,
        "rank_means":  np.ndarray,   # среднее ранга по каждому направлению
        "rank_stds":   np.ndarray,   # std ранга по каждому направлению
        "res_means":   np.ndarray,   # среднее residual по каждому направлению
        "res_stds":    np.ndarray,   # std residual по каждому направлению
        "frac_both_degenerate": float,  # итоговая доля по всем парам
        "n_centers":   int,
        "n_directions": int,
      }
    """
    if not metrics_data:
        print("  [WARN] Нет данных для сводного графика.")
        return

    labels = [d["metric_name"] for d in metrics_data]
    # Укорачиваем длинные имена для читаемости осей.
    short_labels = [short_metric_name(l) for l in labels]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(4, 1, figsize=(max(12, len(labels) * 1.2), 18))
    n_centers = metrics_data[0]["n_centers"]
    fig.suptitle(
        f"Сводная диагностика локальных отображений\n"
        f"Центров на направление: {n_centers} | Метрик: {len(labels)} | "
        f"Порог вырожденности: {threshold:.0e}",
        fontsize=13,
    )

    # --- 1. Стабильность ранга: среднее std по направлениям ---
    ax = axes[0]
    mean_rank_stds = [float(np.mean(d["rank_stds"])) for d in metrics_data]
    mean_rank_means = [float(np.mean(d["rank_means"])) for d in metrics_data]
    bars = ax.bar(
        x, mean_rank_stds, color="steelblue", alpha=0.8, label="mean(std ранга)"
    )
    # Поверх баров — среднее значение ранга как точки.
    ax2 = ax.twinx()
    ax2.plot(
        x, mean_rank_means, "D--", color="darkred", markersize=6, label="mean(ранг)"
    )
    ax2.set_ylabel("Среднее значение ранга", color="darkred")
    ax2.tick_params(axis="y", labelcolor="darkred")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Среднее std ранга по направлениям")
    ax.set_title("Стабильность ранга отображения M\n(чем меньше std — тем стабильнее)")
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

    # --- 2. Ошибка решения: среднее residual по направлениям ---
    ax = axes[1]
    mean_res = [float(np.mean(d["res_means"])) for d in metrics_data]
    std_res = [float(np.mean(d["res_stds"])) for d in metrics_data]
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
    ax.set_ylabel("Среднее residual ||Xc @ M - Yc||_F")
    ax.set_title(
        "Ошибка решения линейного уравнения\n(чем меньше — тем лучше линейное приближение)"
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

    # --- 3. Одновременная вырожденность ---
    ax = axes[2]
    frac_both_deg = [d["frac_both_degenerate"] for d in metrics_data]
    bars = ax.bar(x, frac_both_deg, color="mediumpurple", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Доля центров, где оба направления вырождены")
    ax.set_title(
        "Одновременная вырожденность X->Y и Y->X\n"
        "(по гипотезе должно быть близко к 0)"
    )
    ax.set_ylim(0, max(max(frac_both_deg) * 1.3, 0.05))
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, frac_both_deg):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(frac_both_deg) * 0.01 + 0.001,
            f"{val:.2%}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # --- 4. Спектр сингулярных значений по всем метрикам на одном поле ---
    ax = axes[3]
    # Цветовая палитра для метрик.
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
        "Спектр сингулярных значений матрицы M\n(медиана ± std по всем центрам и направлениям)"
    )
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fpath = os.path.join(plots_dir, "summary_diagnostics.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Сохранён сводный график: {fpath}")


def _collect_metric_data(
    artifacts: Dict[str, np.ndarray],
    directions: List[Tuple[str, str]],
    metric_name: str,
    both_deg_stats: Dict,
    threshold: float,
) -> Dict:
    """
    Собирает сводные данные по одной метрике для последующей передачи в _plot_summary.
    """
    rank_means, rank_stds = [], []
    res_means, res_stds = [], []

    # Для спектра: собираем все сингулярные значения по всем центрам и всем направлениям.
    all_sv_lists: List[np.ndarray] = []

    for mi, mj in directions:
        sv, res, ranks = _get_direction_data(artifacts, mi, mj)
        s = _compute_direction_stats(sv, res, ranks, threshold=threshold)
        rank_means.append(s["rank_mean"])
        rank_stds.append(s["rank_std"])
        res_means.append(s["residual_mean"])
        res_stds.append(s["residual_std"])
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
        "res_means": np.array(res_means),
        "res_stds": np.array(res_stds),
        "frac_both_degenerate": both_deg_stats["frac_both_degenerate_overall"],
        "n_centers": n_centers,
        "n_directions": len(directions),
        "spectrum_median": spectrum_median,  # (min_len,) - медиана сингулярных значений
        "spectrum_std": spectrum_std,  # (min_len,) - std сингулярных значений
    }


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
) -> Optional[Dict]:
    """
    Запускает полную диагностику для одного файла артефактов.
    Если collect_for_summary=True — возвращает сводные данные для _plot_summary,
    иначе возвращает None.
    """
    plots_dir, reports_dir = _make_out_dirs(out_dir)
    metric_name = os.path.basename(artifacts_path).replace("_artifacts.npz", "")

    print(f"\nМетрика: {metric_name}")
    print(f"Загрузка артефактов: {artifacts_path}")
    artifacts = _load_artifacts(artifacts_path)

    directions = _list_directions(artifacts)
    if not directions:
        print(
            f"  [WARN] Не найдено ни одного направления в {artifacts_path}, пропускаем."
        )
        return None

    print(f"Направлений: {len(directions)}")

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
    )
    _save_report(directions, artifacts, both_deg_stats, reports_dir, metric_name)

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
                artifacts, model_a, model_b, plots_dir, metric_name, threshold=threshold
            )

    if collect_for_summary:
        return _collect_metric_data(
            artifacts, directions, metric_name, both_deg_stats, threshold
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
            )
            if data is not None:
                metrics_data.append(data)

        # Сводный график по всем метрикам.
        print("\nПостроение сводного графика...")
        plots_dir, _ = _make_out_dirs(args.out_dir)
        _plot_summary(metrics_data, plots_dir, threshold=args.degenerate_threshold)

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
        )

    print("\nГотово.")


if __name__ == "__main__":
    main()
