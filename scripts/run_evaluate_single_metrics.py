from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr


# ============================================================
# Вспомогательные функции
# ============================================================


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_float(x: Any) -> float:
    value = float(x)
    if not np.isfinite(value):
        raise ValueError(f"Ожидалось конечное число, получено {value}")
    return value


def mean_ignore_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def compute_correlations(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if x.shape != y.shape:
        raise ValueError(f"Размеры не совпадают: x.shape={x.shape}, y.shape={y.shape}")

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 2:
        return {
            "n_pairs_used": int(x.size),
            "spearman": float("nan"),
            "pearson": float("nan"),
            "kendall": float("nan"),
        }

    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {
            "n_pairs_used": int(x.size),
            "spearman": float("nan"),
            "pearson": float("nan"),
            "kendall": float("nan"),
        }

    return {
        "n_pairs_used": int(x.size),
        "spearman": float(spearmanr(x, y).statistic),
        "pearson": float(pearsonr(x, y).statistic),
        "kendall": float(kendalltau(x, y).statistic),
    }


# ============================================================
# Загрузка single metrics
# Формат: <single_metrics_dir>/<metric_name>/<model>.json
# ============================================================


def discover_metric_dirs(single_metrics_dir: Path) -> list[Path]:
    return sorted([p for p in single_metrics_dir.iterdir() if p.is_dir()])


def load_single_metric_scores(metric_dir: Path) -> tuple[str, dict[str, float], bool]:
    metric_name = metric_dir.name
    scores: dict[str, float] = {}
    higher_is_better: bool | None = None

    json_files = sorted(metric_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"В папке метрики нет json-файлов: {metric_dir}")

    for path in json_files:
        payload = load_json(path)

        if not isinstance(payload, dict):
            continue
        if "model_name" not in payload:
            continue

        model_name = str(payload["model_name"])
        value = safe_float(payload["value"])
        hib = bool(payload["higher_is_better"])

        if higher_is_better is None:
            higher_is_better = hib
        elif higher_is_better != hib:
            raise ValueError(
                f"Для метрики {metric_name} найдено несогласованное поле higher_is_better"
            )

        if model_name in scores:
            raise ValueError(
                f"Дублирующееся имя модели {model_name} в метрике {metric_name}"
            )

        scores[model_name] = value

    if not scores:
        raise RuntimeError(f"В метрике {metric_name} нет валидных json-файлов")

    if higher_is_better is None:
        raise RuntimeError(f"Не удалось определить higher_is_better для {metric_name}")

    return metric_name, scores, higher_is_better


# ============================================================
# Downstream
# Формат проекта: model -> task -> score
# ============================================================


def load_downstream_table(path: Path) -> dict[str, dict[str, float]]:
    obj = load_json(path)

    if not isinstance(obj, dict):
        raise ValueError(
            "Downstream json должен быть словарём вида model -> task -> score"
        )

    table: dict[str, dict[str, float]] = {}

    for model_name, task_scores in obj.items():
        if not isinstance(task_scores, dict):
            raise ValueError(
                "Downstream json должен быть словарём вида model -> task -> score"
            )

        row: dict[str, float] = {}
        for task_name, score in task_scores.items():
            row[str(task_name)] = safe_float(score)

        table[str(model_name)] = row

    return table


# ============================================================
# Построение пар
# ============================================================


def align_scores(scores: dict[str, float], higher_is_better: bool) -> dict[str, float]:
    if higher_is_better:
        return dict(scores)
    return {k: -v for k, v in scores.items()}


def intersect_models(
    single_scores: dict[str, float],
    downstream: dict[str, dict[str, float]],
) -> list[str]:
    common = sorted(set(single_scores.keys()) & set(downstream.keys()))
    if len(common) < 2:
        raise ValueError(
            "Слишком мало общих моделей между single-metrics и downstream: " f"{common}"
        )
    return common


def intersect_tasks(
    common_models: list[str],
    downstream: dict[str, dict[str, float]],
) -> list[str]:
    if not common_models:
        return []

    task_sets = [set(downstream[m].keys()) for m in common_models]
    common_tasks = sorted(set.intersection(*task_sets))
    if not common_tasks:
        raise ValueError("Не найдено общих downstream-задач для всех общих моделей")
    return common_tasks


def build_pairs_for_task(
    metric_name: str,
    aligned_scores: dict[str, float],
    downstream: dict[str, dict[str, float]],
    common_models: list[str],
    task_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model_i, model_j in combinations(common_models, 2):
        metric_i = aligned_scores[model_i]
        metric_j = aligned_scores[model_j]

        score_i = downstream[model_i][task_name]
        score_j = downstream[model_j][task_name]

        delta_metric_signed = metric_i - metric_j
        delta_metric_abs = abs(delta_metric_signed)

        # ВАЖНО:
        # порядок вычитания должен совпадать с delta_metric_signed,
        # иначе signed-корреляция инвертируется по знаку.
        delta_score_signed = score_i - score_j
        delta_score_abs = abs(delta_score_signed)

        rows.append(
            {
                "metric_name": metric_name,
                "task": task_name,
                "model_i": model_i,
                "model_j": model_j,
                "metric_i_aligned": metric_i,
                "metric_j_aligned": metric_j,
                "delta_metric_signed": delta_metric_signed,
                "delta_metric_abs": delta_metric_abs,
                "score_i": score_i,
                "score_j": score_j,
                "delta_score_signed": delta_score_signed,
                "delta_score_abs": delta_score_abs,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Evaluation
# ============================================================


def evaluate_pairs_dataframe(df_pairs: pd.DataFrame, protocol: str) -> dict[str, float]:
    if protocol == "signed":
        x = df_pairs["delta_metric_signed"].to_numpy(dtype=np.float64)
        y = df_pairs["delta_score_signed"].to_numpy(dtype=np.float64)
        target_name = "delta_signed"
    elif protocol == "abs":
        x = df_pairs["delta_metric_abs"].to_numpy(dtype=np.float64)
        y = df_pairs["delta_score_abs"].to_numpy(dtype=np.float64)
        target_name = "delta_abs"
    else:
        raise ValueError(f"Неизвестный protocol: {protocol}")

    corr = compute_correlations(x, y)
    corr["target"] = target_name
    return corr


def evaluate_single_metric(
    metric_name: str,
    scores: dict[str, float],
    higher_is_better: bool,
    downstream: dict[str, dict[str, float]],
    protocol: str,
    out_pairs_dir: Path | None = None,
) -> list[dict[str, Any]]:
    aligned_scores = align_scores(scores, higher_is_better)
    common_models = intersect_models(aligned_scores, downstream)
    common_tasks = intersect_tasks(common_models, downstream)

    results: list[dict[str, Any]] = []
    export_frames: list[pd.DataFrame] = []

    for task_name in common_tasks:
        df_pairs = build_pairs_for_task(
            metric_name=metric_name,
            aligned_scores=aligned_scores,
            downstream=downstream,
            common_models=common_models,
            task_name=task_name,
        )

        export_frames.append(df_pairs)

        corr = evaluate_pairs_dataframe(df_pairs, protocol=protocol)

        results.append(
            {
                "metric_name": metric_name,
                "metric_kind": "single_diff",
                "protocol": protocol,
                "task": task_name,
                "target": corr["target"],
                "n_models_common": len(common_models),
                "n_pairs": corr["n_pairs_used"],
                "spearman": corr["spearman"],
                "pearson": corr["pearson"],
                "kendall": corr["kendall"],
            }
        )

    rows_protocol = results[:]
    results.append(
        {
            "metric_name": metric_name,
            "metric_kind": "single_diff",
            "protocol": protocol,
            "task": "__mean__",
            "target": "delta_signed" if protocol == "signed" else "delta_abs",
            "n_models_common": len(common_models),
            "n_pairs": int(
                mean_ignore_nan([float(r["n_pairs"]) for r in rows_protocol])
            ),
            "spearman": mean_ignore_nan([float(r["spearman"]) for r in rows_protocol]),
            "pearson": mean_ignore_nan([float(r["pearson"]) for r in rows_protocol]),
            "kendall": mean_ignore_nan([float(r["kendall"]) for r in rows_protocol]),
        }
    )

    if out_pairs_dir is not None and export_frames:
        out_pairs_dir.mkdir(parents=True, exist_ok=True)
        df_export = pd.concat(export_frames, axis=0, ignore_index=True)
        save_csv(df_export, out_pairs_dir / f"{metric_name}_pairs.csv")

    return results


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Оценка одиночных метрик через разности по парам моделей."
    )
    parser.add_argument(
        "--single_metrics_dir",
        type=Path,
        required=True,
        help="Папка с single-metrics в формате <metric>/<model>.json.",
    )
    parser.add_argument(
        "--downstream_json",
        type=Path,
        required=True,
        help="JSON-файл с downstream-оценками в формате model -> task -> score.",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        required=True,
        help="Путь к итоговому CSV с корреляциями.",
    )
    parser.add_argument(
        "--out_pairs_dir",
        type=Path,
        default=None,
        help="Необязательная папка для сохранения попарных таблиц.",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        required=True,
        choices=["signed", "abs"],
        help=(
            "Режим сравнения: "
            "signed -> corr(u(e1)-u(e2), Δacc), "
            "abs -> corr(|u(e1)-u(e2)|, |Δacc|)."
        ),
    )
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    args = parse_args()

    single_metrics_dir: Path = args.single_metrics_dir
    downstream_json: Path = args.downstream_json
    out_csv: Path = args.out_csv
    out_pairs_dir: Path | None = args.out_pairs_dir
    protocol: str = args.protocol

    if not single_metrics_dir.exists():
        raise FileNotFoundError(
            f"Папка single-metrics не найдена: {single_metrics_dir}"
        )

    if not downstream_json.exists():
        raise FileNotFoundError(f"Файл downstream_json не найден: {downstream_json}")

    metric_dirs = discover_metric_dirs(single_metrics_dir)
    if not metric_dirs:
        raise FileNotFoundError(
            f"В папке {single_metrics_dir} не найдено директорий с метриками"
        )

    downstream = load_downstream_table(downstream_json)

    print("============================================================")
    print("ОЦЕНКА ОДИНОЧНЫХ МЕТРИК")
    print("============================================================")
    print(f"Папка single-metrics : {single_metrics_dir}")
    print(f"Файл downstream      : {downstream_json}")
    print(f"Найдено метрик       : {len(metric_dirs)}")
    print(f"Найдено моделей down : {len(downstream)}")
    print(f"Protocol             : {protocol}")
    print("============================================================")

    all_results: list[dict[str, Any]] = []

    for metric_dir in metric_dirs:
        metric_name, scores, higher_is_better = load_single_metric_scores(metric_dir)

        metric_results = evaluate_single_metric(
            metric_name=metric_name,
            scores=scores,
            higher_is_better=higher_is_better,
            downstream=downstream,
            protocol=protocol,
            out_pairs_dir=out_pairs_dir,
        )
        all_results.extend(metric_results)

        mean_rows = [r for r in metric_results if r["task"] == "__mean__"]
        print(f"[OK] {metric_name}")
        for row in mean_rows:
            print(
                f"     protocol={row['protocol']:<6} "
                f"spearman={row['spearman']:.4f} "
                f"pearson={row['pearson']:.4f} "
                f"kendall={row['kendall']:.4f} "
                f"n_pairs={row['n_pairs']}"
            )

    df_results = pd.DataFrame(all_results)
    save_csv(df_results, out_csv)

    print("\n============================================================")
    print("ОЦЕНКА ОДИНОЧНЫХ МЕТРИК ЗАВЕРШЕНА")
    print("============================================================")
    print(f"Результаты сохранены в: {out_csv}")
    if out_pairs_dir is not None:
        print(f"Попарные таблицы      : {out_pairs_dir}")
    print("============================================================")


if __name__ == "__main__":
    main()
