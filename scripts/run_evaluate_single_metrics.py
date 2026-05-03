from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


VALID_PLOT_EXTS = ("png", "pdf", "svg")


# ============================================================
# Вспомогательные функции
# ============================================================


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_summary_csv(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_metric: dict[str, list[dict[str, Any]]] = {}

    for row in results:
        by_metric.setdefault(str(row["metric_name"]), []).append(row)

    for metric_name, metric_rows in sorted(by_metric.items()):
        mean_rows = [r for r in metric_rows if str(r.get("task")) == "__mean__"]
        if not mean_rows:
            continue

        mean_row = mean_rows[0]
        task_rows = [r for r in metric_rows if str(r.get("task")) != "__mean__"]
        rows.append(
            {
                "metric_file": metric_name,
                "pair_agg": "single_diff",
                "is_paired": False,
                "is_symmetric": mean_row["protocol"] == "abs",
                "pairs_total": mean_row["n_pairs"],
                "tasks": len(task_rows),
                "spearman_mean": mean_row["spearman"],
                "pearson_mean": mean_row["pearson"],
                "kendall_mean": mean_row["kendall"],
                "correct_ratio_mean": mean_row.get("correct_ratio_mean", float("nan")),
                "correct_ratio_flip_mean": mean_row.get(
                    "correct_ratio_flip_mean", float("nan")
                ),
                "correct_ratio_adjusted_mean": mean_row.get(
                    "correct_ratio_adjusted_mean", float("nan")
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "metric_file",
            "pair_agg",
            "is_paired",
            "is_symmetric",
            "pairs_total",
            "tasks",
            "spearman_mean",
            "pearson_mean",
            "kendall_mean",
            "correct_ratio_mean",
            "correct_ratio_flip_mean",
            "correct_ratio_adjusted_mean",
        ],
    )


def parse_plots_exts(raw: str) -> list[str]:
    items = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not items:
        raise ValueError("`--plots_ext` должен содержать хотя бы одно расширение.")

    invalid = [ext for ext in items if ext not in VALID_PLOT_EXTS]
    if invalid:
        raise ValueError(
            f"Неподдерживаемые расширения графиков: {invalid}. "
            f"Допустимые: {list(VALID_PLOT_EXTS)}"
        )

    unique: list[str] = []
    seen = set()
    for ext in items:
        if ext not in seen:
            unique.append(ext)
            seen.add(ext)
    return unique


def _dataset_base_from_key(dataset_key: str) -> str:
    key = str(dataset_key)
    for split in ("_test", "_train", "_val", "_valid"):
        pos = key.find(split)
        if pos >= 0:
            return key[:pos]
    return key


def _infer_downstream_json(dataset_key: str) -> Path:
    dataset = _dataset_base_from_key(dataset_key)
    candidates = [
        Path("data") / "downstream" / f"{dataset_key}_mlp.json",
        Path("data") / "downstream" / f"{dataset_key}_linear_probe.json",
        Path("data") / "downstream" / f"{dataset}_mlp.json",
        Path("data") / "downstream" / f"{dataset}_linear_probe.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Не удалось автоматически найти downstream JSON. Проверены: "
        + ", ".join(str(p) for p in candidates)
    )


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


def correct_ranking_ratio(
    metric_vals: np.ndarray,
    delta_vals: np.ndarray,
    boundary: float = 0.0,
) -> float:
    m = np.asarray(metric_vals, dtype=np.float64)
    d = np.asarray(delta_vals, dtype=np.float64)
    mask = np.isfinite(m) & np.isfinite(d)
    m = m[mask]
    d = d[mask]
    if m.size == 0:
        return float("nan")

    ok = ((m >= boundary) & (d >= 0.0)) | ((m <= boundary) & (d <= 0.0))
    return float(np.mean(ok))


def adjust_single_correct_ratio(cr: float) -> float:
    if np.isnan(cr):
        return cr
    return max(cr, 1.0 - cr)


# ============================================================
# Загрузка single metrics
# Canonical формат: <single_metrics_dir>/metrics/<metric_name>/<model>.json
# Legacy формат:    <single_metrics_dir>/<metric_name>/<model>.json
# ============================================================


def resolve_metrics_dir(single_metrics_dir: Path) -> Path:
    canonical = single_metrics_dir / "metrics"
    if canonical.is_dir():
        return canonical
    return single_metrics_dir


def discover_metric_dirs(single_metrics_dir: Path) -> list[Path]:
    metrics_dir = resolve_metrics_dir(single_metrics_dir)
    return sorted([p for p in metrics_dir.iterdir() if p.is_dir()])


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
        cr = correct_ranking_ratio(x, y, boundary=0.0)
        cr_flip = adjust_single_correct_ratio(cr)
        cr_adjusted = cr_flip
    elif protocol == "abs":
        x = df_pairs["delta_metric_abs"].to_numpy(dtype=np.float64)
        y = df_pairs["delta_score_abs"].to_numpy(dtype=np.float64)
        target_name = "delta_abs"
        cr = cr_flip = cr_adjusted = float("nan")
    else:
        raise ValueError(f"Неизвестный protocol: {protocol}")

    corr = compute_correlations(x, y)
    corr["target"] = target_name
    corr["correct_ratio"] = cr
    corr["correct_ratio_flip_invariant"] = cr_flip
    corr["correct_ratio_adjusted_like_last_year"] = cr_adjusted
    return corr


def evaluate_single_metric(
    metric_name: str,
    scores: dict[str, float],
    higher_is_better: bool,
    downstream: dict[str, dict[str, float]],
    protocol: str,
    out_pairs_dir: Path | None = None,
    plots_dir: Path | None = None,
    plots_mode: str = "none",
    plots_exts: list[str] | None = None,
) -> list[dict[str, Any]]:
    aligned_scores = align_scores(scores, higher_is_better)
    common_models = intersect_models(aligned_scores, downstream)
    common_tasks = intersect_tasks(common_models, downstream)
    if plots_exts is None:
        plots_exts = ["png"]

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
                "correct_ratio": corr["correct_ratio"],
                "correct_ratio_flip_invariant": corr["correct_ratio_flip_invariant"],
                "correct_ratio_adjusted_like_last_year": corr[
                    "correct_ratio_adjusted_like_last_year"
                ],
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
            "correct_ratio_mean": mean_ignore_nan(
                [float(r["correct_ratio"]) for r in rows_protocol]
            ),
            "correct_ratio_flip_mean": mean_ignore_nan(
                [float(r["correct_ratio_flip_invariant"]) for r in rows_protocol]
            ),
            "correct_ratio_adjusted_mean": mean_ignore_nan(
                [
                    float(r["correct_ratio_adjusted_like_last_year"])
                    for r in rows_protocol
                ]
            ),
        }
    )

    if out_pairs_dir is not None and export_frames:
        out_pairs_dir.mkdir(parents=True, exist_ok=True)
        df_export = pd.concat(export_frames, axis=0, ignore_index=True)
        save_csv(df_export, out_pairs_dir / f"{metric_name}_pairs.csv")

    maybe_make_plots(
        metric_name=metric_name,
        aligned_scores=aligned_scores,
        downstream=downstream,
        common_models=common_models,
        common_tasks=common_tasks,
        protocol=protocol,
        plots_dir=plots_dir,
        plots_mode=plots_mode,
        plots_exts=plots_exts,
    )

    return results


# ============================================================
# Plotting
# ============================================================


def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(s))


def pairs_xy_and_info(df_pairs: pd.DataFrame, protocol: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
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
    if protocol == "signed":
        cr = correct_ranking_ratio(x, y, boundary=0.0)
        cr_adj = adjust_single_correct_ratio(cr)
    else:
        cr = cr_adj = float("nan")

    return x, y, {
        "target": target_name,
        "n_pairs": corr["n_pairs_used"],
        "spearman": corr["spearman"],
        "pearson": corr["pearson"],
        "kendall": corr["kendall"],
        "correct_ratio": cr,
        "correct_ratio_adjusted": cr_adj,
    }


def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    out_path: Path,
    subtitle: str,
    plots_exts: list[str],
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib недоступен; сохранить графики нельзя")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=18)
    ax.axvline(0.0)
    ax.axhline(0.0)
    ax.set_title(title)
    ax.set_xlabel("single metric diff")
    ax.set_ylabel("target")
    fig.text(0.01, 0.01, subtitle, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    base = out_path.with_suffix("")
    for ext in plots_exts:
        fig.savefig(base.with_suffix(f".{ext}"))
    plt.close(fig)


def maybe_make_plots(
    metric_name: str,
    aligned_scores: dict[str, float],
    downstream: dict[str, dict[str, float]],
    common_models: list[str],
    common_tasks: list[str],
    protocol: str,
    plots_dir: Path | None,
    plots_mode: str,
    plots_exts: list[str],
) -> None:
    if plots_dir is None or plots_mode == "none":
        return
    if plt is None:
        raise RuntimeError("matplotlib недоступен; сохранить графики нельзя")

    def subtitle(info: dict[str, Any]) -> str:
        return (
            f"spearman={info['spearman']:.3f}, pearson={info['pearson']:.3f}, "
            f"kendall={info['kendall']:.3f}, cr={info['correct_ratio']:.3f}, "
            f"cr_adj={info['correct_ratio_adjusted']:.3f} | "
            f"pairs={info['n_pairs']} | protocol={protocol}"
        )

    if plots_mode in {"all", "alltasks"}:
        frames = [
            build_pairs_for_task(
                metric_name=metric_name,
                aligned_scores=aligned_scores,
                downstream=downstream,
                common_models=common_models,
                task_name=task_name,
            )
            for task_name in common_tasks
        ]
        if frames:
            df_all = pd.concat(frames, axis=0, ignore_index=True)
            x, y, info = pairs_xy_and_info(df_all, protocol=protocol)
            plot_scatter(
                x=x,
                y=y,
                title=f"{metric_name} | ALL TASKS | single_{protocol}",
                out_path=plots_dir / f"{safe_filename(metric_name)}__ALLTASKS.png",
                subtitle=subtitle(info),
                plots_exts=plots_exts,
            )

    if plots_mode in {"all", "tasks"} and len(common_tasks) > 1:
        for task_name in common_tasks:
            df_task = build_pairs_for_task(
                metric_name=metric_name,
                aligned_scores=aligned_scores,
                downstream=downstream,
                common_models=common_models,
                task_name=task_name,
            )
            x, y, info = pairs_xy_and_info(df_task, protocol=protocol)
            plot_scatter(
                x=x,
                y=y,
                title=f"{metric_name} | task={task_name} | single_{protocol}",
                out_path=plots_dir
                / f"{safe_filename(metric_name)}__task_{safe_filename(task_name)}.png",
                subtitle=subtitle(info),
                plots_exts=plots_exts,
            )


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
        default=None,
        help=(
            "Папка с single-metrics. Поддерживает canonical "
            "<dataset>/metrics/<metric>/<model>.json и legacy <metric>/<model>.json."
        ),
    )
    parser.add_argument(
        "--dataset_key",
        type=str,
        default="",
        help="Имя датасета/сплита, например food101_test. Автоматически задает стандартные пути.",
    )
    parser.add_argument(
        "--single_metrics_root",
        type=Path,
        default=Path("data") / "single_metrics",
        help="Корень single metrics store для режима --dataset_key.",
    )
    parser.add_argument(
        "--experiment_dir",
        type=Path,
        default=None,
        help="Если задано, out_csv/out_pairs_dir/plots_dir выводятся стандартно из experiment_dir.",
    )
    parser.add_argument(
        "--downstream_json",
        type=Path,
        default=None,
        help="JSON-файл с downstream-оценками в формате model -> task -> score.",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=None,
        help=(
            "Путь к итоговому CSV с корреляциями. Формат совпадает с "
            "run_evaluate_metrics.py: одна строка на метрику, агрегаты по задачам."
        ),
    )
    parser.add_argument(
        "--out_pairs_dir",
        type=Path,
        default=None,
        help="Необязательная папка для сохранения попарных таблиц.",
    )
    parser.add_argument(
        "--plots_dir",
        type=Path,
        default=None,
        help="Необязательная папка для scatter-графиков single metric diff vs target.",
    )
    parser.add_argument(
        "--plots_mode",
        type=str,
        choices=["none", "all", "alltasks", "tasks"],
        default="none",
        help=(
            "Какие scatter-графики строить: none, all, alltasks или tasks. "
            "Работает только если задан --plots_dir."
        ),
    )
    parser.add_argument(
        "--plots_ext",
        type=str,
        default="png",
        help="Одно или несколько расширений графиков через запятую, например png или svg,png.",
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

    dataset_key = str(args.dataset_key).strip()
    if args.single_metrics_dir is None:
        if not dataset_key:
            raise ValueError("Нужно указать либо --single_metrics_dir, либо --dataset_key.")
        args.single_metrics_dir = args.single_metrics_root / dataset_key
    if args.downstream_json is None:
        if not dataset_key:
            dataset_key = args.single_metrics_dir.name
        args.downstream_json = _infer_downstream_json(dataset_key)
    if args.experiment_dir is not None:
        dataset_base = _dataset_base_from_key(dataset_key or args.single_metrics_dir.name)
        if args.out_csv is None:
            args.out_csv = (
                args.experiment_dir
                / "reports"
                / f"{dataset_base}_single_metric_eval_signed.csv"
            )
        if args.out_pairs_dir is None:
            args.out_pairs_dir = (
                args.experiment_dir
                / "reports"
                / f"{dataset_base}_single_metric_pairs_signed"
            )
        if args.plots_dir is None:
            args.plots_dir = (
                args.experiment_dir
                / "plots"
                / f"{dataset_base}_single_metric_scatter_signed"
            )
    if args.out_csv is None:
        raise ValueError("Нужно указать либо --out_csv, либо --experiment_dir.")

    single_metrics_dir: Path = args.single_metrics_dir
    downstream_json: Path = args.downstream_json
    out_csv: Path = args.out_csv
    out_pairs_dir: Path | None = args.out_pairs_dir
    plots_dir: Path | None = args.plots_dir
    plots_mode: str = args.plots_mode
    plots_exts = parse_plots_exts(args.plots_ext)
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
    print(f"Папка значений       : {resolve_metrics_dir(single_metrics_dir)}")
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
            plots_dir=plots_dir,
            plots_mode=plots_mode,
            plots_exts=plots_exts,
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
                f"cr_adj={row['correct_ratio_adjusted_mean']:.4f} "
                f"n_pairs={row['n_pairs']}"
            )

    df_results = build_summary_csv(all_results)
    save_csv(df_results, out_csv)

    print("\n============================================================")
    print("ОЦЕНКА ОДИНОЧНЫХ МЕТРИК ЗАВЕРШЕНА")
    print("============================================================")
    print(f"Результаты сохранены в: {out_csv}")
    if out_pairs_dir is not None:
        print(f"Попарные таблицы      : {out_pairs_dir}")
    if plots_dir is not None and plots_mode != "none":
        print(f"Scatter-графики       : {plots_dir}")
    print("============================================================")


if __name__ == "__main__":
    main()
