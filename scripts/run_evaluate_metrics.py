"""
Шаг 2 протокола:

Берём посчитанные матрицы метрик (metric_matrices/*.npz) и downstream-качества моделей (json),
и строим supervised оценку корреляции метрики с downstream.

Поддерживаем 2 протокола:

1) delta_signed (как раньше):
   ordered пары (A,B), target = Δacc(B)-Δacc(A)
   применимо к directed/antisym (и формально можно к sym, но смысла мало)

2) delta_abs:
   unordered пары (A,B) с A<B, target = |Δacc(B)-Δacc(A)|
   это естественный вариант для симметричных метрик (sim).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau

# Опционально для графиков (не делаем обязательной зависимостью)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# ================================================================
# Загрузка матрицы метрики (.npz)
# ================================================================

def load_metric_matrix(path: str) -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    """
    Загружает .npz-файл метрики.

    Поддерживает два формата:
    1) Новый (как в run_compute_embedding_metrics.py):
       - model_names
       - matrix
       - meta_json (опционально)
    2) Старый/альтернативный:
       - model_names
       - scores
       - meta_json (опционально)

    Возвращает:
    - model_names: list[str]
    - scores: (M, M) float
    - meta: dict
    """
    data = np.load(path, allow_pickle=True)

    if "model_names" not in data.files:
        raise KeyError(f"В {path} отсутствует 'model_names'. Ключи: {data.files}")
    model_names = list(data["model_names"].tolist())

    if "scores" in data.files:
        scores = np.asarray(data["scores"], dtype=float)
    elif "matrix" in data.files:
        scores = np.asarray(data["matrix"], dtype=float)
    else:
        raise KeyError(f"В {path} нет ни 'scores', ни 'matrix'. Ключи: {data.files}")

    meta: Dict[str, Any] = {}
    if "meta_json" in data.files:
        meta_json = data["meta_json"]
        meta_json = meta_json.item() if getattr(meta_json, "shape", None) == () else meta_json.tolist()

        if isinstance(meta_json, str):
            try:
                meta = json.loads(meta_json)
            except Exception:
                meta = {}
        elif isinstance(meta_json, dict):
            meta = meta_json

    return model_names, scores, meta


# ================================================================
# Загрузка downstream-оценок моделей
# ================================================================

def load_downstream_table(path: str) -> Dict[str, Dict[str, float]]:
    """
    Формат входного файла: JSON

    {
      "modelA": {"task1": 0.77, "task2": 0.61},
      "modelB": {"task1": 0.80, "task2": 0.59},
      ...
    }

    model -> task -> score
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError("Downstream json должен быть словарём вида model->task->score")

    return obj


def intersect_models(metric_models: List[str], downstream: Dict[str, Dict[str, float]]) -> List[str]:
    """
    Оставляем только те модели, которые есть и в матрице метрики, и в downstream-таблице.
    """
    dset = set(downstream.keys())
    keep = [m for m in metric_models if m in dset]
    if len(keep) < 2:
        raise RuntimeError("Слишком мало моделей на пересечении между metric и downstream.")
    return keep


# ================================================================
# Построение пар и оценка как в прошлом году
# ================================================================

@dataclass
class TaskEval:
    task: str
    n_pairs: int
    spearman: float
    pearson: float
    kendall: float
    correct_ratio: float
    correct_ratio_flip_invariant: float
    correct_ratio_adjusted_like_last_year: float


def _correct_ranking_ratio(metric_vals: np.ndarray, delta_vals: np.ndarray, boundary: float = 0.0) -> float:
    """
    Как в прошлогоднем репозитории:
      correct = (metric>=b and delta>=0) or (metric<=b and delta<=0)
    """
    m = metric_vals
    d = delta_vals
    ok = ((m >= boundary) & (d >= 0)) | ((m <= boundary) & (d <= 0))
    return float(np.mean(ok)) if ok.size else float("nan")


def _adjust_correct_ratio_like_last_year(cr: float, is_paired: bool, is_symmetric: bool) -> float:
    """
    Ровно логика прошлого года:

    if not is_paired or is_symmetric:
        cr = max(cr, 1 - cr)

    Смысл:
    - для непарных метрик (где сравнение идёт через разность) знак может быть выбран наоборот
    - для антисимметричных метрик знак меняется при swap (X,Y) -> (Y,X), поэтому тоже flip-invariant
    - для направленных парных метрик (is_paired=True, is_symmetric=False) переворот НЕ допускаем
    """
    if np.isnan(cr):
        return cr
    if (not is_paired) or is_symmetric:
        return max(cr, 1.0 - cr)
    return cr


def eval_one_metric(
    metric_path: str,
    downstream: Dict[str, Dict[str, float]],
    use_tasks: Optional[List[str]] = None,
    eval_protocol: str = "delta_signed",
) -> Tuple[str, Dict[str, Any], List[TaskEval], Dict[str, float]]:
    model_names, M, meta = load_metric_matrix(metric_path)
    models = intersect_models(model_names, downstream)
    idx = {m: i for i, m in enumerate(model_names)}

    # ИСПРАВЛЕНИЕ: используем флаги так же, как в прошлом году
    is_paired = bool(meta.get("is_paired", True))
    is_symmetric = bool(meta.get("is_symmetric", False))

    all_tasks = sorted({t for m in models for t in downstream[m].keys()})
    tasks = use_tasks if use_tasks else all_tasks
    tasks = [t for t in tasks if t in all_tasks]
    if not tasks:
        raise RuntimeError("После фильтрации use_tasks не осталось задач.")

    per_task: List[TaskEval] = []

    for task in tasks:
        avail = [m for m in models if task in downstream[m]]
        if len(avail) < 2:
            continue

        metric_vals = []
        delta_vals = []

        if eval_protocol == "delta_signed":
            for a in avail:
                for b in avail:
                    if a == b:
                        continue
                    i = idx[a]
                    j = idx[b]
                    v = float(M[i, j])
                    if np.isnan(v):
                        continue

                    da = float(downstream[a][task])
                    db = float(downstream[b][task])
                    delta = db - da

                    metric_vals.append(v)
                    delta_vals.append(delta)

        elif eval_protocol == "delta_abs":
            # Неупорядоченные пары (i<j), целевая величина — |Δacc|
            for ii in range(len(avail)):
                a = avail[ii]
                da = float(downstream[a][task])
                for jj in range(ii + 1, len(avail)):
                    b = avail[jj]
                    i = idx[a]
                    j = idx[b]
                    v = float(M[i, j])
                    if np.isnan(v):
                        continue
                    db = float(downstream[b][task])
                    delta_abs = abs(db - da)

                    metric_vals.append(v)
                    delta_vals.append(delta_abs)
        else:
            raise ValueError(f"Неизвестный eval_protocol: {eval_protocol}")

        metric_vals = np.asarray(metric_vals, dtype=float)
        delta_vals = np.asarray(delta_vals, dtype=float)

        if metric_vals.size < 2:
            sp = pr = kd = float("nan")
        else:
            sp = float(spearmanr(metric_vals, delta_vals).correlation)
            pr = float(pearsonr(metric_vals, delta_vals)[0])
            kd = float(kendalltau(metric_vals, delta_vals).correlation)

        if eval_protocol == "delta_signed":
            cr = _correct_ranking_ratio(metric_vals, delta_vals, boundary=0.0)
            cr_flip = max(cr, 1.0 - cr) if not np.isnan(cr) else cr
            cr_adj = _adjust_correct_ratio_like_last_year(cr, is_paired=is_paired, is_symmetric=is_symmetric)
        else:
            # Для протокола |Δacc| "correct ratio" по знаку не определено
            cr = cr_flip = cr_adj = float("nan")

        per_task.append(TaskEval(
            task=task,
            n_pairs=int(metric_vals.size),
            spearman=sp,
            pearson=pr,
            kendall=kd,
            correct_ratio=cr,
            correct_ratio_flip_invariant=cr_flip,
            correct_ratio_adjusted_like_last_year=cr_adj,
        ))

    # Сводка: средние по задачам (как в логе)
    def _nanmean(vals: List[float]) -> float:
        arr = np.asarray(vals, dtype=float)
        return float(np.nanmean(arr)) if arr.size else float("nan")

    summary = {
        "pairs_total": int(sum(t.n_pairs for t in per_task)),
        "tasks": int(len(per_task)),
        "spearman_mean": _nanmean([t.spearman for t in per_task]),
        "pearson_mean": _nanmean([t.pearson for t in per_task]),
        "kendall_mean": _nanmean([t.kendall for t in per_task]),
        "correct_ratio_mean": _nanmean([t.correct_ratio for t in per_task]),
        "correct_ratio_flip_mean": _nanmean([t.correct_ratio_flip_invariant for t in per_task]),
        "correct_ratio_adjusted_mean": _nanmean([t.correct_ratio_adjusted_like_last_year for t in per_task]),
    }

    # meta -> добавим то, что полезно видеть в отчёте
    meta_out = dict(meta)
    meta_out["is_paired"] = is_paired
    meta_out["is_symmetric"] = is_symmetric

    return os.path.basename(metric_path), meta_out, per_task, summary


def _compute_pair_cloud(
    metric_path: str,
    downstream: Dict[str, Dict[str, float]],
    task: Optional[str] = None,
    eval_protocol: str = "delta_signed",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Готовит облако точек для scatter: x=metric(A,B), y=target.

    delta_signed: target = Δacc(B)-Δacc(A), упорядоченные пары
    delta_abs: target = |Δacc|, неупорядоченные пары
    """
    model_names, M, meta = load_metric_matrix(metric_path)
    models = intersect_models(model_names, downstream)
    idx = {m: i for i, m in enumerate(model_names)}

    is_paired = bool(meta.get("is_paired", True))
    is_symmetric = bool(meta.get("is_symmetric", False))

    all_tasks = sorted({t for m in models for t in downstream[m].keys()})
    tasks = [task] if task is not None else all_tasks

    xs: List[float] = []
    ys: List[float] = []

    for t in tasks:
        avail = [m for m in models if t in downstream[m]]
        if len(avail) < 2:
            continue

        if eval_protocol == "delta_signed":
            for a in avail:
                for b in avail:
                    if a == b:
                        continue
                    i = idx[a]
                    j = idx[b]
                    v = float(M[i, j])
                    if np.isnan(v):
                        continue
                    da = float(downstream[a][t])
                    db = float(downstream[b][t])
                    xs.append(v)
                    ys.append(db - da)
        elif eval_protocol == "delta_abs":
            for ii in range(len(avail)):
                a = avail[ii]
                da = float(downstream[a][t])
                for jj in range(ii + 1, len(avail)):
                    b = avail[jj]
                    i = idx[a]
                    j = idx[b]
                    v = float(M[i, j])
                    if np.isnan(v):
                        continue
                    db = float(downstream[b][t])
                    xs.append(v)
                    ys.append(abs(db - da))
        else:
            raise ValueError(f"Неизвестный eval_protocol: {eval_protocol}")

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size >= 2:
        sp = float(spearmanr(x, y).correlation)
        pr = float(pearsonr(x, y)[0])
        kd = float(kendalltau(x, y).correlation)
        if eval_protocol == "delta_signed":
            cr = _correct_ranking_ratio(x, y, boundary=0.0)
            cr_adj = _adjust_correct_ratio_like_last_year(cr, is_paired=is_paired, is_symmetric=is_symmetric)
        else:
            cr_adj = float("nan")
    else:
        sp = pr = kd = cr_adj = float("nan")

    info = {
        "task": ("ALL_TASKS" if task is None else str(task)),
        "n_pairs": int(x.size),
        "spearman": sp,
        "pearson": pr,
        "kendall": kd,
        "cr_adj": cr_adj,
        "eval_protocol": eval_protocol,
        "meta_is_paired": is_paired,
        "meta_is_symmetric": is_symmetric,
    }
    return x, y, info


def _safe_filename(s: str) -> str:
    return "".join((c if c.isalnum() or c in "._-" else "_") for c in s)


def _plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    out_path: str,
    subtitle: str,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib недоступен; сохранить графики нельзя")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=18)
    ax.axvline(0.0)
    ax.axhline(0.0)
    ax.set_title(title)
    ax.set_xlabel("metric")
    ax.set_ylabel("target")
    fig.text(0.01, 0.01, subtitle, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)


def _maybe_make_plots(
    metric_path: str,
    downstream: Dict[str, Dict[str, float]],
    plots_dir: str,
    plots_ext: str,
    plots_mode: str,
    use_tasks: Optional[List[str]],
    eval_protocol: str,
) -> None:
    """plots_mode:
      - none
      - all (ALL_TASKS + per-task)
      - alltasks (только ALL_TASKS)
      - tasks (только per-task)
    """
    if plots_mode == "none":
        return

    model_names, _, _ = load_metric_matrix(metric_path)
    models = intersect_models(model_names, downstream)
    all_tasks = sorted({t for m in models for t in downstream[m].keys()})
    tasks = use_tasks if use_tasks else all_tasks
    tasks = [t for t in tasks if t in all_tasks]

    single_task = (len(tasks) == 1)

    stem = os.path.splitext(os.path.basename(metric_path))[0]

    def _subtitle(info: Dict[str, Any]) -> str:
        return (
            f"spearman={info['spearman']:.3f}, pearson={info['pearson']:.3f}, "
            f"kendall={info['kendall']:.3f}, cr_adj={info['cr_adj']:.3f} | "
            f"pairs={info['n_pairs']} | paired={info['meta_is_paired']} sym={info['meta_is_symmetric']} | "
            f"protocol={info['eval_protocol']}"
        )

    if plots_mode in {"all", "alltasks"}:
        x, y, info = _compute_pair_cloud(metric_path, downstream, task=None, eval_protocol=eval_protocol)
        out = os.path.join(plots_dir, f"{_safe_filename(stem)}__ALLTASKS.{plots_ext}")
        _plot_scatter(x, y, title=f"{stem} | ALL TASKS | {eval_protocol}", out_path=out, subtitle=_subtitle(info))

    if plots_mode in {"all", "tasks"}:
        if single_task:
            return
        for t in tasks:
            x, y, info = _compute_pair_cloud(metric_path, downstream, task=t, eval_protocol=eval_protocol)
            out = os.path.join(plots_dir, f"{_safe_filename(stem)}__task_{_safe_filename(str(t))}.{plots_ext}")
            _plot_scatter(x, y, title=f"{stem} | task={t} | {eval_protocol}", out_path=out, subtitle=_subtitle(info))


# ================================================================
# CSV / Markdown-отчёты
# ================================================================

def save_csv(out_csv: str, reports: List[Tuple[str, Dict[str, Any], List[TaskEval], Dict[str, float]]]) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Плоская таблица: одна строка на метрику, агрегаты по задачам
    fieldnames = [
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
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for metric_file, meta, _per_task, summary in reports:
            row = {
                "metric_file": metric_file,
                "pair_agg": meta.get("pair_agg", ""),
                "is_paired": meta.get("is_paired", ""),
                "is_symmetric": meta.get("is_symmetric", ""),
                **summary,
            }
            w.writerow(row)


def save_md(out_md: str, reports: List[Tuple[str, Dict[str, Any], List[TaskEval], Dict[str, float]]]) -> None:
    os.makedirs(os.path.dirname(out_md), exist_ok=True)

    # Простая таблица Markdown
    header = "| metric | pair_agg | spearman | pearson | kendall | cr_adj |\n|---|---|---:|---:|---:|---:|\n"
    lines = [header]
    for metric_file, meta, _per_task, summary in reports:
        lines.append(
            f"| {metric_file} | {meta.get('pair_agg','')} | "
            f"{summary['spearman_mean']:.4f} | {summary['pearson_mean']:.4f} | {summary['kendall_mean']:.4f} | "
            f"{summary['correct_ratio_adjusted_mean']:.4f} |\n"
        )

    with open(out_md, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(description="Оценить матрицы метрик эмбеддингов по downstream-оценкам.")
    parser.add_argument("--experiment_dir", type=str, default="", help="Если задано, использовать стандартную структуру эксперимента внутри этой папки.")
    parser.add_argument("--metrics_dir", type=str, default="", help="Папка с матрицами метрик (*.npz). Если пусто, путь берётся из --experiment_dir.")
    parser.add_argument("--downstream_json", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="", help="Путь к выходному CSV. Если пусто, путь берётся из --experiment_dir.")
    parser.add_argument("--tasks", type=str, default="", help="Имена задач через запятую для использования (пусто = все задачи).")
    parser.add_argument("--out_md", type=str, default="", help="Необязательный путь для сохранения сводной таблицы в Markdown.")
    parser.add_argument("--eval_protocol", type=str, default="delta_signed", choices=["delta_signed", "delta_abs"],
                        help="Как строить supervision из downstream: delta_signed использует упорядоченные пары и Δacc(B)-Δacc(A); delta_abs использует неупорядоченные пары и |Δacc|.")

    # Построение графиков (необязательно)
    parser.add_argument("--plots_dir", type=str, default="", help="Если задано, дополнительно сохранять scatter-графики в эту папку.")
    parser.add_argument("--plots_ext", type=str, default="png", choices=["png", "pdf", "svg"], help="Расширение файлов графиков.")
    parser.add_argument(
        "--plots_mode",
        type=str,
        default="alltasks",
        choices=["none", "all", "alltasks", "tasks"],
        help=(
            "Какие графики строить: none | all (ALL_TASKS+per-task) | alltasks (только ALL_TASKS) | tasks (только per-task). "
            "Если задача только одна, графики per-task пропускаются, чтобы не делать дубликаты."
        ),
    )

    args = parser.parse_args()

    if args.experiment_dir:
        if not args.metrics_dir:
            args.metrics_dir = os.path.join(args.experiment_dir, 'metric_matrices')
        if not args.out_csv:
            # Пытаемся получить стабильное имя из downstream_json (например, cifar10_linear_probe.json -> cifar10_eval.csv)
            stem = os.path.splitext(os.path.basename(args.downstream_json))[0]
            dataset = stem.split('_')[0] if stem else 'eval'
            args.out_csv = os.path.join(args.experiment_dir, 'reports', f'{dataset}_eval.csv')
        if args.plots_dir == "":
            args.plots_dir = os.path.join(args.experiment_dir, 'plots')
        # Сохраняем копию downstream_json внутри эксперимента для воспроизводимости
        dstdir = os.path.join(args.experiment_dir, 'downstream')
        os.makedirs(dstdir, exist_ok=True)
        dst = os.path.join(dstdir, os.path.basename(args.downstream_json))
        try:
            if os.path.abspath(dst) != os.path.abspath(args.downstream_json):
                shutil.copy2(args.downstream_json, dst)
        except Exception:
            pass

    if not args.metrics_dir:
        raise ValueError('Нужно указать либо --metrics_dir, либо --experiment_dir.')
    if not args.out_csv:
        raise ValueError('Нужно указать либо --out_csv, либо --experiment_dir.')

    downstream = load_downstream_table(args.downstream_json)

    use_tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else None

    # Метрики = все .npz в metrics_dir
    files = [f for f in os.listdir(args.metrics_dir) if f.endswith(".npz")]
    files.sort()

    chosen = files
    if not chosen:
        raise RuntimeError(f"В {args.metrics_dir} не найдено файлов метрик .npz")

    reports = []

    print("\n" + "=" * 100)
    print(f"ОТЧЁТ ПО ОЦЕНКЕ | protocol={args.eval_protocol}")
    print("=" * 100 + "\n")

    for fname in chosen:
        path = os.path.join(args.metrics_dir, fname)
        metric_file, meta, per_task, summary = eval_one_metric(path, downstream, use_tasks=use_tasks, eval_protocol=args.eval_protocol)
        reports.append((metric_file, meta, per_task, summary))

        print(f"Метрика: {metric_file}")
        print(f"  pairs_total: {summary['pairs_total']}, tasks: {summary['tasks']}")
        print(f"  meta: is_paired={meta.get('is_paired')}, is_symmetric={meta.get('is_symmetric')}, pair_agg={meta.get('pair_agg')}")
        print(f"  Spearman (среднее по задачам): {summary['spearman_mean']:.4f}")
        print(f"  Pearson  (среднее по задачам): {summary['pearson_mean']:.4f}")
        print(f"  Kendall  (среднее по задачам): {summary['kendall_mean']:.4f}")
        print(f"  correct_ratio_adjusted_mean: {summary['correct_ratio_adjusted_mean']:.4f}")
        print("-" * 100)

        # Графики
        if args.plots_dir:
            _maybe_make_plots(path, downstream, plots_dir=args.plots_dir, plots_ext=args.plots_ext, plots_mode=args.plots_mode, use_tasks=use_tasks, eval_protocol=args.eval_protocol)

    save_csv(args.out_csv, reports)
    print(f"\nСохранён CSV: {args.out_csv}")

    if args.out_md:
        save_md(args.out_md, reports)
        print(f"Сохранён MD: {args.out_md}")


if __name__ == "__main__":
    main()