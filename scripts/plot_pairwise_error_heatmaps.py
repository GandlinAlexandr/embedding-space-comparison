from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any, Optional

import matplotlib.pyplot as plt
import numpy as np

# ВАЖНО: запускаем как модуль: python -m scripts.plot_pairwise_error_heatmaps
from configs.metric_configs import short_metric_name as _short_metric_name

VALID_PLOT_EXTS = ("png", "pdf", "svg")


# ============================================================
# Utility
# ============================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def iter_plot_paths(path: Path, plots_exts: Iterable[str]) -> List[Path]:
    return [path.with_suffix(f".{ext}") for ext in plots_exts]


def save_figure_variants(
    fig: plt.Figure,
    out_path: Path,
    plots_exts: Iterable[str],
) -> List[Path]:
    out_paths = iter_plot_paths(out_path, plots_exts)
    for save_path in out_paths:
        fig.savefig(save_path, bbox_inches="tight")
    return out_paths


def pretty_metric_name(name: str) -> str:
    # Single-метрики: полные читаемые названия
    single_mapping = {
        "stable_rank": "Стабильный ранг",
        "rankme": "RankMe",
        "coherence": "Когерентность",
        "pseudo_condition_number": "Псевдо-число обусловленности",
        "pseudo_cond": "Псевдо-число обусловленности",
        "nesum": "NESum",
        "self_cluster": "SelfCluster",
        "alpha_req": "α-ReQ",
    }
    if name in single_mapping:
        return single_mapping[name]

    # Pairwise-метрики: делегируем в configs.metric_configs.short_metric_name.
    return _short_metric_name(name)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rankdata_average(x: np.ndarray) -> np.ndarray:
    """
    Аналог scipy.stats.rankdata(method='average'), но без scipy.
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)

    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    return ranks


def zscore_population(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    sigma = np.std(x, ddof=0)
    if sigma == 0 or not np.isfinite(sigma):
        return np.zeros_like(x)
    return (x - mu) / sigma


def finite_pair_arrays(
    x: Iterable[float], y: Iterable[float]
) -> Tuple[np.ndarray, np.ndarray]:
    xx = np.asarray(list(x), dtype=float)
    yy = np.asarray(list(y), dtype=float)
    valid = np.isfinite(xx) & np.isfinite(yy)
    return xx[valid], yy[valid]


def corr_from_vectors(x: np.ndarray, y: np.ndarray, corr_type: str) -> float:
    """
    Pearson или Spearman без scipy.
    Pearson: mean(z(x) * z(y))
    Spearman: Pearson по average-ranks
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return float("nan")

    if corr_type == "pearson":
        x_work = x
        y_work = y
    elif corr_type == "spearman":
        x_work = rankdata_average(x)
        y_work = rankdata_average(y)
    else:
        raise ValueError(f"Unknown corr_type: {corr_type}")

    zx = zscore_population(x_work)
    zy = zscore_population(y_work)

    if zx.size == 0 or zy.size == 0:
        return float("nan")

    return float(np.mean(zx * zy))


def safe_nanmean(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return float("nan")
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


# ============================================================
# Downstream loading
# ============================================================


def load_downstream_scores(
    path: Path, task_name: str | None = None
) -> Dict[str, float]:
    """
    Поддерживает форматы:
        1) {"resnet18": 0.81, ...}
        2) {"scores": {"resnet18": 0.81, ...}}
        3) {"resnet18": {"main": 0.81}, ...}
    """
    obj = load_json(path)

    # Вариант 1: {model: scalar}
    if isinstance(obj, dict) and all(isinstance(v, (int, float)) for v in obj.values()):
        return {str(k): float(v) for k, v in obj.items()}

    # Вариант 2: {"scores": {model: scalar}}
    if isinstance(obj, dict) and "scores" in obj and isinstance(obj["scores"], dict):
        sub = obj["scores"]
        if all(isinstance(v, (int, float)) for v in sub.values()):
            return {str(k): float(v) for k, v in sub.items()}

    # Вариант 3: {model: {task: score}}
    if isinstance(obj, dict):
        out: Dict[str, float] = {}
        task = task_name if task_name is not None else "main"

        for model_name, value in obj.items():
            if isinstance(value, dict):
                if task in value and isinstance(value[task], (int, float)):
                    out[str(model_name)] = float(value[task])
                elif len(value) == 1:
                    only_val = next(iter(value.values()))
                    if isinstance(only_val, (int, float)):
                        out[str(model_name)] = float(only_val)

        if out:
            return out

    raise ValueError(
        f"Не удалось распарсить downstream_json: {path}\n"
        f"Ожидался формат {{model: {{task: score}}}} или {{model: score}}."
    )


# ============================================================
# Single metrics loading
# ============================================================


def load_single_metric_scalar(path: Path) -> float:
    """
    Поддерживает JSON:
        {
          "metric_name": "...",
          "model_name": "...",
          "value": 123.456,
          ...
        }

    И fallback для npz, если вдруг встретится.
    """
    if path.suffix.lower() == ".json":
        obj = load_json(path)
        if isinstance(obj, dict) and "value" in obj:
            return float(obj["value"])
        raise ValueError(f"В JSON нет ключа 'value': {path}")

    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        for key in ["value", "score", "metric", "result", "arr_0"]:
            if key in data:
                arr = np.asarray(data[key])
                if arr.size == 1:
                    return float(arr.reshape(-1)[0])
        for key in data.files:
            arr = np.asarray(data[key])
            if arr.size == 1:
                return float(arr.reshape(-1)[0])
        raise ValueError(f"Не удалось извлечь скаляр из npz: {path}")

    raise ValueError(f"Неподдерживаемый формат single-metric файла: {path}")


def collect_single_metric_values(
    single_metrics_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """
    Ожидается структура:
        single_metrics/
            stable_rank/
                resnet18.json
                resnet50.json
                ...
            rankme/
                ...
    """
    if not single_metrics_dir.exists():
        raise FileNotFoundError(
            f"Папка single_metrics_dir не найдена: {single_metrics_dir}"
        )

    result: Dict[str, Dict[str, float]] = {}

    metric_dirs = sorted([p for p in single_metrics_dir.iterdir() if p.is_dir()])
    if not metric_dirs:
        raise ValueError(
            f"В {single_metrics_dir} не найдено поддиректорий метрик. "
            f"Ожидалась структура single_metrics/<metric_name>/<model>.json"
        )

    for metric_dir in metric_dirs:
        metric_name = metric_dir.name
        metric_values: Dict[str, float] = {}

        files = sorted(list(metric_dir.glob("*.json")) + list(metric_dir.glob("*.npz")))
        for fp in files:
            model_name = fp.stem
            metric_values[model_name] = load_single_metric_scalar(fp)

        if metric_values:
            result[metric_name] = metric_values

    if not result:
        raise ValueError(
            f"Не удалось загрузить ни одной single-метрики из {single_metrics_dir}"
        )

    return result


# ============================================================
# Pairwise metrics loading
# ============================================================


def load_pairwise_metric_npz(
    path: Path,
) -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    """
    Поддерживает форматы:
      - model_names + matrix
      - model_names + scores
      - meta_json (опционально)
    """
    data = np.load(path, allow_pickle=True)

    if "model_names" not in data.files:
        raise KeyError(f"В {path} отсутствует 'model_names'. Ключи: {data.files}")
    model_names = list(data["model_names"].tolist())

    if "matrix" in data.files:
        mat = np.asarray(data["matrix"], dtype=float)
    elif "scores" in data.files:
        mat = np.asarray(data["scores"], dtype=float)
    else:
        raise KeyError(f"В {path} нет ни 'matrix', ни 'scores'. Ключи: {data.files}")

    meta: Dict[str, Any] = {}
    if "meta_json" in data.files:
        meta_json = data["meta_json"]
        meta_json = (
            meta_json.item()
            if getattr(meta_json, "shape", None) == ()
            else meta_json.tolist()
        )

        if isinstance(meta_json, str):
            try:
                meta = json.loads(meta_json)
            except Exception:
                meta = {}
        elif isinstance(meta_json, dict):
            meta = meta_json

    return model_names, mat, meta


def collect_pairwise_metric_matrices(
    pairwise_metrics_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """
    Ожидается структура:
        metric_matrices/
            metric1.npz
            metric2.npz
            ...
    """
    if not pairwise_metrics_dir.exists():
        raise FileNotFoundError(
            f"Папка pairwise_metrics_dir не найдена: {pairwise_metrics_dir}"
        )

    result: Dict[str, Dict[str, Any]] = {}

    files = sorted(pairwise_metrics_dir.glob("*.npz"))
    if not files:
        raise ValueError(f"В {pairwise_metrics_dir} не найдено .npz-файлов")

    for fp in files:
        model_names, mat, meta = load_pairwise_metric_npz(fp)
        metric_name = fp.stem
        result[metric_name] = {
            "model_names": model_names,
            "matrix": mat,
            "meta": meta,
            "path": fp,
        }

    return result


# ============================================================
# Family map
# ============================================================


def load_family_map(path: Path) -> Dict[str, str]:
    """
    Ожидается JSON:
    {
      "resnet18": "resnet",
      "resnet50": "resnet",
      "wide_resnet50_2": "resnet",
      "vgg16": "vgg",
      "vit_b_16": "vit"
    }
    """
    obj = load_json(path)
    if not isinstance(obj, dict):
        raise ValueError("family_map_json должен быть словарём вида {model: family}")
    out: Dict[str, str] = {}
    for k, v in obj.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError("family_map_json должен содержать только строки")
        out[k] = v
    return out


def ordered_models(
    available_models: set[str],
    model_order: Optional[List[str]],
) -> List[str]:
    if model_order:
        models = [m for m in model_order if m in available_models]
        missing = [m for m in model_order if m not in available_models]
        if missing:
            print(
                f"[warn] Эти модели отсутствуют в данных и будут пропущены: {missing}"
            )

        rest = sorted([m for m in available_models if m not in set(models)])
        return models + rest

    return sorted(available_models)


# ============================================================
# Pairwise matrices
# ============================================================


def build_pairwise_matrix(
    values: Dict[str, float], models: List[str], protocol: str
) -> np.ndarray:
    arr = np.array([float(values[m]) for m in models], dtype=float)
    n = len(models)
    mat = np.full((n, n), np.nan, dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = arr[i] - arr[j]
            if protocol == "signed":
                mat[i, j] = diff
            elif protocol == "abs":
                mat[i, j] = abs(diff)
            else:
                raise ValueError(f"Unknown protocol: {protocol}")

    return mat


def align_pairwise_matrix_to_models(
    model_names_src: List[str],
    mat_src: np.ndarray,
    models_dst: List[str],
) -> np.ndarray:
    """
    Переставляет/подвыбирает строки-столбцы pairwise-матрицы под порядок models_dst.
    """
    idx = {m: i for i, m in enumerate(model_names_src)}
    n = len(models_dst)
    out = np.full((n, n), np.nan, dtype=float)

    for i, mi in enumerate(models_dst):
        if mi not in idx:
            continue
        for j, mj in enumerate(models_dst):
            if mj not in idx:
                continue
            out[i, j] = float(mat_src[idx[mi], idx[mj]])

    return out


def enforce_matrix_protocol(mat: np.ndarray, protocol: str) -> np.ndarray:
    """
    Приводит уже загруженную pairwise-матрицу к нужному протоколу,
    не пересчитывая её заново из скалярных значений моделей.

    protocol == "signed":
        Антисимметризация: out[i,j] = (mat[i,j] - mat[j,i]) / 2
        Гарантирует out[i,j] == -out[j,i].
        Если исходная матрица уже антисимметрична — значения не меняются.

    protocol == "abs":
        Симметризация + модуль: out[i,j] = |( mat[i,j] + mat[j,i] )| / 2
        Гарантирует out[i,j] == out[j,i] >= 0.
        Если исходная матрица уже симметрична — значения не меняются (кроме знака).

    NaN обрабатывается аккуратно: если хотя бы одна сторона пары — nan,
    результат тоже nan.
    """
    if protocol == "signed":
        return (mat - mat.T) / 2.0
    elif protocol == "abs":
        sym = (mat + mat.T) / 2.0
        return np.abs(sym)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")


# ============================================================
# Family-wise statistics
# ============================================================


def iter_family_subset_pairs(
    models: List[str],
    family_of: Dict[str, str],
    fam_a: str,
    fam_b: str,
    protocol: str,
) -> Iterable[Tuple[int, int]]:
    """
    Возвращает индексы пар для подмножества семейств.

    protocol == "abs":
        используем неупорядоченные пары i < j

    protocol == "signed":
        используем упорядоченные пары i != j,
        поэтому матрица family x family может быть несимметричной.
    """
    n = len(models)

    if protocol == "abs":
        for i in range(n):
            fi = family_of[models[i]]
            for j in range(i + 1, n):
                fj = family_of[models[j]]

                same_direction = fi == fam_a and fj == fam_b
                reverse_direction = fi == fam_b and fj == fam_a

                if same_direction or reverse_direction:
                    yield i, j

    elif protocol == "signed":
        for i in range(n):
            fi = family_of[models[i]]
            for j in range(n):
                if i == j:
                    continue
                fj = family_of[models[j]]

                if fi == fam_a and fj == fam_b:
                    yield i, j
    else:
        raise ValueError(f"Unknown protocol: {protocol}")


def compute_family_subset_matrices(
    metric_mat: np.ndarray,
    target_mat: np.ndarray,
    models: List[str],
    family_of: Dict[str, str],
    corr_type: str,
    protocol: str,
    min_pairs: int,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Для каждой пары семейств (fa, fb) считает:
      - corr(metric, target) по соответствующему подмножеству пар
      - среднее target по этому подмножеству
      - число пар

    Возвращает:
      families, corr_mat, target_mean_mat, count_mat
    """
    families = sorted({family_of[m] for m in models})
    nf = len(families)

    corr_mat = np.full((nf, nf), np.nan, dtype=float)
    target_mean_mat = np.full((nf, nf), np.nan, dtype=float)
    count_mat = np.zeros((nf, nf), dtype=int)

    for a, fa in enumerate(families):
        for b, fb in enumerate(families):
            metric_vals: List[float] = []
            target_vals: List[float] = []

            for i, j in iter_family_subset_pairs(
                models=models,
                family_of=family_of,
                fam_a=fa,
                fam_b=fb,
                protocol=protocol,
            ):
                mv = metric_mat[i, j]
                tv = target_mat[i, j]
                if np.isfinite(mv) and np.isfinite(tv):
                    metric_vals.append(float(mv))
                    target_vals.append(float(tv))

            x, y = finite_pair_arrays(metric_vals, target_vals)
            count_mat[a, b] = int(len(x))
            target_mean_mat[a, b] = safe_nanmean(y)

            if len(x) >= min_pairs:
                corr_mat[a, b] = corr_from_vectors(x, y, corr_type=corr_type)

    return families, corr_mat, target_mean_mat, count_mat


def compute_global_corr(
    metric_mat: np.ndarray,
    target_mat: np.ndarray,
    protocol: str,
    corr_type: str,
) -> Tuple[float, int]:
    xs: List[float] = []
    ys: List[float] = []

    n = metric_mat.shape[0]
    if protocol == "abs":
        for i in range(n):
            for j in range(i + 1, n):
                mv = metric_mat[i, j]
                tv = target_mat[i, j]
                if np.isfinite(mv) and np.isfinite(tv):
                    xs.append(float(mv))
                    ys.append(float(tv))
    elif protocol == "signed":
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                mv = metric_mat[i, j]
                tv = target_mat[i, j]
                if np.isfinite(mv) and np.isfinite(tv):
                    xs.append(float(mv))
                    ys.append(float(tv))
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    x, y = finite_pair_arrays(xs, ys)
    return corr_from_vectors(x, y, corr_type=corr_type), int(len(x))


def compute_leave_one_family_out(
    metric_mat: np.ndarray,
    target_mat: np.ndarray,
    models: List[str],
    family_of: Dict[str, str],
    protocol: str,
    corr_type: str,
) -> List[Dict[str, Any]]:
    """
    Для каждого семейства F:
      - убираем все пары, где участвует F
      - считаем corr на оставшихся парах
      - сравниваем с полной corr
    """
    full_corr, full_n = compute_global_corr(metric_mat, target_mat, protocol, corr_type)
    families = sorted({family_of[m] for m in models})

    rows: List[Dict[str, Any]] = []

    for fam in families:
        xs: List[float] = []
        ys: List[float] = []

        n = len(models)
        if protocol == "abs":
            for i in range(n):
                fi = family_of[models[i]]
                for j in range(i + 1, n):
                    fj = family_of[models[j]]
                    if fi == fam or fj == fam:
                        continue
                    mv = metric_mat[i, j]
                    tv = target_mat[i, j]
                    if np.isfinite(mv) and np.isfinite(tv):
                        xs.append(float(mv))
                        ys.append(float(tv))
        elif protocol == "signed":
            for i in range(n):
                fi = family_of[models[i]]
                for j in range(n):
                    if i == j:
                        continue
                    fj = family_of[models[j]]
                    if fi == fam or fj == fam:
                        continue
                    mv = metric_mat[i, j]
                    tv = target_mat[i, j]
                    if np.isfinite(mv) and np.isfinite(tv):
                        xs.append(float(mv))
                        ys.append(float(tv))
        else:
            raise ValueError(f"Unknown protocol: {protocol}")

        x, y = finite_pair_arrays(xs, ys)
        corr_wo = corr_from_vectors(x, y, corr_type=corr_type)

        rows.append(
            {
                "family": fam,
                "corr_full": full_corr,
                "n_full": full_n,
                "corr_without_family": corr_wo,
                "n_without_family": int(len(x)),
                "delta_corr": (
                    corr_wo - full_corr
                    if np.isfinite(corr_wo) and np.isfinite(full_corr)
                    else float("nan")
                ),
            }
        )

    return rows


# ============================================================
# Plotting
# ============================================================


def setup_matplotlib() -> None:
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 180
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 10


def draw_heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    labels_x: List[str],
    labels_y: List[str],
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = False,
    annotate_text: np.ndarray | None = None,
):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(labels_x)))
    ax.set_yticks(range(len(labels_y)))
    ax.set_xticklabels(labels_x, rotation=45, ha="right")
    ax.set_yticklabels(labels_y)
    ax.set_title(title)

    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isfinite(mat[i, j]):
                    continue

                if annotate_text is not None:
                    txt = str(annotate_text[i, j])
                else:
                    txt = f"{mat[i, j]:.2f}"

                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    return im


def make_corr_annotation_text(
    corr_mat: np.ndarray, count_mat: np.ndarray
) -> np.ndarray:
    out = np.empty(corr_mat.shape, dtype=object)
    for i in range(corr_mat.shape[0]):
        for j in range(corr_mat.shape[1]):
            if np.isfinite(corr_mat[i, j]):
                out[i, j] = f"{corr_mat[i, j]:.2f}\n(n={int(count_mat[i, j])})"
            else:
                if int(count_mat[i, j]) > 0:
                    out[i, j] = f"nan\n(n={int(count_mat[i, j])})"
                else:
                    out[i, j] = ""
    return out


def make_value_count_annotation_text(
    value_mat: np.ndarray, count_mat: np.ndarray, fmt: str
) -> np.ndarray:
    out = np.empty(value_mat.shape, dtype=object)
    for i in range(value_mat.shape[0]):
        for j in range(value_mat.shape[1]):
            if np.isfinite(value_mat[i, j]):
                out[i, j] = (
                    f"{format(value_mat[i, j], fmt)}\n(n={int(count_mat[i, j])})"
                )
            else:
                if int(count_mat[i, j]) > 0:
                    out[i, j] = f"nan\n(n={int(count_mat[i, j])})"
                else:
                    out[i, j] = ""
    return out


def save_family_target_heatmap(
    families: List[str],
    target_mean_mat: np.ndarray,
    count_mat: np.ndarray,
    out_path: Path,
    title_prefix: str,
    protocol: str,
    annotate: bool,
    plots_exts: Iterable[str] = ("png",),
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.6))

    if protocol == "signed":
        target_label = "mean(Δacc)"
        vmax = (
            float(np.nanmax(np.abs(target_mean_mat)))
            if np.isfinite(target_mean_mat).any()
            else 1.0
        )
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        vmin = -vmax
        cmap = "coolwarm"
    else:
        target_label = "mean(|Δacc|)"
        vmax = (
            float(np.nanmax(target_mean_mat))
            if np.isfinite(target_mean_mat).any()
            else 1.0
        )
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        vmin = 0.0
        cmap = "viridis"

    annotate_text = (
        make_value_count_annotation_text(target_mean_mat, count_mat, ".3f")
        if annotate
        else None
    )

    draw_heatmap(
        ax=ax,
        mat=target_mean_mat,
        labels_x=families,
        labels_y=families,
        title=f"{title_prefix} | средняя целевая величина по семействам",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annotate=annotate,
        annotate_text=annotate_text,
    )
    cbar = fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(target_label)

    fig.tight_layout()
    save_figure_variants(fig, out_path, plots_exts)
    plt.close(fig)


def save_family_count_heatmap(
    families: List[str],
    count_mat: np.ndarray,
    out_path: Path,
    title_prefix: str,
    annotate: bool,
    plots_exts: Iterable[str] = ("png",),
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.6))

    vmax = int(np.nanmax(count_mat)) if count_mat.size else 1
    if vmax <= 0:
        vmax = 1

    annotate_text = np.empty(count_mat.shape, dtype=object)
    for i in range(count_mat.shape[0]):
        for j in range(count_mat.shape[1]):
            annotate_text[i, j] = str(int(count_mat[i, j])) if annotate else ""

    draw_heatmap(
        ax=ax,
        mat=count_mat.astype(float),
        labels_x=families,
        labels_y=families,
        title=f"{title_prefix} | количество пар по семействам",
        cmap="Blues",
        vmin=0.0,
        vmax=float(vmax),
        annotate=annotate,
        annotate_text=annotate_text if annotate else None,
    )
    cbar = fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("n_pairs")

    fig.tight_layout()
    save_figure_variants(fig, out_path, plots_exts)
    plt.close(fig)


def _draw_metric_group_grid(
    fig: plt.Figure,
    axes: np.ndarray,
    metric_names: List[str],
    family_corr_mats: Dict[str, np.ndarray],
    family_count_mats: Dict[str, np.ndarray],
    global_corrs: Dict[str, float],
    global_counts: Dict[str, int],
    families: List[str],
    corr_label: str,
    vmin: float,
    vmax: float,
    annotate: bool,
    protocol: str = "abs",
) -> Any:
    """Заполняет подграфики для одной группы метрик. Возвращает последний AxesImage."""
    last_im = None
    for ax, metric_name in zip(axes.flat, metric_names):
        corr_mat = family_corr_mats[metric_name]
        count_mat = family_count_mats[metric_name]
        annotate_text = (
            make_corr_annotation_text(corr_mat, count_mat) if annotate else None
        )
        title_text = (
            f"{pretty_metric_name(metric_name)}\n"
            f"global {corr_label} = {global_corrs[metric_name]:.3f}"
            f" (n={global_counts[metric_name]})"
        )
        im = draw_heatmap(
            ax=ax,
            mat=corr_mat,
            labels_x=families,
            labels_y=families,
            title=title_text,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            annotate=annotate,
            annotate_text=annotate_text,
        )
        last_im = im
    for ax in axes.flat[len(metric_names) :]:
        ax.axis("off")
    return last_im


def save_family_correlation_grid(
    family_corr_mats: Dict[str, np.ndarray],
    family_count_mats: Dict[str, np.ndarray],
    global_corrs: Dict[str, float],
    global_counts: Dict[str, int],
    families: List[str],
    out_path: Path,
    title_prefix: str,
    corr_type: str,
    annotate: bool,
    metric_sources: Optional[Dict[str, str]] = None,
    protocol: str = "abs",
    plots_exts: Iterable[str] = ("png",),
) -> None:
    sources = metric_sources or {}

    single_names = sorted(
        m for m in family_corr_mats if sources.get(m, "single") == "single"
    )
    pairwise_names = sorted(
        m for m in family_corr_mats if sources.get(m, "single") == "pairwise"
    )

    # Если источники не различаются — рисуем один grid без разделения
    if not single_names or not pairwise_names:
        all_names = single_names or pairwise_names
        _save_flat_corr_grid(
            family_corr_mats=family_corr_mats,
            family_count_mats=family_count_mats,
            global_corrs=global_corrs,
            global_counts=global_counts,
            families=families,
            metric_names=all_names,
            out_path=out_path,
            title_prefix=title_prefix,
            corr_type=corr_type,
            annotate=annotate,
            protocol=protocol,
            plots_exts=plots_exts,
        )
        return

    NCOLS = 3
    corr_label = "Пирсон" if corr_type == "pearson" else "Спирмен"

    # Общий диапазон цветовой шкалы по всем метрикам
    all_vals = []
    for m in family_corr_mats:
        vals = family_corr_mats[m][np.isfinite(family_corr_mats[m])]
        if vals.size:
            all_vals.append(vals)
    if all_vals:
        vmax = float(np.nanmax(np.abs(np.concatenate(all_vals))))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
    else:
        vmax = 1.0
    vmin = -vmax

    s_rows = math.ceil(len(single_names) / NCOLS)
    p_rows = math.ceil(len(pairwise_names) / NCOLS)

    # Высота: на каждую строку подграфиков 4.6, плюс 0.45 на section-заголовок,
    # плюс 0.7 на общий suptitle, плюс 0.5 зазор между секциями
    row_h = 4.6
    sec_h = 0.45
    gap_h = 0.5
    sup_h = 0.7
    total_h = sup_h + sec_h + s_rows * row_h + gap_h + sec_h + p_rows * row_h

    fig = plt.figure(figsize=(5.8 * NCOLS + 1.2, total_h))

    # Нормируем высоты строк в figure-координатах
    fig_h = total_h
    # Вертикальные позиции секций (сверху вниз, в figure-координатах)
    # suptitle занимает sup_h сверху
    single_top = 1.0 - sup_h / fig_h
    single_sec_h_frac = sec_h / fig_h
    single_grid_h_frac = (s_rows * row_h) / fig_h
    pairwise_top = single_top - single_sec_h_frac - single_grid_h_frac - gap_h / fig_h
    pairwise_sec_h_frac = sec_h / fig_h
    pairwise_grid_h_frac = (p_rows * row_h) / fig_h

    # --- Section label: single ---
    fig.text(
        0.01,
        single_top - single_sec_h_frac / 2,
        "── single metrics ──────────────────────────",
        fontsize=10,
        color="#2266aa",
        va="center",
        fontstyle="italic",
    )

    # --- Single grid ---
    s_axes = np.array(
        [
            fig.add_axes(
                [
                    (c * (1.0 / NCOLS)) * 0.88 + 0.04,
                    single_top
                    - single_sec_h_frac
                    - (r + 1) * single_grid_h_frac / s_rows
                    + 0.01,
                    0.88 / NCOLS - 0.03,
                    single_grid_h_frac / s_rows - 0.04,
                ]
            )
            for r in range(s_rows)
            for c in range(NCOLS)
        ]
    ).reshape(s_rows, NCOLS)

    last_im = _draw_metric_group_grid(
        fig=fig,
        axes=s_axes,
        metric_names=single_names,
        family_corr_mats=family_corr_mats,
        family_count_mats=family_count_mats,
        global_corrs=global_corrs,
        global_counts=global_counts,
        families=families,
        corr_label=corr_label,
        vmin=vmin,
        vmax=vmax,
        annotate=annotate,
        protocol=protocol,
    )

    # --- Section label: pairwise ---
    fig.text(
        0.01,
        pairwise_top - pairwise_sec_h_frac / 2,
        "── pairwise metrics ────────────────────────",
        fontsize=10,
        color="#aa6622",
        va="center",
        fontstyle="italic",
    )

    # --- Pairwise grid ---
    p_axes = np.array(
        [
            fig.add_axes(
                [
                    (c * (1.0 / NCOLS)) * 0.88 + 0.04,
                    pairwise_top
                    - pairwise_sec_h_frac
                    - (r + 1) * pairwise_grid_h_frac / p_rows
                    + 0.01,
                    0.88 / NCOLS - 0.03,
                    pairwise_grid_h_frac / p_rows - 0.04,
                ]
            )
            for r in range(p_rows)
            for c in range(NCOLS)
        ]
    ).reshape(p_rows, NCOLS)

    last_im2 = _draw_metric_group_grid(
        fig=fig,
        axes=p_axes,
        metric_names=pairwise_names,
        family_corr_mats=family_corr_mats,
        family_count_mats=family_count_mats,
        global_corrs=global_corrs,
        global_counts=global_counts,
        families=families,
        corr_label=corr_label,
        vmin=vmin,
        vmax=vmax,
        annotate=annotate,
        protocol=protocol,
    )

    final_im = last_im2 or last_im

    x_label = "m(i→j) − m(j→i)" if protocol == "signed" else "½(m(i→j) + m(j→i))"
    y_label = "acc_i − acc_j" if protocol == "signed" else "|acc_i − acc_j|"

    fig.suptitle(
        f"{title_prefix} | {corr_label}({x_label}, {y_label}) по подмножествам пар семейств",
        fontsize=13,
    )

    if final_im is not None:
        cbar_ax = fig.add_axes([0.92, 0.08, 0.016, 0.82])
        cbar = fig.colorbar(final_im, cax=cbar_ax)
        cbar.set_label(f"{corr_label} corr", rotation=90, labelpad=10)

    save_figure_variants(fig, out_path, plots_exts)
    plt.close(fig)


def _save_flat_corr_grid(
    family_corr_mats: Dict[str, np.ndarray],
    family_count_mats: Dict[str, np.ndarray],
    global_corrs: Dict[str, float],
    global_counts: Dict[str, int],
    families: List[str],
    metric_names: List[str],
    out_path: Path,
    title_prefix: str,
    corr_type: str,
    annotate: bool,
    protocol: str = "abs",
    plots_exts: Iterable[str] = ("png",),
) -> None:
    """Fallback: один flat grid без разделения на секции."""
    n_metrics = len(metric_names)
    ncols = min(3, n_metrics)
    nrows = math.ceil(n_metrics / ncols)

    all_vals = []
    for m in metric_names:
        vals = family_corr_mats[m][np.isfinite(family_corr_mats[m])]
        if vals.size:
            all_vals.append(vals)
    if all_vals:
        vmax = float(np.nanmax(np.abs(np.concatenate(all_vals))))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
    else:
        vmax = 1.0
    vmin = -vmax

    corr_label = "Пирсон" if corr_type == "pearson" else "Спирмен"
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.8 * ncols, 5.0 * nrows),
        squeeze=False,
    )
    last_im = _draw_metric_group_grid(
        fig=fig,
        axes=axes,
        metric_names=metric_names,
        family_corr_mats=family_corr_mats,
        family_count_mats=family_count_mats,
        global_corrs=global_corrs,
        global_counts=global_counts,
        families=families,
        corr_label=corr_label,
        vmin=vmin,
        vmax=vmax,
        annotate=annotate,
        protocol=protocol,
    )
    x_label = "m(i→j) − m(j→i)" if protocol == "signed" else "½(m(i→j) + m(j→i))"
    y_label = "acc_i − acc_j" if protocol == "signed" else "|acc_i − acc_j|"
    fig.suptitle(
        f"{title_prefix} | {corr_label}({x_label}, {y_label}) по подмножествам пар семейств",
        fontsize=13,
    )
    fig.subplots_adjust(right=0.88, top=0.93, wspace=0.28, hspace=0.45)
    if last_im is not None:
        cbar_ax = fig.add_axes([0.90, 0.18, 0.018, 0.62])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label(f"{corr_label} corr", rotation=90, labelpad=10)
    save_figure_variants(fig, out_path, plots_exts)
    plt.close(fig)


def save_leave_one_family_out_barplot(
    metric_rows: Dict[str, List[Dict[str, Any]]],
    out_path: Path,
    title_prefix: str,
    corr_type: str,
    metric_sources: Optional[Dict[str, str]] = None,
    plots_exts: Iterable[str] = ("png",),
) -> None:
    sources = metric_sources or {}
    corr_label = "Пирсон" if corr_type == "pearson" else "Спирмен"

    single_names = sorted(
        m for m in metric_rows if sources.get(m, "single") == "single"
    )
    pairwise_names = sorted(
        m for m in metric_rows if sources.get(m, "single") == "pairwise"
    )

    # Если только один вид источников — рисуем без разделения
    if not single_names or not pairwise_names:
        all_names = single_names or pairwise_names
        n = len(all_names)
        ncols = min(3, n)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5.6 * ncols, 4.4 * nrows),
            squeeze=False,
        )
        _fill_loo_axes(axes, all_names, metric_rows, corr_label)
        for ax in axes.flat[n:]:
            ax.axis("off")
        fig.suptitle(
            f"{title_prefix} | leave-one-family-out ({corr_label})",
            fontsize=13,
        )
        fig.subplots_adjust(top=0.93, hspace=0.55, wspace=0.35)
        save_figure_variants(fig, out_path, plots_exts)
        plt.close(fig)
        return

    NCOLS = 3
    bar_h = 3.6  # высота одной строки барплотов
    sec_h = 0.40  # высота section-заголовка
    gap_h = 0.45  # зазор между секциями
    sup_h = 0.65  # место под общий заголовок

    s_rows = math.ceil(len(single_names) / NCOLS)
    p_rows = math.ceil(len(pairwise_names) / NCOLS)
    total_h = sup_h + sec_h + s_rows * bar_h + gap_h + sec_h + p_rows * bar_h

    fig = plt.figure(figsize=(5.6 * NCOLS, total_h))

    fh = total_h
    single_top = 1.0 - sup_h / fh
    s_sec_frac = sec_h / fh
    s_grid_frac = (s_rows * bar_h) / fh
    pw_top = single_top - s_sec_frac - s_grid_frac - gap_h / fh
    p_sec_frac = sec_h / fh
    p_grid_frac = (p_rows * bar_h) / fh

    # Section label: single
    fig.text(
        0.01,
        single_top - s_sec_frac / 2,
        "── single metrics ──────────────────────────",
        fontsize=10,
        color="#2266aa",
        va="center",
        fontstyle="italic",
    )

    s_axes = np.array(
        [
            fig.add_axes(
                [
                    c / NCOLS * 0.92 + 0.04,
                    single_top - s_sec_frac - (r + 1) * s_grid_frac / s_rows + 0.02,
                    0.92 / NCOLS - 0.04,
                    s_grid_frac / s_rows - 0.05,
                ]
            )
            for r in range(s_rows)
            for c in range(NCOLS)
        ]
    ).reshape(s_rows, NCOLS)
    _fill_loo_axes(s_axes, single_names, metric_rows, corr_label)
    for ax in s_axes.flat[len(single_names) :]:
        ax.axis("off")

    # Section label: pairwise
    fig.text(
        0.01,
        pw_top - p_sec_frac / 2,
        "── pairwise metrics ────────────────────────",
        fontsize=10,
        color="#aa6622",
        va="center",
        fontstyle="italic",
    )

    p_axes = np.array(
        [
            fig.add_axes(
                [
                    c / NCOLS * 0.92 + 0.04,
                    pw_top - p_sec_frac - (r + 1) * p_grid_frac / p_rows + 0.02,
                    0.92 / NCOLS - 0.04,
                    p_grid_frac / p_rows - 0.05,
                ]
            )
            for r in range(p_rows)
            for c in range(NCOLS)
        ]
    ).reshape(p_rows, NCOLS)
    _fill_loo_axes(p_axes, pairwise_names, metric_rows, corr_label)
    for ax in p_axes.flat[len(pairwise_names) :]:
        ax.axis("off")

    fig.suptitle(
        f"{title_prefix} | leave-one-family-out ({corr_label})",
        fontsize=13,
    )
    save_figure_variants(fig, out_path, plots_exts)
    plt.close(fig)


def _fill_loo_axes(
    axes: np.ndarray,
    metric_names: List[str],
    metric_rows: Dict[str, List[Dict[str, Any]]],
    corr_label: str,
) -> None:
    """Заполняет барплоты LOO для переданного списка метрик."""
    for ax, metric_name in zip(axes.flat, metric_names):
        rows = metric_rows[metric_name]
        families = [r["family"] for r in rows]
        vals = [r["delta_corr"] for r in rows]
        x = np.arange(len(families))
        ax.bar(x, vals)
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(families, rotation=40, ha="right")
        ax.set_ylabel("corr_without − corr_full", fontsize=8)
        ax.set_title(pretty_metric_name(metric_name))


# ============================================================
# CSV summary
# ============================================================


def save_global_summary_csv(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    fieldnames = [
        "metric_name",
        "metric_source",
        "protocol",
        "corr_type",
        "global_corr",
        "global_n_pairs",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_family_summary_csv(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    fieldnames = [
        "metric_name",
        "metric_source",
        "protocol",
        "corr_type",
        "family_a",
        "family_b",
        "subset_corr",
        "subset_target_mean",
        "subset_n_pairs",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_leave_one_family_out_csv(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    fieldnames = [
        "metric_name",
        "metric_source",
        "protocol",
        "corr_type",
        "family",
        "corr_full",
        "n_full",
        "corr_without_family",
        "n_without_family",
        "delta_corr",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Строит family-wise heatmap'ы корреляции: "
            "для каждой пары семейств (family_a, family_b) считает корреляцию "
            "между значениями метрики и target на соответствующем подмножестве пар."
        )
    )
    parser.add_argument("--downstream_json", type=Path, required=True)
    parser.add_argument("--family_map_json", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)

    parser.add_argument(
        "--single_metrics_dir",
        type=Path,
        default=None,
        help="Папка single_metrics/<metric>/<model>.json",
    )
    parser.add_argument(
        "--pairwise_metrics_dir",
        type=Path,
        default=None,
        help="Папка metric_matrices/*.npz",
    )

    parser.add_argument(
        "--protocol",
        choices=["signed", "abs"],
        default="abs",
        help="signed -> Δu vs Δacc, abs -> |Δu| vs |Δacc|",
    )
    parser.add_argument(
        "--corr_type",
        choices=["spearman", "pearson"],
        default="spearman",
        help="Какую корреляцию считать внутри family-subset",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="main",
        help="Какой task брать из downstream_json формата {model: {task: score}}",
    )
    parser.add_argument(
        "--model_order",
        nargs="*",
        default=None,
        help="Явный порядок моделей",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Например: CIFAR10",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Подписывать значения в ячейках",
    )
    parser.add_argument(
        "--min_pairs",
        type=int,
        default=3,
        help="Минимальное число пар для вычисления subset correlation",
    )
    parser.add_argument(
        "--skip_leave_one_family_out",
        action="store_true",
        help="Не считать leave-one-family-out summary",
    )
    parser.add_argument(
        "--plots_ext",
        type=str,
        default="png",
        help="Одно или несколько расширений графиков через запятую, например png или svg,png.",
    )

    args = parser.parse_args()

    if args.single_metrics_dir is None and args.pairwise_metrics_dir is None:
        parser.error(
            "Нужно указать хотя бы один из аргументов: --single_metrics_dir или --pairwise_metrics_dir"
        )

    return args


# ============================================================
# Main
# ============================================================


def main() -> None:
    args = parse_args()
    args.plots_ext = parse_plots_exts(args.plots_ext)
    setup_matplotlib()
    ensure_dir(args.out_dir)

    downstream_scores = load_downstream_scores(
        args.downstream_json, task_name=args.task_name
    )
    family_map = load_family_map(args.family_map_json)

    single_metrics: Dict[str, Dict[str, float]] = {}
    if args.single_metrics_dir is not None:
        single_metrics = collect_single_metric_values(args.single_metrics_dir)

    pairwise_metrics: Dict[str, Dict[str, Any]] = {}
    if args.pairwise_metrics_dir is not None:
        pairwise_metrics = collect_pairwise_metric_matrices(args.pairwise_metrics_dir)

    available_models = set(downstream_scores.keys())

    if single_metrics:
        common_single_models = set(downstream_scores.keys())
        for metric_vals in single_metrics.values():
            common_single_models &= set(metric_vals.keys())
        if common_single_models:
            available_models |= common_single_models

    if pairwise_metrics:
        pairwise_union_models = set()
        for spec in pairwise_metrics.values():
            pairwise_union_models |= set(spec["model_names"])
        available_models |= pairwise_union_models

    available_models &= set(family_map.keys())
    available_models &= set(downstream_scores.keys())

    if not available_models:
        raise ValueError("Нет общего множества моделей между downstream и family_map")

    models = ordered_models(
        available_models=available_models, model_order=args.model_order
    )

    # Оставляем только модели, которые реально имеют family label и downstream score
    models = [m for m in models if m in family_map and m in downstream_scores]

    if len(models) < 2:
        raise ValueError("Нужно хотя бы две модели после всех пересечений")

    family_of = {m: family_map[m] for m in models}
    title_prefix = args.title.strip() if args.title.strip() else "Experiment"

    target_mat = build_pairwise_matrix(downstream_scores, models, args.protocol)

    global_rows: List[Dict[str, Any]] = []
    family_rows: List[Dict[str, Any]] = []
    loo_rows_all: List[Dict[str, Any]] = []

    family_corr_mats: Dict[str, np.ndarray] = {}
    family_count_mats: Dict[str, np.ndarray] = {}
    global_corrs: Dict[str, float] = {}
    global_counts: Dict[str, int] = {}
    metric_sources: Dict[str, str] = {}  # "single" | "pairwise" для каждой метрики
    leave_one_family_out_by_metric: Dict[str, List[Dict[str, Any]]] = {}

    target_mean_ref: np.ndarray | None = None
    count_ref: np.ndarray | None = None
    families_ref: List[str] | None = None

    # --------------------------------------------------------
    # Single metrics
    # --------------------------------------------------------
    for metric_name, metric_values in sorted(single_metrics.items()):
        metric_models = [m for m in models if m in metric_values]
        if len(metric_models) < 2:
            continue

        # Работаем на общем списке models; где модели отсутствуют — такого здесь уже нет.
        metric_mat = build_pairwise_matrix(metric_values, models, args.protocol)

        families, corr_mat, target_mean_mat, count_mat = compute_family_subset_matrices(
            metric_mat=metric_mat,
            target_mat=target_mat,
            models=models,
            family_of=family_of,
            corr_type=args.corr_type,
            protocol=args.protocol,
            min_pairs=args.min_pairs,
        )

        global_corr, global_n = compute_global_corr(
            metric_mat=metric_mat,
            target_mat=target_mat,
            protocol=args.protocol,
            corr_type=args.corr_type,
        )

        family_corr_mats[metric_name] = corr_mat
        family_count_mats[metric_name] = count_mat
        global_corrs[metric_name] = global_corr
        global_counts[metric_name] = global_n
        metric_sources[metric_name] = "single"

        if families_ref is None:
            families_ref = families
            target_mean_ref = target_mean_mat
            count_ref = count_mat

        global_rows.append(
            {
                "metric_name": metric_name,
                "metric_source": "single",
                "protocol": args.protocol,
                "corr_type": args.corr_type,
                "global_corr": global_corr,
                "global_n_pairs": global_n,
            }
        )

        for i, fa in enumerate(families):
            for j, fb in enumerate(families):
                family_rows.append(
                    {
                        "metric_name": metric_name,
                        "metric_source": "single",
                        "protocol": args.protocol,
                        "corr_type": args.corr_type,
                        "family_a": fa,
                        "family_b": fb,
                        "subset_corr": corr_mat[i, j],
                        "subset_target_mean": target_mean_mat[i, j],
                        "subset_n_pairs": int(count_mat[i, j]),
                    }
                )

        if not args.skip_leave_one_family_out:
            rows = compute_leave_one_family_out(
                metric_mat=metric_mat,
                target_mat=target_mat,
                models=models,
                family_of=family_of,
                protocol=args.protocol,
                corr_type=args.corr_type,
            )
            leave_one_family_out_by_metric[metric_name] = rows

            for row in rows:
                loo_rows_all.append(
                    {
                        "metric_name": metric_name,
                        "metric_source": "single",
                        "protocol": args.protocol,
                        "corr_type": args.corr_type,
                        **row,
                    }
                )

    # --------------------------------------------------------
    # Pairwise metrics
    # --------------------------------------------------------
    for metric_name, spec in sorted(pairwise_metrics.items()):
        src_models = spec["model_names"]
        src_mat = spec["matrix"]

        keep_models = [m for m in models if m in src_models]
        if len(keep_models) < 2:
            continue

        # Строим матрицу в порядке полного списка models.
        # Если какой-то модели нет в src_models, останется nan.
        metric_mat = align_pairwise_matrix_to_models(
            model_names_src=src_models,
            mat_src=src_mat,
            models_dst=models,
        )

        # Приводим загруженную матрицу к протоколу без пересчёта из скалярных значений.
        # signed -> антисимметризация (m[i,j] = -m[j,i])
        # abs    -> симметризация + |·|  (m[i,j] = m[j,i] >= 0)
        metric_mat = enforce_matrix_protocol(metric_mat, args.protocol)

        families, corr_mat, target_mean_mat, count_mat = compute_family_subset_matrices(
            metric_mat=metric_mat,
            target_mat=target_mat,
            models=models,
            family_of=family_of,
            corr_type=args.corr_type,
            protocol=args.protocol,
            min_pairs=args.min_pairs,
        )

        global_corr, global_n = compute_global_corr(
            metric_mat=metric_mat,
            target_mat=target_mat,
            protocol=args.protocol,
            corr_type=args.corr_type,
        )

        family_corr_mats[metric_name] = corr_mat
        family_count_mats[metric_name] = count_mat
        global_corrs[metric_name] = global_corr
        global_counts[metric_name] = global_n
        metric_sources[metric_name] = "pairwise"

        if families_ref is None:
            families_ref = families
            target_mean_ref = target_mean_mat
            count_ref = count_mat

        global_rows.append(
            {
                "metric_name": metric_name,
                "metric_source": "pairwise",
                "protocol": args.protocol,
                "corr_type": args.corr_type,
                "global_corr": global_corr,
                "global_n_pairs": global_n,
            }
        )

        for i, fa in enumerate(families):
            for j, fb in enumerate(families):
                family_rows.append(
                    {
                        "metric_name": metric_name,
                        "metric_source": "pairwise",
                        "protocol": args.protocol,
                        "corr_type": args.corr_type,
                        "family_a": fa,
                        "family_b": fb,
                        "subset_corr": corr_mat[i, j],
                        "subset_target_mean": target_mean_mat[i, j],
                        "subset_n_pairs": int(count_mat[i, j]),
                    }
                )

        if not args.skip_leave_one_family_out:
            rows = compute_leave_one_family_out(
                metric_mat=metric_mat,
                target_mat=target_mat,
                models=models,
                family_of=family_of,
                protocol=args.protocol,
                corr_type=args.corr_type,
            )
            leave_one_family_out_by_metric[metric_name] = rows

            for row in rows:
                loo_rows_all.append(
                    {
                        "metric_name": metric_name,
                        "metric_source": "pairwise",
                        "protocol": args.protocol,
                        "corr_type": args.corr_type,
                        **row,
                    }
                )

    if not family_corr_mats:
        raise ValueError(
            "Не удалось построить family-wise heatmaps: нет совместимых метрик"
        )

    assert families_ref is not None
    assert target_mean_ref is not None
    assert count_ref is not None

    # --------------------------------------------------------
    # Save target / count heatmaps
    # --------------------------------------------------------
    target_plot = args.out_dir / (
        "family_target_mean_signed.png"
        if args.protocol == "signed"
        else "family_target_mean_abs.png"
    )
    save_family_target_heatmap(
        families=families_ref,
        target_mean_mat=target_mean_ref,
        count_mat=count_ref,
        out_path=target_plot,
        title_prefix=title_prefix,
        protocol=args.protocol,
        annotate=args.annotate,
        plots_exts=args.plots_ext,
    )

    count_plot = args.out_dir / (
        "family_pair_counts_signed.png"
        if args.protocol == "signed"
        else "family_pair_counts_abs.png"
    )
    save_family_count_heatmap(
        families=families_ref,
        count_mat=count_ref,
        out_path=count_plot,
        title_prefix=title_prefix,
        annotate=args.annotate,
        plots_exts=args.plots_ext,
    )

    # --------------------------------------------------------
    # Save main correlation grid
    # --------------------------------------------------------
    corr_plot = (
        args.out_dir
        / f"family_subset_correlations_{args.corr_type}_{args.protocol}.png"
    )
    save_family_correlation_grid(
        family_corr_mats=family_corr_mats,
        family_count_mats=family_count_mats,
        global_corrs=global_corrs,
        global_counts=global_counts,
        families=families_ref,
        out_path=corr_plot,
        title_prefix=title_prefix,
        corr_type=args.corr_type,
        annotate=args.annotate,
        metric_sources=metric_sources,
        protocol=args.protocol,
        plots_exts=args.plots_ext,
    )

    # --------------------------------------------------------
    # Save CSVs
    # --------------------------------------------------------
    global_csv = (
        args.out_dir
        / f"family_subset_global_summary_{args.corr_type}_{args.protocol}.csv"
    )
    save_global_summary_csv(global_csv, global_rows)

    family_csv = (
        args.out_dir / f"family_subset_details_{args.corr_type}_{args.protocol}.csv"
    )
    save_family_summary_csv(family_csv, family_rows)

    # --------------------------------------------------------
    # Leave-one-family-out
    # --------------------------------------------------------
    loo_csv = None
    loo_plot = None

    if not args.skip_leave_one_family_out and leave_one_family_out_by_metric:
        loo_csv = (
            args.out_dir / f"leave_one_family_out_{args.corr_type}_{args.protocol}.csv"
        )
        save_leave_one_family_out_csv(loo_csv, loo_rows_all)

        loo_plot = (
            args.out_dir / f"leave_one_family_out_{args.corr_type}_{args.protocol}.png"
        )
        save_leave_one_family_out_barplot(
            metric_rows=leave_one_family_out_by_metric,
            out_path=loo_plot,
            title_prefix=title_prefix,
            corr_type=args.corr_type,
            metric_sources=metric_sources,
            plots_exts=args.plots_ext,
        )

    print("Saved:")
    for save_path in iter_plot_paths(target_plot, args.plots_ext):
        print(f"  - {save_path}")
    for save_path in iter_plot_paths(count_plot, args.plots_ext):
        print(f"  - {save_path}")
    for save_path in iter_plot_paths(corr_plot, args.plots_ext):
        print(f"  - {save_path}")
    print(f"  - {global_csv}")
    print(f"  - {family_csv}")
    if loo_csv is not None:
        print(f"  - {loo_csv}")
    if loo_plot is not None:
        for save_path in iter_plot_paths(loo_plot, args.plots_ext):
            print(f"  - {save_path}")


if __name__ == "__main__":
    main()
