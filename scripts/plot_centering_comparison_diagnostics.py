from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def _parse_label_path(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw)
        return path.name, path
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Пустая подпись в элементе --experiment: {raw!r}")
    return label, Path(path.strip())


def _resolve_artifact_path(path: Path, dataset_key: str, artifact_name: str) -> Path:
    if path.is_file():
        return path
    candidate = path / "metric_matrices" / dataset_key / "artifacts" / artifact_name
    if candidate.exists():
        return candidate
    direct = path / artifact_name
    if direct.exists():
        return direct
    raise FileNotFoundError(
        "Артефакт не найден. Проверены пути:\n" f"  {candidate}\n" f"  {direct}"
    )


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _load_geometry_from_meta(artifacts: Dict[str, np.ndarray]) -> str:
    raw = artifacts.get("diagnostics_meta_json")
    if raw is None:
        return ""
    try:
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        meta = json.loads(str(raw))
        return str(meta.get("local_geometry_mode", ""))
    except Exception:
        return ""


def _direction_prefixes(artifacts: Dict[str, np.ndarray]) -> List[str]:
    prefixes = set()
    for key in artifacts:
        if "/" not in key:
            continue
        prefix, field = key.split("/", 1)
        if "_to_" not in prefix:
            continue
        if field in {"metric_ranks", "ranks", "relative_residuals", "residuals"}:
            prefixes.add(prefix)
    return sorted(prefixes)


def _values_for_field(
    artifacts: Dict[str, np.ndarray],
    prefixes: Sequence[str],
    preferred: str,
    fallback: str,
) -> np.ndarray:
    values: List[np.ndarray] = []
    for prefix in prefixes:
        key = f"{prefix}/{preferred}"
        if key not in artifacts:
            key = f"{prefix}/{fallback}"
        if key not in artifacts:
            continue
        arr = np.asarray(artifacts[key], dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr)
    if not values:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(values)


def _summary(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
        }
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
    }


def _display_label(label: str) -> str:
    mapping = {
        "abs": "без центрирования",
        "absolute": "без центрирования",
        "absolute_coords_v0": "без центрирования",
        "centered_v1": "центрирование v1",
        "centered_offsets_v1": "центрирование v1",
        "centered_v2": "центрирование v2",
        "centered_offsets_v2": "центрирование v2",
    }
    return mapping.get(label, label)


def _metric_display_name(artifact_name: str) -> str:
    stem = artifact_name.replace("_artifacts.npz", "")
    mapping = {
        "lin_k5_antisym": "линейная kNN-метрика, k=5",
        "lin_k10_antisym": "линейная kNN-метрика, k=10",
        "lin_k20_antisym": "линейная kNN-метрика, k=20",
        "lin_k40_antisym": "линейная kNN-метрика, k=40",
        "lin_k80_antisym": "линейная kNN-метрика, k=80",
        "rff_k10_antisym": "RFF kNN-метрика, k=10",
        "directed_k10": "направленная kNN-метрика, k=10",
        "multiscale_mean_antisym": "мульти-масштабная kNN-метрика",
        "lin_eps_5_antisym": "epsilon-метрика, p=5",
        "lin_eps_10_antisym": "epsilon-метрика, p=10",
        "lin_eps_20_antisym": "epsilon-метрика, p=20",
    }
    return mapping.get(stem, stem)


def _write_summary_csv(
    path: Path,
    rows: Iterable[Dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = [
        "artifact",
        "label",
        "geometry_from_meta",
        "quantity",
        "n",
        "mean",
        "std",
        "median",
        "q25",
        "q75",
        "artifact_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_boxplots(
    *,
    artifact_name: str,
    labels: Sequence[str],
    rank_values: Sequence[np.ndarray],
    residual_values: Sequence[np.ndarray],
    out_base: Path,
    plot_exts: Sequence[str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    display_labels = [_display_label(label) for label in labels]
    fig.suptitle(
        f"Сравнение режимов центрирования: {_metric_display_name(artifact_name)}"
    )

    panels = [
        (axes[0], rank_values, "Распределение RankMe по центрам", "RankMe"),
        (
            axes[1],
            residual_values,
            "Распределение относительной ошибки по центрам",
            "Относительная ошибка",
        ),
    ]
    for ax, values, title, ylabel in panels:
        nonempty = [v if v.size else np.asarray([np.nan]) for v in values]
        try:
            ax.boxplot(
                nonempty, tick_labels=display_labels, showmeans=True, meanline=True
            )
        except TypeError:
            ax.boxplot(nonempty, labels=display_labels, showmeans=True, meanline=True)
        ax.set_title(title)
        ax.set_xlabel("Режим центрирования")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=0)
        ax.legend(
            handles=[
                Line2D([0], [0], color="tab:orange", linewidth=1.5, label="медиана"),
                Line2D(
                    [0],
                    [0],
                    color="green",
                    linestyle="--",
                    linewidth=1.5,
                    label="среднее",
                ),
            ],
            fontsize=8,
            loc="best",
        )

    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in plot_exts:
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Построить компактные boxplot-диагностики для сравнения режимов центрирования."
    )
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        help=(
            "Запись эксперимента/артефакта вида label=PATH. PATH может быть "
            "папкой эксперимента, папкой artifacts или прямым путём к "
            "*_artifacts.npz. Повторить для каждого режима центрирования."
        ),
    )
    parser.add_argument("--dataset_key", default="cifar10_test")
    parser.add_argument(
        "--artifact_name",
        required=True,
        help="Имя файла артефактов, например lin_k10_antisym_artifacts.npz.",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--plots_ext", default="png,svg")
    args = parser.parse_args()

    entries = [_parse_label_path(raw) for raw in args.experiment]
    plot_exts = [
        part.strip().lstrip(".") for part in args.plots_ext.split(",") if part.strip()
    ]
    out_dir = Path(args.out_dir)

    labels: List[str] = []
    rank_values: List[np.ndarray] = []
    residual_values: List[np.ndarray] = []
    summary_rows: List[Dict[str, object]] = []

    for label, raw_path in entries:
        artifact_path = _resolve_artifact_path(
            raw_path, args.dataset_key, args.artifact_name
        )
        artifacts = _load_npz(artifact_path)
        prefixes = _direction_prefixes(artifacts)
        ranks = _values_for_field(artifacts, prefixes, "metric_ranks", "ranks")
        residuals = _values_for_field(
            artifacts, prefixes, "relative_residuals", "residuals"
        )
        geometry = _load_geometry_from_meta(artifacts)

        labels.append(label)
        rank_values.append(ranks)
        residual_values.append(residuals)

        for quantity, values in [
            ("metric_ranks", ranks),
            ("relative_residuals", residuals),
        ]:
            stats = _summary(values)
            summary_rows.append(
                {
                    "artifact": args.artifact_name,
                    "label": label,
                    "geometry_from_meta": geometry,
                    "quantity": quantity,
                    "artifact_path": str(artifact_path),
                    **stats,
                }
            )
        print(
            f"{label}: {artifact_path} | directions={len(prefixes)} | "
            f"rank_values={ranks.size} | residual_values={residuals.size} | geometry={geometry or '?'}"
        )

    stem = args.artifact_name.replace("_artifacts.npz", "")
    out_base = out_dir / f"centering_boxplot_{stem}"
    _plot_boxplots(
        artifact_name=args.artifact_name,
        labels=labels,
        rank_values=rank_values,
        residual_values=residual_values,
        out_base=out_base,
        plot_exts=plot_exts,
    )
    csv_path = out_dir / f"centering_boxplot_{stem}_summary.csv"
    _write_summary_csv(csv_path, summary_rows)
    print(f"Сохранены графики: {out_base}.[{','.join(plot_exts)}]")
    print(f"Сохранена сводка: {csv_path}")


if __name__ == "__main__":
    main()
