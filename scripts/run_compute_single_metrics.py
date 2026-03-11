from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


# ============================================================
# Вспомогательные функции
# ============================================================

SUPPORTED_EXTENSIONS = {".npy", ".npz", ".pt", ".pth"}


def _maybe_import_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        return None


def ensure_2d_float_array(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if arr.ndim != 2:
        raise ValueError(f"Ожидался 2D массив, получена форма {arr.shape}")

    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"Ожидался числовой массив, dtype={arr.dtype}")

    arr = np.asarray(arr, dtype=np.float64)

    if not np.isfinite(arr).all():
        raise ValueError("Эмбеддинги содержат NaN или Inf")

    return arr


def center_embeddings(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def covariance_eigenvalues(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.shape[0] == 0 or x.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)

    xc = center_embeddings(x)
    denom = max(int(xc.shape[0]) - 1, 1)
    cov = (xc.T @ xc) / float(denom)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvals[eigvals < 0.0] = 0.0
    eigvals = eigvals[::-1]
    eigvals = eigvals[eigvals > eps]
    return eigvals


def select_array_from_npz(npz_obj: np.lib.npyio.NpzFile) -> np.ndarray:
    preferred_keys = [
        "embeddings",
        "features",
        "reps",
        "representations",
        "X",
        "arr_0",
    ]

    for key in preferred_keys:
        if key in npz_obj.files:
            return npz_obj[key]

    arrays_2d: list[np.ndarray] = []
    for key in npz_obj.files:
        value = npz_obj[key]
        if isinstance(value, np.ndarray) and value.ndim == 2:
            arrays_2d.append(value)

    if len(arrays_2d) == 1:
        return arrays_2d[0]

    raise ValueError(
        f"Не удалось однозначно определить массив эмбеддингов в npz. Ключи: {npz_obj.files}"
    )


def load_embeddings(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        return ensure_2d_float_array(arr)

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            arr = select_array_from_npz(data)
        return ensure_2d_float_array(arr)

    if suffix in {".pt", ".pth"}:
        torch = _maybe_import_torch()
        if torch is None:
            raise RuntimeError(
                f"Файл {path} требует torch для загрузки, но torch не установлен"
            )

        obj = torch.load(path, map_location="cpu")

        if isinstance(obj, dict):
            preferred_keys = [
                "embeddings",
                "features",
                "reps",
                "representations",
                "X",
            ]
            for key in preferred_keys:
                if key in obj:
                    value = obj[key]
                    if hasattr(value, "detach"):
                        value = value.detach().cpu().numpy()
                    return ensure_2d_float_array(np.asarray(value))

            arrays_2d = []
            for _, value in obj.items():
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                value = np.asarray(value)
                if value.ndim == 2:
                    arrays_2d.append(value)

            if len(arrays_2d) == 1:
                return ensure_2d_float_array(arrays_2d[0])

            raise ValueError(
                f"Не удалось однозначно определить тензор эмбеддингов в {path}"
            )

        if hasattr(obj, "detach"):
            obj = obj.detach().cpu().numpy()

        return ensure_2d_float_array(np.asarray(obj))

    raise ValueError(f"Неподдерживаемое расширение файла: {path.suffix}")


def discover_embedding_files(embeddings_dir: Path) -> list[Path]:
    files = [
        Path(os.path.join(embeddings_dir, fn))
        for fn in sorted(os.listdir(embeddings_dir))
        if Path(fn).suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if files:
        return files

    subdirs = [p.name for p in sorted(embeddings_dir.iterdir()) if p.is_dir()]
    if subdirs:
        raise FileNotFoundError(
            f"В {embeddings_dir} не найдено файлов эмбеддингов с расширениями "
            f"{sorted(SUPPORTED_EXTENSIONS)}. "
            f"Похоже, это корневая папка embeddings с подпапками: {subdirs}. "
            f"Нужно передать конкретную папку с файлами моделей."
        )

    raise FileNotFoundError(
        f"В {embeddings_dir} не найдено файлов эмбеддингов "
        f"с расширениями {sorted(SUPPORTED_EXTENSIONS)}"
    )


def infer_model_name(path: Path) -> str:
    return path.stem


# ============================================================
# Метрики
# ============================================================

def stable_rank(x: np.ndarray, epsilon: float = 1e-12) -> float:
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    if s.size == 0:
        return 0.0

    fro_sq = float(np.sum(s ** 2))
    max_sq = float(np.max(s ** 2)) + epsilon
    return fro_sq / max_sq


def pseudo_condition_number(x: np.ndarray, epsilon: float = 1e-12) -> float:
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    if s.size == 0:
        return float("inf")

    s_pos = s[s > epsilon]
    if s_pos.size == 0:
        return float("inf")

    s_max = float(np.max(s))
    s_min = float(np.min(s_pos)) + epsilon
    return s_max / s_min


def coherence(x: np.ndarray) -> float:
    u, _, _ = np.linalg.svd(x, full_matrices=False)
    if u.size == 0:
        return 0.0

    row_sq = np.sum(u ** 2, axis=1)
    return float(np.max(row_sq))


def rankme(x: np.ndarray, epsilon: float = 1e-12, normalize: bool = False) -> float:
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    if s.size == 0:
        return 0.0

    p = s / (float(np.sum(s)) + epsilon)
    p = p + epsilon

    entropy = -float(np.sum(p * np.log(p)))
    value = float(np.exp(entropy))

    if normalize:
        value /= float(min(x.shape[0], x.shape[1]))

    return value


def nesum(x: np.ndarray, epsilon: float = 1e-12) -> float:
    eigvals = covariance_eigenvalues(x, eps=epsilon)
    if eigvals.size == 0:
        return 0.0

    lambda_1 = float(eigvals[0])
    if lambda_1 <= epsilon:
        return 0.0

    return float(np.sum(eigvals) / (lambda_1 + epsilon))


def self_cluster(x: np.ndarray, epsilon: float = 1e-12) -> float:
    n, d = x.shape
    if n <= 1 or d <= 1:
        return 0.0

    w = l2_normalize_rows(x, eps=epsilon)
    gram_feature = w.T @ w
    gram_fro = float(np.linalg.norm(gram_feature, ord="fro"))

    baseline_random = float(n + (n * (n - 1)) / d)
    max_value = float(n * n)
    denom = max_value - baseline_random
    if abs(denom) <= epsilon:
        return 0.0

    value = (gram_fro - baseline_random) / denom
    return float(value)


def alpha_req(x: np.ndarray, epsilon: float = 1e-12) -> float:
    eigvals = covariance_eigenvalues(x, eps=epsilon)
    if eigvals.size < 2:
        return 0.0

    ranks = np.arange(1, eigvals.size + 1, dtype=np.float64)
    log_ranks = np.log(ranks)
    log_eigvals = np.log(np.maximum(eigvals, epsilon))

    slope, _ = np.polyfit(log_ranks, log_eigvals, deg=1)
    alpha = -float(slope)
    return alpha


@dataclass(frozen=True)
class MetricSpec:
    name: str
    fn: Callable[[np.ndarray], float]
    higher_is_better: bool


METRICS: dict[str, MetricSpec] = {
    "stable_rank": MetricSpec(
        name="stable_rank",
        fn=stable_rank,
        higher_is_better=True,
    ),
    "pseudo_condition_number": MetricSpec(
        name="pseudo_condition_number",
        fn=pseudo_condition_number,
        higher_is_better=False,
    ),
    "coherence": MetricSpec(
        name="coherence",
        fn=coherence,
        higher_is_better=False,
    ),
    "rankme": MetricSpec(
        name="rankme",
        fn=rankme,
        higher_is_better=True,
    ),
    "nesum": MetricSpec(
        name="nesum",
        fn=nesum,
        higher_is_better=True,
    ),
    "self_cluster": MetricSpec(
        name="self_cluster",
        fn=self_cluster,
        higher_is_better=False,
    ),
    "alpha_req": MetricSpec(
        name="alpha_req",
        fn=alpha_req,
        higher_is_better=True,
    ),
}


# ============================================================
# IO
# ============================================================

def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_output_path(out_dir: Path, metric_name: str, model_name: str) -> Path:
    return out_dir / metric_name / f"{model_name}.json"


# ============================================================
# Основная программа
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Вычисление одиночных метрик качества эмбеддингов для одной папки embeddings."
    )
    parser.add_argument(
        "--embeddings_dir",
        type=Path,
        required=True,
        help="Папка с файлами эмбеддингов моделей (.npy/.npz/.pt/.pth).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Папка для сохранения результатов single-метрик.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=[
            "stable_rank",
            "pseudo_condition_number",
            "coherence",
            "rankme",
            "nesum",
            "self_cluster",
        ],
        choices=sorted(METRICS.keys()),
        help="Список метрик для вычисления.",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Центрировать эмбеддинги перед вычислением метрик.",
    )
    parser.add_argument(
        "--row_l2_normalize",
        action="store_true",
        help="L2-нормализовать строки перед вычислением метрик.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Пересчитывать метрики даже если результат уже существует.",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Остановиться при первой ошибке.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    embeddings_dir: Path = args.embeddings_dir
    out_dir: Path = args.out_dir
    metric_names: list[str] = args.metrics

    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Директория с эмбеддингами не найдена: {embeddings_dir}")

    files = discover_embedding_files(embeddings_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "embeddings_dir": str(embeddings_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "metrics": metric_names,
        "center": bool(args.center),
        "row_l2_normalize": bool(args.row_l2_normalize),
        "n_embedding_files": len(files),
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
    }
    save_json(out_dir / "manifest.json", manifest)

    errors: list[dict[str, str]] = []
    total_jobs = len(files) * len(metric_names)
    done_jobs = 0
    skipped_jobs = 0

    print("============================================================")
    print("ВЫЧИСЛЕНИЕ ОДИНОЧНЫХ МЕТРИК")
    print("============================================================")
    print(f"Папка эмбеддингов      : {embeddings_dir}")
    print(f"Директория результатов : {out_dir}")
    print(f"Файлов эмбеддингов     : {len(files)}")
    print(f"Метрики                : {metric_names}")
    print("============================================================")

    for emb_path in files:
        model_name = infer_model_name(emb_path)

        try:
            x = load_embeddings(emb_path)

            if args.center:
                x = center_embeddings(x)

            if args.row_l2_normalize:
                x = l2_normalize_rows(x)

        except Exception as e:
            error = {
                "file": str(emb_path),
                "model_name": model_name,
                "error": repr(e),
            }
            errors.append(error)
            print(f"[ОШИБКА] загрузка не удалась для {model_name}: {e}")
            if args.fail_fast:
                raise
            continue

        for metric_name in metric_names:
            spec = METRICS[metric_name]
            out_path = build_output_path(out_dir, metric_name, model_name)

            if out_path.exists() and not args.overwrite:
                skipped_jobs += 1
                print(f"[ПРОПУСК] {metric_name} | {model_name} -> уже существует")
                continue

            try:
                value = float(spec.fn(x))

                payload = {
                    "metric_name": spec.name,
                    "model_name": model_name,
                    "source_file": str(emb_path.resolve()),
                    "n_samples": int(x.shape[0]),
                    "dim": int(x.shape[1]),
                    "center": bool(args.center),
                    "row_l2_normalize": bool(args.row_l2_normalize),
                    "higher_is_better": bool(spec.higher_is_better),
                    "value": value,
                }
                save_json(out_path, payload)

                done_jobs += 1
                print(f"[OK]   {metric_name} | {model_name} -> {value:.6f}")

            except Exception as e:
                error = {
                    "file": str(emb_path),
                    "model_name": model_name,
                    "metric_name": metric_name,
                    "error": repr(e),
                }
                errors.append(error)
                print(f"[ОШИБКА] {metric_name} не вычислена для {model_name}: {e}")
                if args.fail_fast:
                    raise

    summary = {
        "total_jobs": total_jobs,
        "computed_jobs": done_jobs,
        "skipped_jobs": skipped_jobs,
        "n_errors": len(errors),
        "errors": errors,
    }
    save_json(out_dir / "summary.json", summary)

    print("============================================================")
    print("ВЫЧИСЛЕНИЕ ОДИНОЧНЫХ МЕТРИК ЗАВЕРШЕНО")
    print("============================================================")
    print(f"Вычислено              : {done_jobs}")
    print(f"Пропущено              : {skipped_jobs}")
    print(f"Ошибок                 : {len(errors)}")
    print("============================================================")


if __name__ == "__main__":
    main()
