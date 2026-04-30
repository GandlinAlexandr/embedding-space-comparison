from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


DEFAULT_SINGLE_METRICS_ROOT = Path("data") / "single_metrics"


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


def slugify_dataset_key(raw: str) -> str:
    cleaned = []
    for ch in raw.strip():
        if ch.isalnum() or ch in {"-", "_", "."}:
            cleaned.append(ch)
        elif ch in {" ", "/", "\\"}:
            cleaned.append("_")
    key = "".join(cleaned).strip("._-")
    if not key:
        raise ValueError(f"Не удалось получить dataset_key из {raw!r}")
    return key


def infer_dataset_key(embeddings_dir: Path) -> str:
    return slugify_dataset_key(embeddings_dir.name)


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

    fro_sq = float(np.sum(s**2))
    max_sq = float(np.max(s**2))
    if max_sq <= epsilon:
        return 0.0
    return fro_sq / max_sq


def pseudo_condition_number(x: np.ndarray, epsilon: float = 1e-12) -> float:
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    if s.size == 0:
        return float("inf")

    s_pos = s[s > epsilon]
    if s_pos.size == 0:
        return float("inf")

    s_max = float(np.max(s))
    s_min = float(np.min(s_pos))
    return s_max / s_min


def coherence(x: np.ndarray, epsilon: float = 1e-12) -> float:
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    if u.size == 0 or s.size == 0:
        return 0.0

    rank = int(np.sum(s > epsilon))
    if rank == 0:
        return 0.0

    u_rank = u[:, :rank]
    vh_rank = vh[:rank, :]

    left_mu = (float(x.shape[0]) / float(rank)) * float(
        np.max(np.sum(u_rank**2, axis=1))
    )
    right_mu = (float(x.shape[1]) / float(rank)) * float(
        np.max(np.sum(vh_rank.T**2, axis=1))
    )
    return float(max(left_mu, right_mu))


def rankme(x: np.ndarray, epsilon: float = 1e-12, normalize: bool = False) -> float:
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    if s.size == 0:
        return 0.0

    total = float(np.sum(s))
    if total <= epsilon:
        return 0.0

    p = s / total
    p = p[p > 0.0]

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

    return float(np.sum(eigvals) / lambda_1)


def self_cluster(x: np.ndarray, epsilon: float = 1e-12) -> float:
    n, d = x.shape
    if n <= 1 or d <= 1:
        return 0.0

    w = l2_normalize_rows(x, eps=epsilon)
    gram_feature = w.T @ w
    gram_fro_sq = float(np.linalg.norm(gram_feature, ord="fro") ** 2)

    baseline_random = float(n + (n * (n - 1)) / d)
    max_value = float(n * n)
    denom = max_value - baseline_random
    if abs(denom) <= epsilon:
        return 0.0

    value = (gram_fro_sq - baseline_random) / denom
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


@dataclass
class SingleMetricResult:
    values: dict[str, float]
    artifacts: dict[str, np.ndarray]
    backend: str
    device: str


def _torch_available_device(requested: str):
    torch = _maybe_import_torch()
    if torch is None:
        if requested.lower().startswith("cuda"):
            raise RuntimeError("Запрошен CUDA-расчёт, но torch не установлен")
        return None, "cpu", "numpy"

    requested_norm = requested.lower()
    if requested_norm == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_norm == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Запрошен --device cuda, но torch.cuda.is_available() == False")
        device_name = "cuda"
    elif requested_norm == "cpu":
        device_name = "cpu"
    else:
        device_name = requested

    if device_name == "cpu":
        return torch, device_name, "numpy"
    return torch, device_name, "torch"


def compute_single_metrics_numpy(
    x: np.ndarray,
    metric_names: list[str],
    epsilon: float = 1e-12,
) -> SingleMetricResult:
    values: dict[str, float] = {}
    artifacts: dict[str, np.ndarray] = {}

    need_svd = any(
        name in {"stable_rank", "pseudo_condition_number", "coherence", "rankme"}
        for name in metric_names
    )
    if need_svd:
        u, s, vh = np.linalg.svd(x, full_matrices=False)
        artifacts["singular_values"] = np.asarray(s, dtype=np.float64)
    else:
        u = np.empty((x.shape[0], 0), dtype=np.float64)
        s = np.empty(0, dtype=np.float64)
        vh = np.empty((0, x.shape[1]), dtype=np.float64)

    need_cov = any(name in {"nesum", "alpha_req"} for name in metric_names)
    if need_cov:
        eigvals = covariance_eigenvalues(x, eps=epsilon)
        artifacts["covariance_eigenvalues"] = np.asarray(eigvals, dtype=np.float64)
    else:
        eigvals = np.empty(0, dtype=np.float64)

    for metric_name in metric_names:
        if metric_name == "stable_rank":
            if s.size == 0:
                values[metric_name] = 0.0
            else:
                max_sq = float(np.max(s**2))
                values[metric_name] = (
                    0.0 if max_sq <= epsilon else float(np.sum(s**2) / max_sq)
                )

        elif metric_name == "pseudo_condition_number":
            s_pos = s[s > epsilon]
            if s.size == 0 or s_pos.size == 0:
                values[metric_name] = float("inf")
            else:
                values[metric_name] = float(np.max(s) / np.min(s_pos))

        elif metric_name == "coherence":
            rank = int(np.sum(s > epsilon))
            if rank == 0:
                values[metric_name] = 0.0
            else:
                u_rank = u[:, :rank]
                vh_rank = vh[:rank, :]
                left_mu = (float(x.shape[0]) / float(rank)) * float(
                    np.max(np.sum(u_rank**2, axis=1))
                )
                right_mu = (float(x.shape[1]) / float(rank)) * float(
                    np.max(np.sum(vh_rank.T**2, axis=1))
                )
                values[metric_name] = float(max(left_mu, right_mu))

        elif metric_name == "rankme":
            if s.size == 0:
                values[metric_name] = 0.0
            else:
                total = float(np.sum(s))
                if total <= epsilon:
                    values[metric_name] = 0.0
                else:
                    p = s / total
                    p = p[p > 0.0]
                    values[metric_name] = float(
                        np.exp(-float(np.sum(p * np.log(p))))
                    )

        elif metric_name == "nesum":
            if eigvals.size == 0 or float(eigvals[0]) <= epsilon:
                values[metric_name] = 0.0
            else:
                values[metric_name] = float(np.sum(eigvals) / float(eigvals[0]))

        elif metric_name == "self_cluster":
            values[metric_name] = self_cluster(x, epsilon=epsilon)

        elif metric_name == "alpha_req":
            if eigvals.size < 2:
                values[metric_name] = 0.0
            else:
                ranks = np.arange(1, eigvals.size + 1, dtype=np.float64)
                slope, _ = np.polyfit(
                    np.log(ranks),
                    np.log(np.maximum(eigvals, epsilon)),
                    deg=1,
                )
                values[metric_name] = -float(slope)

        else:
            values[metric_name] = float(METRICS[metric_name].fn(x))

    return SingleMetricResult(
        values=values,
        artifacts=artifacts,
        backend="numpy",
        device="cpu",
    )


def compute_single_metrics_torch(
    x: np.ndarray,
    metric_names: list[str],
    torch_module: Any,
    device_name: str,
    epsilon: float = 1e-12,
) -> SingleMetricResult:
    torch = torch_module
    xt = torch.as_tensor(x, dtype=torch.float64, device=device_name)
    values: dict[str, float] = {}
    artifacts: dict[str, np.ndarray] = {}

    need_svd = any(
        name in {"stable_rank", "pseudo_condition_number", "coherence", "rankme"}
        for name in metric_names
    )
    if need_svd:
        u, s, vh = torch.linalg.svd(xt, full_matrices=False)
        artifacts["singular_values"] = s.detach().cpu().numpy().astype(np.float64)
    else:
        u = torch.empty((xt.shape[0], 0), dtype=xt.dtype, device=xt.device)
        s = torch.empty((0,), dtype=xt.dtype, device=xt.device)
        vh = torch.empty((0, xt.shape[1]), dtype=xt.dtype, device=xt.device)

    need_cov = any(name in {"nesum", "alpha_req"} for name in metric_names)
    if need_cov:
        xc = xt - xt.mean(dim=0, keepdim=True)
        denom = max(int(xt.shape[0]) - 1, 1)
        cov = (xc.T @ xc) / float(denom)
        eigvals = torch.linalg.eigvalsh(cov).flip(0)
        eigvals = torch.clamp(eigvals, min=0.0)
        eigvals = eigvals[eigvals > epsilon]
        artifacts["covariance_eigenvalues"] = eigvals.detach().cpu().numpy().astype(np.float64)
    else:
        eigvals = torch.empty((0,), dtype=xt.dtype, device=xt.device)

    for metric_name in metric_names:
        if metric_name == "stable_rank":
            if s.numel() == 0:
                values[metric_name] = 0.0
            else:
                max_sq = torch.max(s**2)
                values[metric_name] = (
                    0.0
                    if float(max_sq.detach().cpu()) <= epsilon
                    else float((torch.sum(s**2) / max_sq).detach().cpu())
                )

        elif metric_name == "pseudo_condition_number":
            s_pos = s[s > epsilon]
            if s.numel() == 0 or s_pos.numel() == 0:
                values[metric_name] = float("inf")
            else:
                values[metric_name] = float((torch.max(s) / torch.min(s_pos)).detach().cpu())

        elif metric_name == "coherence":
            rank = int(torch.sum(s > epsilon).detach().cpu())
            if rank == 0:
                values[metric_name] = 0.0
            else:
                u_rank = u[:, :rank]
                vh_rank = vh[:rank, :]
                left_mu = (float(xt.shape[0]) / float(rank)) * float(
                    torch.max(torch.sum(u_rank**2, dim=1)).detach().cpu()
                )
                right_mu = (float(xt.shape[1]) / float(rank)) * float(
                    torch.max(torch.sum(vh_rank.T**2, dim=1)).detach().cpu()
                )
                values[metric_name] = float(max(left_mu, right_mu))

        elif metric_name == "rankme":
            if s.numel() == 0:
                values[metric_name] = 0.0
            else:
                total = torch.sum(s)
                if float(total.detach().cpu()) <= epsilon:
                    values[metric_name] = 0.0
                else:
                    p = s / total
                    p = p[p > 0.0]
                    values[metric_name] = float(torch.exp(-torch.sum(p * torch.log(p))).detach().cpu())

        elif metric_name == "nesum":
            if eigvals.numel() == 0 or float(eigvals[0].detach().cpu()) <= epsilon:
                values[metric_name] = 0.0
            else:
                values[metric_name] = float((torch.sum(eigvals) / eigvals[0]).detach().cpu())

        elif metric_name == "self_cluster":
            n, d = int(xt.shape[0]), int(xt.shape[1])
            if n <= 1 or d <= 1:
                values[metric_name] = 0.0
            else:
                norms = torch.linalg.norm(xt, dim=1, keepdim=True)
                w = xt / torch.clamp(norms, min=epsilon)
                gram_feature = w.T @ w
                gram_fro_sq = torch.linalg.norm(gram_feature, ord="fro") ** 2
                artifacts["self_cluster_feature_gram_fro_sq"] = np.asarray(
                    [float(gram_fro_sq.detach().cpu())],
                    dtype=np.float64,
                )
                baseline_random = float(n + (n * (n - 1)) / d)
                max_value = float(n * n)
                denom = max_value - baseline_random
                values[metric_name] = (
                    0.0
                    if abs(denom) <= epsilon
                    else float(((gram_fro_sq - baseline_random) / denom).detach().cpu())
                )

        elif metric_name == "alpha_req":
            if eigvals.numel() < 2:
                values[metric_name] = 0.0
            else:
                ranks = torch.arange(
                    1,
                    eigvals.numel() + 1,
                    dtype=xt.dtype,
                    device=xt.device,
                )
                log_ranks = torch.log(ranks)
                log_eigvals = torch.log(torch.clamp(eigvals, min=epsilon))
                x_centered = log_ranks - torch.mean(log_ranks)
                y_centered = log_eigvals - torch.mean(log_eigvals)
                slope = torch.sum(x_centered * y_centered) / torch.sum(x_centered**2)
                values[metric_name] = -float(slope.detach().cpu())

        else:
            values[metric_name] = float(METRICS[metric_name].fn(x))

    return SingleMetricResult(
        values=values,
        artifacts=artifacts,
        backend="torch",
        device=device_name,
    )


def compute_single_metrics(
    x: np.ndarray,
    metric_names: list[str],
    requested_device: str,
) -> SingleMetricResult:
    torch, device_name, backend = _torch_available_device(requested_device)
    if backend == "torch":
        try:
            return compute_single_metrics_torch(
                x=x,
                metric_names=metric_names,
                torch_module=torch,
                device_name=device_name,
            )
        except Exception as exc:
            if requested_device.lower() == "cuda":
                raise
            print(f"[warn] GPU/torch расчёт не удался ({exc!r}); fallback на numpy/CPU")

    return compute_single_metrics_numpy(x=x, metric_names=metric_names)


# ============================================================
# IO
# ============================================================


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_values_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_key",
        "metric_name",
        "model_name",
        "value",
        "higher_is_better",
        "n_samples",
        "dim",
        "backend",
        "device",
        "source_file",
        "artifact_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_saved_value_rows(metrics_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("*/*.json")):
        try:
            with path.open("r", encoding="utf-8-sig") as f:
                payload = json.load(f)
        except Exception:
            continue
        if isinstance(payload, dict) and "value" in payload:
            rows.append(payload)
    return rows


def build_metric_output_path(metrics_dir: Path, metric_name: str, model_name: str) -> Path:
    return metrics_dir / metric_name / f"{model_name}.json"


def build_artifact_output_path(artifacts_dir: Path, model_name: str) -> Path:
    return artifacts_dir / f"{model_name}.npz"


def resolve_output_layout(args: argparse.Namespace) -> tuple[str, Path, Path, Path]:
    dataset_key = (
        slugify_dataset_key(args.dataset_key)
        if args.dataset_key
        else infer_dataset_key(args.embeddings_dir)
    )

    if args.out_dir is not None:
        dataset_dir = args.out_dir
    else:
        dataset_dir = args.out_root / dataset_key

    return dataset_key, dataset_dir, dataset_dir / "metrics", dataset_dir / "artifacts"


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
        "--out_root",
        type=Path,
        default=DEFAULT_SINGLE_METRICS_ROOT,
        help=(
            "Корневая папка canonical single-metrics store. "
            "Результаты пишутся в <out_root>/<dataset_key>/."
        ),
    )
    parser.add_argument(
        "--dataset_key",
        type=str,
        default=None,
        help=(
            "Имя датасета/сплита внутри data/single_metrics. "
            "Если не задано, берётся имя папки --embeddings_dir."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help=(
            "Legacy/override: конкретная папка результата. "
            "Если задано, используется вместо <out_root>/<dataset_key>."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Устройство для расчёта: auto, cuda, cpu или torch device string.",
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
            "alpha_req",
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
    metric_names: list[str] = args.metrics
    dataset_key, dataset_dir, metrics_dir, artifacts_dir = resolve_output_layout(args)

    if not embeddings_dir.exists():
        raise FileNotFoundError(
            f"Директория с эмбеддингами не найдена: {embeddings_dir}"
        )

    files = discover_embedding_files(embeddings_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 2,
        "dataset_key": dataset_key,
        "embeddings_dir": str(embeddings_dir.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "metrics_dir": str(metrics_dir.resolve()),
        "artifacts_dir": str(artifacts_dir.resolve()),
        "metrics": metric_names,
        "center": bool(args.center),
        "row_l2_normalize": bool(args.row_l2_normalize),
        "device": str(args.device),
        "n_embedding_files": len(files),
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
    }
    save_json(dataset_dir / "manifest.json", manifest)

    errors: list[dict[str, str]] = []
    total_jobs = len(files) * len(metric_names)
    done_jobs = 0
    skipped_jobs = 0

    print("============================================================")
    print("ВЫЧИСЛЕНИЕ ОДИНОЧНЫХ МЕТРИК")
    print("============================================================")
    print(f"Dataset key            : {dataset_key}")
    print(f"Папка эмбеддингов      : {embeddings_dir}")
    print(f"Директория результатов : {dataset_dir}")
    print(f"Папка значений метрик  : {metrics_dir}")
    print(f"Папка артефактов       : {artifacts_dir}")
    print(f"Устройство             : {args.device}")
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

        existing_metric_paths = [
            build_metric_output_path(metrics_dir, metric_name, model_name)
            for metric_name in metric_names
        ]
        artifact_path = build_artifact_output_path(artifacts_dir, model_name)
        if (
            all(path.exists() for path in existing_metric_paths)
            and artifact_path.exists()
            and not args.overwrite
        ):
            skipped_jobs += len(metric_names)
            print(f"[ПРОПУСК] {model_name} -> значения и артефакты уже существуют")
            continue

        try:
            computed = compute_single_metrics(
                x=x,
                metric_names=metric_names,
                requested_device=str(args.device),
            )
            artifact_payload = {
                **computed.artifacts,
                "metric_names": np.asarray(metric_names, dtype=object),
                "source_file": np.asarray(str(emb_path.resolve()), dtype=object),
                "model_name": np.asarray(model_name, dtype=object),
                "dataset_key": np.asarray(dataset_key, dtype=object),
                "backend": np.asarray(computed.backend, dtype=object),
                "device": np.asarray(computed.device, dtype=object),
            }
            np.savez_compressed(artifact_path, **artifact_payload)
        except Exception as e:
            for metric_name in metric_names:
                errors.append(
                    {
                        "file": str(emb_path),
                        "model_name": model_name,
                        "metric_name": metric_name,
                        "error": repr(e),
                    }
                )
            print(f"[ОШИБКА] метрики не вычислены для {model_name}: {e}")
            if args.fail_fast:
                raise
            continue

        for metric_name in metric_names:
            spec = METRICS[metric_name]
            out_path = build_metric_output_path(metrics_dir, metric_name, model_name)
            try:
                value = float(computed.values[metric_name])

                payload = {
                    "schema_version": 2,
                    "dataset_key": dataset_key,
                    "metric_name": spec.name,
                    "model_name": model_name,
                    "source_file": str(emb_path.resolve()),
                    "artifact_file": str(artifact_path.resolve()),
                    "n_samples": int(x.shape[0]),
                    "dim": int(x.shape[1]),
                    "center": bool(args.center),
                    "row_l2_normalize": bool(args.row_l2_normalize),
                    "backend": computed.backend,
                    "device": computed.device,
                    "higher_is_better": bool(spec.higher_is_better),
                    "value": value,
                }
                save_json(out_path, payload)
                done_jobs += 1
                print(
                    f"[OK]   {metric_name} | {model_name} -> {value:.6f} "
                    f"({computed.backend}/{computed.device})"
                )

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
        "schema_version": 2,
        "dataset_key": dataset_key,
        "total_jobs": total_jobs,
        "computed_jobs": done_jobs,
        "skipped_jobs": skipped_jobs,
        "n_errors": len(errors),
        "errors": errors,
    }
    save_json(dataset_dir / "summary.json", summary)
    saved_rows = collect_saved_value_rows(metrics_dir)
    if saved_rows:
        save_values_csv(dataset_dir / "values.csv", saved_rows)

    print("============================================================")
    print("ВЫЧИСЛЕНИЕ ОДИНОЧНЫХ МЕТРИК ЗАВЕРШЕНО")
    print("============================================================")
    print(f"Вычислено              : {done_jobs}")
    print(f"Пропущено              : {skipped_jobs}")
    print(f"Ошибок                 : {len(errors)}")
    print("============================================================")


if __name__ == "__main__":
    main()
