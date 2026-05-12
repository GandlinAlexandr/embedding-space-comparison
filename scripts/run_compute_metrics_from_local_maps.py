from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm

from configs.benchmark_configs import VKR_2024_2025_MODEL_NAMES
from configs.text_benchmark_configs import (
    TEXT_EMBEDDING_CPU_MODEL_IDS,
    TEXT_EMBEDDING_SNOWFLAKE_MODEL_IDS,
    TEXT_EMBEDDING_TEXT20_MODEL_IDS,
)
import scripts.run_compute_embedding_metrics as legacy
import scripts.run_compute_single_metrics as single_metrics


NON_MODEL_STEMS = {
    "subset_indices",
    "labels",
    "targets",
    "embeddings_manifest",
    "subset_manifest",
}

LOCAL_MAP_AGGREGATIONS_ALL = (
    "rankme",
    "stable_rank",
    "nesum",
    "pseudo_condition_number",
    "alpha_req",
    "spectral_entropy",
    "hard_rank",
    "tail_spectrum_log_ratio",
)
LOCAL_MAP_BASELINE_COMPATIBLE_AGGREGATIONS = (
    "rankme",
    "stable_rank",
    "nesum",
    "pseudo_condition_number",
    "alpha_req",
)


def _parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _parse_model_names(raw: str) -> List[str]:
    names = _parse_csv(raw)
    aliases = {
        "primary": VKR_2024_2025_MODEL_NAMES,
        "cpu": TEXT_EMBEDDING_CPU_MODEL_IDS,
        "snowflake": TEXT_EMBEDDING_SNOWFLAKE_MODEL_IDS,
        "text20": TEXT_EMBEDDING_TEXT20_MODEL_IDS,
    }
    out: List[str] = []
    for name in names:
        out.extend(aliases.get(name, [name]))
    return list(dict.fromkeys(out))


def _parse_single_metric_names(raw: str) -> List[str]:
    names = _parse_csv(raw)
    if any(name.lower() == "all" for name in names):
        return sorted(single_metrics.METRICS.keys())
    missing = [name for name in names if name not in single_metrics.METRICS]
    if missing:
        raise ValueError(
            f"--single_metrics: неизвестные baseline-метрики: {missing}. "
            f"Доступные: {sorted(single_metrics.METRICS.keys())} или all"
        )
    return list(dict.fromkeys(names))


def _parse_aggregation_names(raw: str) -> List[str]:
    names = _parse_csv(raw)
    if any(name.lower() == "baseline_compatible" for name in names):
        return list(LOCAL_MAP_BASELINE_COMPATIBLE_AGGREGATIONS)
    if any(name.lower() == "all" for name in names):
        return list(LOCAL_MAP_AGGREGATIONS_ALL)
    missing = [name for name in names if name not in LOCAL_MAP_AGGREGATIONS_ALL]
    if missing:
        raise ValueError(
            f"--aggregations: неизвестные агрегаторы: {missing}. "
            "Доступные: "
            f"{list(LOCAL_MAP_AGGREGATIONS_ALL)}, baseline_compatible или all"
        )
    return list(dict.fromkeys(names))


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_model_names(maps_dir: Path, models_raw: str) -> List[str]:
    if (maps_dir / "model_names.json").exists():
        names = json.loads((maps_dir / "model_names.json").read_text(encoding="utf-8"))
    else:
        manifest = _load_json(maps_dir / "manifest.json")
        names = manifest["model_names"]
    names = [str(x) for x in names]
    requested = _parse_model_names(models_raw)
    if requested:
        missing = [m for m in requested if m not in names]
        if missing:
            raise ValueError(
                f"--models отсутствуют в store: {missing}. Доступные: {names}"
            )
        names = [m for m in names if m in set(requested)]
    return names


def _dataset_key_from_embeddings_dir(path: str) -> str:
    return Path(path).name


def _dataset_base_from_key(dataset_key: str) -> str:
    key = str(dataset_key)
    for split in ("_test", "_train", "_val", "_valid"):
        pos = key.find(split)
        if pos >= 0:
            return key[:pos]
    return key


def _sampled_dataset_key(
    dataset_key: str,
    sample_size: int,
    sample_seed: int,
    sample_strategy: str = "stratified",
) -> str:
    key = f"{dataset_key}_s{int(sample_size)}_seed{int(sample_seed)}"
    if str(sample_strategy):
        key = f"{key}_{sample_strategy}"
    return key


def _effective_dataset_key(
    dataset_key: str,
    sample_size: int,
    sample_seed: int,
    sample_strategy: str = "stratified",
) -> str:
    if int(sample_size) <= 0:
        return str(dataset_key)
    return _sampled_dataset_key(
        str(dataset_key),
        int(sample_size),
        int(sample_seed),
        str(sample_strategy),
    )


def _infer_downstream_json(dataset_key: str) -> str:
    dataset = _dataset_base_from_key(dataset_key)
    candidates = [
        Path("data") / "downstream" / f"{dataset_key}_mlp.json",
        Path("data") / "downstream" / f"{dataset_key}_linear_probe.json",
        Path("data") / "downstream" / f"{dataset}_mlp.json",
        Path("data") / "downstream" / f"{dataset}_linear_probe.json",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        "Не удалось автоматически найти downstream JSON. Проверены: "
        + ", ".join(str(p) for p in candidates)
    )


def _infer_embeddings_dir(dataset_key: str) -> Path:
    return Path("data") / "embeddings" / str(dataset_key)


def _infer_sampled_embeddings_dir(dataset_key: str) -> Path:
    return Path("data") / "embeddings" / "samples" / str(dataset_key)


def _parse_include_metric(
    metric_name: str,
) -> Tuple[str, int | None, str, str, List[int], str]:
    """
    Parse old-style metric names into constructor pieces.

    Returns: metric_name, fixed_k, selector, aggregation, k_list, store_key.
    """
    import re

    name = metric_name.strip()
    lower = name.lower()
    pair_agg = "antisym"
    if lower.endswith("_sym"):
        pair_agg = "sym"
        stem = lower[: -len("_sym")]
    elif lower.endswith("_antisym"):
        pair_agg = "antisym"
        stem = lower[: -len("_antisym")]
    else:
        stem = lower
        if lower.startswith("directed_"):
            pair_agg = "directed"

    m = re.fullmatch(r"lin_k(\d+)(?:_(.+))?", stem)
    if m:
        k = int(m.group(1))
        aggregation = m.group(2) or "rankme"
        return name, k, "fixed_k", aggregation, [k], f"k{k}"

    m = re.fullmatch(r"directed_k(\d+)(?:_(.+))?", stem)
    if m:
        k = int(m.group(1))
        aggregation = m.group(2) or "rankme"
        return name, k, "fixed_k", aggregation, [k], f"k{k}"

    m = re.fullmatch(r"lin_eps_(\d+)(?:_(.+))?", stem)
    if m:
        p = int(m.group(1))
        aggregation = m.group(2) or "rankme"
        return name, None, "fixed_store_key", aggregation, [], f"eps_p{p}"

    m = re.fullmatch(r"w_eps_(\d+)(?:_(rsc|ransac))?(?:_(.+))?", stem)
    if m:
        p = int(m.group(1))
        is_ransac = bool(m.group(2))
        aggregation = m.group(3) or "rankme"
        key = f"w_eps_p{p}_ransac" if is_ransac else f"w_eps_p{p}"
        return name, None, "fixed_store_key", aggregation, [], key

    m = re.fullmatch(r"rff_k(\d+)(?:_(.+))?", stem)
    if m:
        k = int(m.group(1))
        aggregation = m.group(2) or "rankme"
        return name, None, "fixed_store_key", aggregation, [], f"rff_k{k}"

    m = re.fullmatch(r"adaptive_k([0-9_]+)(?:_(.+))?", stem)
    if m:
        k_list = [int(x) for x in m.group(1).split("_") if x]
        aggregation = m.group(2) or "rankme"
        return name, None, "adaptive", aggregation, k_list, ""

    if stem.startswith("single_"):
        metric = stem[len("single_") :]
        if metric not in single_metrics.METRICS:
            raise ValueError(
                f"Неизвестная single metric '{metric}'. "
                f"Доступные: {sorted(single_metrics.METRICS.keys())}"
            )
        return name, None, "single_baseline", metric, [], ""

    raise ValueError(
        f"Не удалось разобрать metric include '{metric_name}'. "
        "Поддерживаются lin_k5_antisym, lin_k5_stable_rank_antisym, "
        "adaptive_k5_10_20_40_80_antisym и аналогичные."
    )


def _requested_from_include(
    include_raw: str,
) -> Tuple[List[Tuple[str, int | None, str, str, str]], List[int], List[str], str]:
    parsed = [_parse_include_metric(x) for x in _parse_csv(include_raw)]
    if not parsed:
        return [], [], [], "antisym"
    pair_aggs = {p[0]: p for p in []}
    del pair_aggs
    pair_agg_values = {
        (
            "directed"
            if p[2] == "fixed_k" and p[0].lower().startswith("directed_")
            else ("sym" if p[0].lower().endswith("_sym") else "antisym")
        )
        for p in parsed
    }
    if len(pair_agg_values) != 1:
        raise ValueError(
            "--include должен содержать метрики с одним pair_agg за запуск "
            f"(получено: {sorted(pair_agg_values)})"
        )
    required_ks = sorted({k for p in parsed for k in p[4]})
    required_store_keys = sorted({p[5] for p in parsed if p[5]})
    requested = [(p[0], p[1], p[2], p[3], p[5]) for p in parsed]
    return requested, required_ks, required_store_keys, next(iter(pair_agg_values))


def _find_maps_dir(
    *,
    maps_root: str,
    dataset_key: str,
    required_ks: List[int],
    required_store_keys: List[str],
    models: List[str],
    seed: int,
    n_centers: int,
    local_geometry_mode: str,
) -> Path:
    root = Path(maps_root) / dataset_key
    if root.exists():
        manifest_paths = list(root.glob("*/manifest.json"))
    else:
        manifest_paths = list(Path(maps_root).glob("**/manifest.json"))
    matches: List[Path] = []
    for manifest_path in manifest_paths:
        try:
            manifest = _load_json(manifest_path)
        except Exception:
            continue
        if dataset_key and str(manifest.get("dataset_key", dataset_key)) != str(
            dataset_key
        ):
            continue
        manifest_ks = {int(k) for k in manifest.get("k_list", [])}
        manifest_store_keys = {
            str(s.get("key"))
            for s in manifest.get("neighborhood_specs", [])
            if isinstance(s, dict) and s.get("key")
        }
        manifest_models = {str(m) for m in manifest.get("model_names", [])}
        if not set(required_ks).issubset(manifest_ks):
            continue
        if required_store_keys and not set(required_store_keys).issubset(
            manifest_store_keys
        ):
            continue
        if models and not set(models).issubset(manifest_models):
            continue
        if int(manifest.get("seed", seed)) != int(seed):
            continue
        if int(manifest.get("n_centers", n_centers)) != int(n_centers):
            continue
        if str(manifest.get("local_geometry_mode", "")) != str(local_geometry_mode):
            continue
        matches.append(manifest_path.parent)
    if not matches:
        raise FileNotFoundError(
            "Не найден compatible local-map store. Нужно построить его командой "
            "scripts.run_compute_local_map_store с теми же --dataset_key, --models, "
            "--k_list, --seed, --n_centers и --local_geometry_mode."
        )
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _store_key_to_required_parts(store_key: str) -> Tuple[List[int], List[str]]:
    if store_key.startswith("k") and store_key[1:].isdigit():
        return [int(store_key[1:])], [store_key]
    return [], [store_key]


def _find_maps_dirs_by_store_key(
    *,
    maps_root: str,
    dataset_key: str,
    store_keys: List[str],
    models: List[str],
    seed: int,
    n_centers: int,
    local_geometry_mode: str,
) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for key in sorted(set(store_keys)):
        req_ks, req_store_keys = _store_key_to_required_parts(key)
        out[key] = _find_maps_dir(
            maps_root=maps_root,
            dataset_key=dataset_key,
            required_ks=req_ks,
            required_store_keys=req_store_keys,
            models=models,
            seed=seed,
            n_centers=n_centers,
            local_geometry_mode=local_geometry_mode,
        )
    return out


def _pair_path(maps_dir: Path, model_i: str, model_j: str) -> Path:
    return maps_dir / "pairs" / f"{model_i}_to_{model_j}.npz"


def _load_pair(maps_dir: Path, model_i: str, model_j: str) -> Dict[str, Any]:
    path = _pair_path(maps_dir, model_i, model_j)
    if not path.exists():
        raise FileNotFoundError(f"Нет сохранённого направления: {path}")
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _adaptive_store_keys_from_name(metric_name: str) -> List[str]:
    import re

    stem = metric_name.lower()
    for suffix in ("_antisym", "_sym"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    m = re.fullmatch(r"adaptive_k([0-9_]+)(?:_.+)?", stem)
    if not m:
        return []
    return [f"k{int(x)}" for x in m.group(1).split("_") if x]


def _assert_same_centers(
    base: Dict[str, Any], other: Dict[str, Any], *, key: str
) -> None:
    a = np.asarray(base["center_indices"], dtype=np.int64).reshape(-1)
    b = np.asarray(other["center_indices"], dtype=np.int64).reshape(-1)
    if a.shape != b.shape or not np.array_equal(a, b):
        raise RuntimeError(
            f"Нельзя собрать adaptive из разных stores: center_indices отличаются для {key}."
        )


def _load_pair_for_store_keys(
    store_dirs_by_key: Dict[str, Path],
    model_i: str,
    model_j: str,
    store_keys: List[str],
) -> Dict[str, Any]:
    keys = list(dict.fromkeys(store_keys))
    if not keys:
        raise ValueError("store_keys пустой")
    merged: Dict[str, Any] | None = None
    errors: List[np.ndarray] = []
    k_candidates: List[int] = []
    for key in keys:
        pair = _load_pair(store_dirs_by_key[key], model_i, model_j)
        if merged is None:
            merged = {
                k: v
                for k, v in pair.items()
                if "/" not in k
                and k not in {"k_candidates", "center_prediction_errors", "selected_ks"}
            }
        else:
            _assert_same_centers(merged, pair, key=key)
        for field_key, value in pair.items():
            if field_key.startswith(f"{key}/"):
                merged[field_key] = value
        if key.startswith("k") and key[1:].isdigit():
            k_candidates.append(int(key[1:]))
            err = np.asarray(pair["center_prediction_errors"], dtype=np.float32)
            pair_ks = [
                int(x)
                for x in np.asarray(pair["k_candidates"], dtype=np.int32).reshape(-1)
            ]
            if int(key[1:]) not in pair_ks:
                raise RuntimeError(
                    f"Хранилище {store_dirs_by_key[key]} не содержит {key}."
                )
            errors.append(err[:, [pair_ks.index(int(key[1:]))]])

    assert merged is not None
    if k_candidates:
        order = np.argsort(np.asarray(k_candidates, dtype=np.int32))
        ordered_ks = np.asarray(k_candidates, dtype=np.int32)[order]
        ordered_errors = [errors[int(pos)] for pos in order]
        center_errors = np.concatenate(ordered_errors, axis=1)
        selected_pos = np.argmin(center_errors, axis=1)
        merged["k_candidates"] = ordered_ks
        merged["center_prediction_errors"] = center_errors
        merged["selected_ks"] = ordered_ks[selected_pos].astype(np.int32)
    return merged


def _discover_models_from_embeddings(
    dataset_key: str, sampled: bool = False
) -> List[str]:
    embeddings_dir = (
        _infer_sampled_embeddings_dir(dataset_key)
        if sampled
        else _infer_embeddings_dir(dataset_key)
    )
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Не найдена папка эмбеддингов: {embeddings_dir}")
    names = []
    for path in sorted(embeddings_dir.iterdir()):
        if (
            path.suffix.lower() in {".npy", ".npz", ".pt", ".pth"}
            and path.stem not in NON_MODEL_STEMS
        ):
            names.append(path.stem)
    if not names:
        raise FileNotFoundError(f"В {embeddings_dir} не найдены embedding-файлы")
    return names


def _single_metric_path(dataset_key: str, metric_name: str, model_name: str) -> Path:
    return (
        Path("data")
        / "single_metrics"
        / str(dataset_key)
        / "metrics"
        / metric_name
        / f"{model_name}.json"
    )


def _ensure_single_metric_values(
    dataset_key: str,
    model_names: List[str],
    metric_names: List[str],
    models_raw: str,
) -> None:
    missing = []
    for metric_name in metric_names:
        for model_name in model_names:
            if not _single_metric_path(dataset_key, metric_name, model_name).exists():
                missing.append((metric_name, model_name))
    if not missing:
        return
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_compute_single_metrics",
        "--dataset_key",
        str(dataset_key),
        "--metrics",
        *metric_names,
    ]
    if models_raw:
        cmd.extend(["--models", str(models_raw)])
    print("\nНе найдены значения одиночных базовых метрик; считаю их:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_single_scores(
    dataset_key: str,
    metric_name: str,
    model_names: List[str],
) -> Tuple[Dict[str, float], bool]:
    scores: Dict[str, float] = {}
    higher_is_better: bool | None = None
    for model_name in model_names:
        path = _single_metric_path(dataset_key, metric_name, model_name)
        if not path.exists():
            raise FileNotFoundError(f"Не найден файл одиночной метрики: {path}")
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        scores[model_name] = float(payload["value"])
        hib = bool(payload.get("higher_is_better", True))
        if higher_is_better is None:
            higher_is_better = hib
        elif higher_is_better != hib:
            raise ValueError(f"Несогласованный higher_is_better для {metric_name}")
    return scores, bool(higher_is_better)


def _single_baseline_matrix(
    dataset_key: str,
    metric_name: str,
    model_names: List[str],
    pair_agg: str,
) -> np.ndarray:
    scores, higher_is_better = _load_single_scores(
        dataset_key, metric_name, model_names
    )
    aligned = dict(scores) if higher_is_better else {m: -v for m, v in scores.items()}
    matrix = np.full((len(model_names), len(model_names)), np.nan, dtype=np.float32)
    np.fill_diagonal(matrix, 0.0)
    for i, mi in enumerate(model_names):
        for j in range(i + 1, len(model_names)):
            mj = model_names[j]
            delta = float(aligned[mj] - aligned[mi])
            if pair_agg == "sym":
                val = np.float32(abs(delta))
                matrix[i, j] = val
                matrix[j, i] = val
            else:
                matrix[i, j] = np.float32(delta)
    return matrix


def _spectral_entropy_from_singular_values(s: np.ndarray) -> float:
    arr = np.asarray(s, dtype=np.float64).reshape(-1)
    arr = np.abs(arr[np.isfinite(arr)])
    if arr.size == 0:
        return float("nan")
    p = arr / (float(np.sum(arr)) + 1e-10)
    return float(-np.sum(p * np.log(p + 1e-10)))


def _aggregate_singular_values(
    s: np.ndarray,
    aggregation: str,
    *,
    hard_rank_threshold: float,
    weak_spectrum_count: int,
) -> float:
    if aggregation == "rankme":
        return legacy.rankme(s)
    if aggregation == "stable_rank":
        return single_metrics.stable_rank_from_singular_values(s)
    if aggregation == "nesum":
        return single_metrics.nesum_from_singular_values(s)
    if aggregation == "pseudo_condition_number":
        return single_metrics.pseudo_condition_number_from_singular_values(s)
    if aggregation == "alpha_req":
        return single_metrics.alpha_req_from_singular_values(s)
    if aggregation == "spectral_entropy":
        return _spectral_entropy_from_singular_values(s)
    if aggregation == "hard_rank":
        return legacy.hard_rank(s, threshold=hard_rank_threshold)
    if aggregation == "tail_spectrum_log_ratio":
        return legacy.tail_spectrum_log_ratio(
            s,
            weak_spectrum_count=weak_spectrum_count,
        )
    raise ValueError(f"Неизвестная aggregation: {aggregation}")


def _metric_name(
    *,
    selector: str,
    k: int | None,
    k_list: List[int],
    aggregation: str,
    pair_agg: str,
) -> str:
    agg_suffix = "" if aggregation == "rankme" else f"_{aggregation}"
    pair_suffix = "" if pair_agg == "directed" else f"_{pair_agg}"
    if selector == "fixed_k":
        assert k is not None
        if pair_agg == "directed" and aggregation == "rankme":
            return f"directed_k{k}"
        return f"lin_k{k}{agg_suffix}{pair_suffix}"
    if selector == "adaptive":
        k_part = "_".join(str(x) for x in k_list)
        return f"adaptive_k{k_part}{agg_suffix}{pair_suffix}"
    raise ValueError(f"Неизвестный selector: {selector}")


def _available_k_list(pair: Dict[str, Any]) -> List[int]:
    return [
        int(x) for x in np.asarray(pair["k_candidates"], dtype=np.int32).reshape(-1)
    ]


def _direction_value_fixed(
    pair: Dict[str, Any],
    *,
    store_key: str,
    aggregation: str,
    hard_rank_threshold: float,
    weak_spectrum_count: int,
) -> float:
    if aggregation == "rankme" and f"{store_key}/metric_rankme" in pair:
        vals = np.asarray(pair[f"{store_key}/metric_rankme"], dtype=np.float64).reshape(
            -1
        )
        return float(np.nanmean(vals))
    sv = np.asarray(pair[f"{store_key}/singular_values"], dtype=np.float64)
    vals = [
        _aggregate_singular_values(
            row,
            aggregation,
            hard_rank_threshold=hard_rank_threshold,
            weak_spectrum_count=weak_spectrum_count,
        )
        for row in sv
    ]
    return float(np.nanmean(np.asarray(vals, dtype=np.float64)))


def _direction_value_adaptive(
    pair: Dict[str, Any],
    *,
    aggregation: str,
    hard_rank_threshold: float,
    weak_spectrum_count: int,
) -> float:
    k_list = _available_k_list(pair)
    selected = np.asarray(pair["selected_ks"], dtype=np.int32).reshape(-1)
    vals: List[float] = []
    for center_pos, k in enumerate(selected):
        k_int = int(k)
        if k_int not in k_list:
            vals.append(float("nan"))
            continue
        if aggregation == "rankme" and f"k{k_int}/metric_rankme" in pair:
            vals.append(float(np.asarray(pair[f"k{k_int}/metric_rankme"])[center_pos]))
        else:
            sv = np.asarray(pair[f"k{k_int}/singular_values"], dtype=np.float64)[
                center_pos
            ]
            vals.append(
                _aggregate_singular_values(
                    sv,
                    aggregation,
                    hard_rank_threshold=hard_rank_threshold,
                    weak_spectrum_count=weak_spectrum_count,
                )
            )
    return float(np.nanmean(np.asarray(vals, dtype=np.float64)))


def _direction_value(
    maps_dir: Path,
    model_i: str,
    model_j: str,
    *,
    selector: str,
    k: int | None,
    store_key: str,
    aggregation: str,
    hard_rank_threshold: float,
    weak_spectrum_count: int,
) -> float:
    pair = _load_pair(maps_dir, model_i, model_j)
    if selector in {"fixed_k", "fixed_store_key"}:
        if selector == "fixed_k":
            assert k is not None
            store_key = f"k{k}"
        return _direction_value_fixed(
            pair,
            store_key=store_key,
            aggregation=aggregation,
            hard_rank_threshold=hard_rank_threshold,
            weak_spectrum_count=weak_spectrum_count,
        )
    if selector == "adaptive":
        return _direction_value_adaptive(
            pair,
            aggregation=aggregation,
            hard_rank_threshold=hard_rank_threshold,
            weak_spectrum_count=weak_spectrum_count,
        )
    raise ValueError(f"Неизвестный selector: {selector}")


def _object_array(items: List[np.ndarray]) -> np.ndarray:
    arr = np.empty(len(items), dtype=object)
    for idx, item in enumerate(items):
        arr[idx] = item
    return arr


def _artifact_metric_values(
    singular_values: np.ndarray,
    aggregation: str,
    *,
    hard_rank_threshold: float,
    weak_spectrum_count: int,
) -> np.ndarray:
    vals = [
        _aggregate_singular_values(
            row,
            aggregation,
            hard_rank_threshold=hard_rank_threshold,
            weak_spectrum_count=weak_spectrum_count,
        )
        for row in np.asarray(singular_values, dtype=np.float64)
    ]
    return np.asarray(vals, dtype=np.float32)


def _artifact_hard_ranks(singular_values: np.ndarray) -> np.ndarray:
    vals: List[int] = []
    for row in np.asarray(singular_values, dtype=np.float64):
        s = np.asarray(row, dtype=np.float64).reshape(-1)
        tol = 1e-10 * float(np.max(s)) if s.size > 0 else 0.0
        vals.append(int(np.sum(s > tol)))
    return np.asarray(vals, dtype=np.int32)


def _artifact_object_field(
    pair: Dict[str, Any],
    key: str,
    *,
    fallback_count: int,
    fallback_size: int = 0,
    fill: float = 1.0,
    dtype: Any = np.float32,
) -> np.ndarray:
    if key in pair:
        raw = pair[key]
        if np.asarray(raw).dtype == object:
            return raw
        return _object_array([np.asarray(x, dtype=dtype) for x in raw])
    return _object_array(
        [np.full((fallback_size,), fill, dtype=dtype) for _ in range(fallback_count)]
    )


def _artifact_fixed_from_pair(
    pair: Dict[str, Any],
    *,
    store_key: str,
    aggregation: str,
    hard_rank_threshold: float,
    weak_spectrum_count: int,
) -> Dict[str, Any]:
    sv = np.asarray(pair[f"{store_key}/singular_values"], dtype=np.float64)
    c_count = int(sv.shape[0])
    neighbor_indices = pair.get(f"{store_key}/neighbor_indices")
    if f"{store_key}/neighbor_sizes" in pair:
        neighbor_sizes = np.asarray(pair[f"{store_key}/neighbor_sizes"], dtype=np.int32)
    elif neighbor_indices is not None and np.asarray(neighbor_indices).dtype != object:
        neighbor_sizes = np.full(
            (c_count,), int(np.asarray(neighbor_indices).shape[1]), dtype=np.int32
        )
    else:
        neighbor_sizes = np.full((c_count,), 0, dtype=np.int32)

    if f"{store_key}/metric_rankme" in pair and aggregation == "rankme":
        metric_ranks = np.asarray(pair[f"{store_key}/metric_rankme"], dtype=np.float32)
    else:
        metric_ranks = _artifact_metric_values(
            sv,
            aggregation,
            hard_rank_threshold=hard_rank_threshold,
            weak_spectrum_count=weak_spectrum_count,
        )

    sample_weights = _artifact_object_field(
        pair,
        f"{store_key}/sample_weights",
        fallback_count=c_count,
        fallback_size=(
            int(neighbor_sizes[0])
            if neighbor_sizes.size and np.all(neighbor_sizes == neighbor_sizes[0])
            else 0
        ),
        fill=1.0,
        dtype=np.float32,
    )
    if f"{store_key}/sample_weights" not in pair:
        sample_weights = _object_array(
            [np.ones((int(size),), dtype=np.float32) for size in neighbor_sizes]
        )

    inlier_masks = _artifact_object_field(
        pair,
        f"{store_key}/inlier_masks",
        fallback_count=c_count,
        dtype=bool,
    )
    if f"{store_key}/inlier_masks" not in pair:
        inlier_masks = _object_array(
            [np.ones((int(size),), dtype=bool) for size in neighbor_sizes]
        )
    inlier_counts = np.asarray(
        [int(np.sum(mask)) for mask in inlier_masks], dtype=np.int32
    )
    inlier_fracs = np.asarray(
        [
            float(np.sum(mask) / len(mask)) if len(mask) > 0 else float("nan")
            for mask in inlier_masks
        ],
        dtype=np.float32,
    )

    return {
        "singular_values": _object_array(
            [np.asarray(row, dtype=np.float64) for row in sv]
        ),
        "residuals": np.asarray(pair[f"{store_key}/residuals"], dtype=np.float32),
        "relative_residuals": np.asarray(
            pair[f"{store_key}/relative_residuals"], dtype=np.float32
        ),
        "ranks": _artifact_hard_ranks(sv),
        "metric_ranks": metric_ranks,
        "neighbor_sizes": neighbor_sizes,
        "neighbor_distances": _artifact_object_field(
            pair,
            f"{store_key}/neighbor_distances",
            fallback_count=c_count,
            dtype=np.float32,
        ),
        "sigma_values": np.asarray(
            pair.get(f"{store_key}/sigma_values", np.full((c_count,), np.nan)),
            dtype=np.float32,
        ),
        "eps_values": np.asarray(
            pair.get(f"{store_key}/eps_values", np.full((c_count,), np.nan)),
            dtype=np.float32,
        ),
        "sample_weights": sample_weights,
        "inlier_masks": inlier_masks,
        "inlier_counts": inlier_counts,
        "inlier_fracs": inlier_fracs,
        "selected_ks": np.zeros((c_count,), dtype=np.int32),
        "center_prediction_errors": _object_array(
            [np.zeros((0,), dtype=np.float32) for _ in range(c_count)]
        ),
    }


def _artifact_adaptive_from_pair(
    pair: Dict[str, Any],
    *,
    aggregation: str,
    hard_rank_threshold: float,
    weak_spectrum_count: int,
) -> Dict[str, Any]:
    selected = np.asarray(pair["selected_ks"], dtype=np.int32).reshape(-1)
    center_errors = np.asarray(pair["center_prediction_errors"], dtype=np.float32)
    rows: Dict[str, List[Any]] = {
        "singular_values": [],
        "residuals": [],
        "relative_residuals": [],
        "metric_ranks": [],
        "neighbor_sizes": [],
        "neighbor_distances": [],
        "sample_weights": [],
        "inlier_masks": [],
    }
    for center_pos, k in enumerate(selected):
        key = f"k{int(k)}"
        sv = np.asarray(pair[f"{key}/singular_values"], dtype=np.float64)[center_pos]
        rows["singular_values"].append(sv)
        rows["residuals"].append(
            float(np.asarray(pair[f"{key}/residuals"])[center_pos])
        )
        rows["relative_residuals"].append(
            float(np.asarray(pair[f"{key}/relative_residuals"])[center_pos])
        )
        if aggregation == "rankme" and f"{key}/metric_rankme" in pair:
            rows["metric_ranks"].append(
                float(np.asarray(pair[f"{key}/metric_rankme"])[center_pos])
            )
        else:
            rows["metric_ranks"].append(
                _aggregate_singular_values(
                    sv,
                    aggregation,
                    hard_rank_threshold=hard_rank_threshold,
                    weak_spectrum_count=weak_spectrum_count,
                )
            )
        dists = np.asarray(pair[f"{key}/neighbor_distances"])[center_pos]
        rows["neighbor_distances"].append(np.asarray(dists, dtype=np.float32))
        rows["neighbor_sizes"].append(int(np.asarray(dists).reshape(-1).size))
        rows["sample_weights"].append(
            np.ones((rows["neighbor_sizes"][-1],), dtype=np.float32)
        )
        if f"{key}/inlier_masks" in pair:
            rows["inlier_masks"].append(
                np.asarray(pair[f"{key}/inlier_masks"])[center_pos].astype(bool)
            )
        else:
            rows["inlier_masks"].append(
                np.ones((rows["neighbor_sizes"][-1],), dtype=bool)
            )

    sv_obj = _object_array(
        [np.asarray(x, dtype=np.float64) for x in rows["singular_values"]]
    )
    inlier_masks = _object_array(
        [np.asarray(x, dtype=bool) for x in rows["inlier_masks"]]
    )
    inlier_counts = np.asarray(
        [int(np.sum(mask)) for mask in inlier_masks], dtype=np.int32
    )
    inlier_fracs = np.asarray(
        [
            float(np.sum(mask) / len(mask)) if len(mask) > 0 else float("nan")
            for mask in inlier_masks
        ],
        dtype=np.float32,
    )
    return {
        "singular_values": sv_obj,
        "residuals": np.asarray(rows["residuals"], dtype=np.float32),
        "relative_residuals": np.asarray(rows["relative_residuals"], dtype=np.float32),
        "ranks": _artifact_hard_ranks(
            np.asarray(rows["singular_values"], dtype=np.float64)
        ),
        "metric_ranks": np.asarray(rows["metric_ranks"], dtype=np.float32),
        "neighbor_sizes": np.asarray(rows["neighbor_sizes"], dtype=np.int32),
        "neighbor_distances": _object_array(
            [np.asarray(x, dtype=np.float32) for x in rows["neighbor_distances"]]
        ),
        "sigma_values": np.full((selected.size,), np.nan, dtype=np.float32),
        "eps_values": np.full((selected.size,), np.nan, dtype=np.float32),
        "sample_weights": _object_array(
            [np.asarray(x, dtype=np.float32) for x in rows["sample_weights"]]
        ),
        "inlier_masks": inlier_masks,
        "inlier_counts": inlier_counts,
        "inlier_fracs": inlier_fracs,
        "selected_ks": selected.astype(np.int32),
        "center_prediction_errors": _object_array(
            [
                np.asarray(center_errors[pos], dtype=np.float32)
                for pos in range(selected.size)
            ]
        ),
    }


def _artifact_from_pair(
    pair: Dict[str, Any],
    *,
    selector: str,
    k: int | None,
    store_key: str,
    aggregation: str,
    hard_rank_threshold: float,
    weak_spectrum_count: int,
) -> Dict[str, Any]:
    if selector == "fixed_k":
        assert k is not None
        store_key = f"k{k}"
    if selector in {"fixed_k", "fixed_store_key"}:
        return _artifact_fixed_from_pair(
            pair,
            store_key=store_key,
            aggregation=aggregation,
            hard_rank_threshold=hard_rank_threshold,
            weak_spectrum_count=weak_spectrum_count,
        )
    if selector == "adaptive":
        return _artifact_adaptive_from_pair(
            pair,
            aggregation=aggregation,
            hard_rank_threshold=hard_rank_threshold,
            weak_spectrum_count=weak_spectrum_count,
        )
    raise ValueError(f"Неизвестный selector: {selector}")


def _add_direction_artifacts(
    artifacts: Dict[str, Any],
    *,
    prefix: str,
    direction_artifacts: Dict[str, Any],
) -> None:
    for key, value in direction_artifacts.items():
        artifacts[f"{prefix}/{key}"] = value


def _save_metric_artifacts(
    path: Path,
    artifacts: Dict[str, Any],
    meta: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifacts)
    payload["diagnostics_meta_json"] = json.dumps(meta, ensure_ascii=False)
    np.savez_compressed(path, **payload)


def _save_matrix(
    out_path: Path,
    matrix: np.ndarray,
    model_names: List[str],
    meta: Dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        matrix=np.asarray(matrix, dtype=np.float32),
        model_names=np.array(model_names, dtype=object),
        meta_json=json.dumps(meta, ensure_ascii=False),
    )


def _iter_requested_metrics(
    selectors: Iterable[str],
    fixed_ks: Iterable[int],
    aggregations: Iterable[str],
    pair_agg: str,
    k_list: List[int],
) -> Iterable[Tuple[str, int | None, str, str, str]]:
    for selector in selectors:
        if selector == "fixed_k":
            for k in fixed_ks:
                for aggregation in aggregations:
                    name = _metric_name(
                        selector=selector,
                        k=int(k),
                        k_list=k_list,
                        aggregation=aggregation,
                        pair_agg=pair_agg,
                    )
                    yield name, int(k), selector, aggregation, f"k{int(k)}"
        elif selector == "adaptive":
            for aggregation in aggregations:
                name = _metric_name(
                    selector=selector,
                    k=None,
                    k_list=k_list,
                    aggregation=aggregation,
                    pair_agg=pair_agg,
                )
                yield name, None, selector, aggregation, ""
        else:
            raise ValueError(f"Неизвестный selector: {selector}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Вычислить фиксированные и адаптивные попарные метрики "
            "по сохранённому хранилищу локальных отображений."
        )
    )
    parser.add_argument(
        "--maps_dir",
        default="",
        help=(
            "Папка хранилища локальных отображений. Если пусто, "
            "ищется автоматически по --dataset_key."
        ),
    )
    parser.add_argument(
        "--embeddings_dir",
        default="",
        help=(
            "Старый псевдоним для вывода dataset_key; расчёт из локальных "
            "отображений эмбеддинги не читает."
        ),
    )
    parser.add_argument(
        "--maps_root",
        default=str(Path("data") / "local_maps"),
        help="Корень хранилищ локальных отображений для автопоиска.",
    )
    parser.add_argument(
        "--experiment_dir",
        default="",
        help=(
            "Если задано, использовать стандартную структуру: "
            "metric_matrices/<dataset_key>, reports/, reports/plots/."
        ),
    )
    parser.add_argument("--out_dir", default="")
    parser.add_argument(
        "--dataset_key",
        default="",
        help=(
            "Имя набора данных для стандартных путей и заголовков. "
            "Если пусто, берётся из манифеста."
        ),
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help=(
            "Если >0, использовать ключ подвыборки "
            "<dataset_key>_sN_seedS_<sample_strategy> и папки samples."
        ),
    )
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument(
        "--sample_strategy", choices=["stratified", "random"], default="stratified"
    )
    parser.add_argument(
        "--downstream_json",
        default="",
        help=(
            "JSON с качеством моделей. Если пусто и не задан --skip_eval, "
            "автоматически ищется data/downstream/<dataset>_mlp.json."
        ),
    )
    parser.add_argument("--tasks", default="")
    parser.add_argument(
        "--eval_protocol",
        choices=["delta_signed", "delta_abs"],
        default="delta_signed",
    )
    parser.add_argument(
        "--reports_dir",
        default="",
        help=(
            "Куда сохранять CSV/MD-отчёты. Если пусто и задан "
            "--experiment_dir, используется reports/."
        ),
    )
    parser.add_argument(
        "--plots_dir",
        default="",
        help=(
            "Куда сохранять точечные графики оценки. Если пусто и задан "
            "--experiment_dir, используется reports/plots/scatter."
        ),
    )
    parser.add_argument(
        "--summary_plots_dir",
        default="",
        help=(
            "Куда сохранять сводный график. Если пусто и задан "
            "--experiment_dir, используется reports/plots/summary."
        ),
    )
    parser.add_argument("--plots_ext", default="png")
    parser.add_argument(
        "--plots_mode",
        choices=["none", "all", "alltasks", "tasks"],
        default="alltasks",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Не запускать отчёты и графики после расчёта матриц метрик.",
    )
    parser.add_argument("--models", default="")
    parser.add_argument(
        "--include",
        default="",
        help=(
            "Старый стиль: имена метрик через запятую, например "
            "single_rankme_antisym,lin_k5_antisym,adaptive_k5_10_20_40_80_antisym. "
            "Если задано, --selectors/--fixed_ks/--aggregations не нужны."
        ),
    )
    parser.add_argument(
        "--selectors",
        default="fixed_k,adaptive",
        help="Через запятую: fixed_k,adaptive.",
    )
    parser.add_argument(
        "--fixed_ks",
        default="",
        help="k для fixed_k. Если пусто, берутся все k из store.",
    )
    parser.add_argument(
        "--k_list",
        default="",
        help=(
            "Кандидаты k, которые должны быть в хранилище. Удобно для "
            "адаптивного запуска без selector fixed_k."
        ),
    )
    parser.add_argument(
        "--aggregations",
        default="rankme",
        help=(
            "rankme,stable_rank,nesum,pseudo_condition_number,alpha_req,"
            "spectral_entropy,hard_rank,tail_spectrum_log_ratio; "
            "baseline_compatible; или all"
        ),
    )
    parser.add_argument(
        "--single_metrics",
        default="",
        help=(
            "Одиночные базовые метрики через запятую или all. "
            "Будут посчитаны в этом же запуске и попадут в те же CSV/графики."
        ),
    )
    parser.add_argument(
        "--pair_agg",
        choices=["directed", "antisym", "sym"],
        default="antisym",
    )
    parser.add_argument("--hard_rank_threshold", type=float, default=1e-2)
    parser.add_argument("--weak_spectrum_count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_centers", type=int, default=200)
    parser.add_argument(
        "--local_geometry_mode",
        choices=list(legacy._LOCAL_GEOMETRY_MODE_CHOICES),
        default="centered_offsets_v2",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Принят для совместимости CLI; расчет из store не решает МНК заново.",
    )
    parser.add_argument("--incremental", action="store_true")
    args = parser.parse_args()

    requested_from_include: List[Tuple[str, int | None, str, str, str]] = []
    required_ks_from_include: List[int] = []
    required_store_keys_from_include: List[str] = []
    pair_agg_from_include: str | None = None
    if args.include:
        (
            requested_from_include,
            required_ks_from_include,
            required_store_keys_from_include,
            pair_agg_from_include,
        ) = _requested_from_include(args.include)
        args.pair_agg = pair_agg_from_include

    single_metrics_requested = _parse_single_metric_names(args.single_metrics)
    requested = list(requested_from_include)
    needs_maps = True
    if requested_from_include:
        needs_maps = any(
            selector != "single_baseline" for _, _, selector, _, _ in requested
        )

    if not args.dataset_key:
        if args.maps_dir:
            args.dataset_key = Path(args.maps_dir).parent.name
        elif args.embeddings_dir:
            # Совместимость со старым CLI
            args.dataset_key = _dataset_key_from_embeddings_dir(args.embeddings_dir)
    base_dataset_key = str(args.dataset_key)
    sampled = bool(int(args.sample_size) > 0)
    if base_dataset_key:
        dataset_key = _effective_dataset_key(
            base_dataset_key,
            int(args.sample_size),
            int(args.sample_seed),
            str(args.sample_strategy),
        )
    else:
        dataset_key = ""

    requested_models = _parse_model_names(args.models)
    maps_dirs_by_key: Dict[str, Path] = {}
    effective_maps_root = (
        str(Path(args.maps_root) / "samples")
        if sampled and Path(args.maps_root) == Path("data") / "local_maps"
        else args.maps_root
    )
    if needs_maps and not args.maps_dir:
        if not dataset_key:
            raise ValueError("Нужно указать либо --maps_dir, либо --dataset_key.")
        if not required_ks_from_include and not required_store_keys_from_include:
            required_ks_from_include = (
                _parse_csv_ints(args.fixed_ks)
                if args.fixed_ks
                else _parse_csv_ints(args.k_list)
            )
        if not required_ks_from_include and not required_store_keys_from_include:
            raise ValueError(
                "Для автопоиска store укажи --include или --fixed_ks, чтобы понять нужные окрестности."
            )
        all_required_store_keys = sorted(
            set(required_store_keys_from_include)
            | {f"k{int(k)}" for k in required_ks_from_include}
        )
        try:
            args.maps_dir = str(
                _find_maps_dir(
                    maps_root=effective_maps_root,
                    dataset_key=dataset_key,
                    required_ks=required_ks_from_include,
                    required_store_keys=required_store_keys_from_include,
                    models=requested_models,
                    seed=int(args.seed),
                    n_centers=int(args.n_centers),
                    local_geometry_mode=str(args.local_geometry_mode),
                )
            )
        except FileNotFoundError:
            maps_dirs_by_key = _find_maps_dirs_by_store_key(
                maps_root=effective_maps_root,
                dataset_key=dataset_key,
                store_keys=all_required_store_keys,
                models=requested_models,
                seed=int(args.seed),
                n_centers=int(args.n_centers),
                local_geometry_mode=str(args.local_geometry_mode),
            )

    maps_dir = Path(args.maps_dir) if args.maps_dir else None
    manifest: Dict[str, Any] = {}
    if needs_maps:
        if maps_dir is not None:
            manifest = _load_json(maps_dir / "manifest.json")
        elif maps_dirs_by_key:
            first_dir = next(iter(maps_dirs_by_key.values()))
            manifest = _load_json(first_dir / "manifest.json")
        else:
            raise ValueError("Не найдено хранилище локальных отображений/спектров.")
        if not dataset_key:
            dataset_key = str(manifest.get("dataset_key", ""))
    elif not dataset_key:
        raise ValueError("Для single_* метрик нужно указать --dataset_key.")
    if args.experiment_dir:
        experiment_dir = Path(args.experiment_dir)
        if not args.out_dir:
            args.out_dir = str(experiment_dir / "metric_matrices" / dataset_key)
        if not args.reports_dir:
            args.reports_dir = str(experiment_dir / "reports")
        if not args.plots_dir:
            args.plots_dir = str(experiment_dir / "plots" / f"{dataset_key}_signed")
        if not args.summary_plots_dir:
            args.summary_plots_dir = str(experiment_dir)
    if not args.out_dir:
        raise ValueError("Нужно указать либо --out_dir, либо --experiment_dir.")
    out_dir = Path(args.out_dir)
    if needs_maps:
        model_source_dir = (
            maps_dir if maps_dir is not None else next(iter(maps_dirs_by_key.values()))
        )
        model_names = _load_model_names(model_source_dir, args.models)
    elif requested_models:
        model_names = requested_models
    else:
        model_names = _discover_models_from_embeddings(dataset_key, sampled=sampled)
    if len(model_names) < 2:
        raise ValueError("Нужно минимум две модели")
    out_dir.mkdir(parents=True, exist_ok=True)

    k_list: List[int] = []
    fixed_ks: List[int] = []
    if needs_maps:
        # Read one pair to discover candidate k values.
        if maps_dir is not None:
            probe = _load_pair(maps_dir, model_names[0], model_names[1])
        else:
            probe = _load_pair_for_store_keys(
                maps_dirs_by_key,
                model_names[0],
                model_names[1],
                sorted(maps_dirs_by_key.keys()),
            )
        k_list = _available_k_list(probe) if "k_candidates" in probe else []
        fixed_ks = (
            _parse_csv_ints(args.fixed_ks)
            if args.fixed_ks
            else (_parse_csv_ints(args.k_list) if args.k_list else k_list)
        )
        missing_k = [k for k in fixed_ks if k not in k_list]
        if missing_k:
            raise ValueError(
                f"fixed_ks отсутствуют в store: {missing_k}. Доступные: {k_list}"
            )

    selectors = _parse_csv(args.selectors)
    aggregations = _parse_aggregation_names(args.aggregations)
    legacy._save_model_list(str(out_dir), model_names)

    if not requested_from_include:
        requested = list(
            _iter_requested_metrics(
                selectors=selectors,
                fixed_ks=fixed_ks,
                aggregations=aggregations,
                pair_agg=str(args.pair_agg),
                k_list=k_list,
            )
        )
    requested.extend(
        (f"single_{metric_name}", None, "single_baseline", metric_name, "")
        for metric_name in single_metrics_requested
    )
    single_metric_names = sorted(
        {
            aggregation
            for _, _, selector, aggregation, _ in requested
            if selector == "single_baseline"
        }
    )
    if single_metric_names:
        if sampled:
            missing = [
                (metric_name, model_name)
                for metric_name in single_metric_names
                for model_name in model_names
                if not _single_metric_path(
                    dataset_key, metric_name, model_name
                ).exists()
            ]
            if missing:
                cmd = [
                    sys.executable,
                    "-m",
                    "scripts.run_compute_single_metrics",
                    "--dataset_key",
                    str(base_dataset_key),
                    "--sample_size",
                    str(int(args.sample_size)),
                    "--sample_seed",
                    str(int(args.sample_seed)),
                    "--sample_strategy",
                    str(args.sample_strategy),
                    "--metrics",
                    *single_metric_names,
                ]
                if args.models:
                    cmd.extend(["--models", str(args.models)])
                print(
                    "\nНе найдены значения одиночных базовых метрик; "
                    "считаю значения для подвыборки:"
                )
                print(" ".join(cmd))
                subprocess.run(cmd, check=True)
        else:
            _ensure_single_metric_values(
                dataset_key,
                model_names,
                single_metric_names,
                str(args.models),
            )

    print(
        "Папка локальных отображений: "
        f"{maps_dir if maps_dir is not None else '<не используется>'}"
    )
    print(f"Папка вывода: {out_dir}")
    print(f"Моделей: {len(model_names)}")
    print(f"Метрик: {len(requested)}")

    for metric_name, fixed_k, selector, aggregation, store_key in requested:
        out_path = out_dir / f"{metric_name}.npz"
        if args.incremental and out_path.exists():
            print(f"[пропуск] {out_path}")
            continue

        if selector == "single_baseline":
            matrix = _single_baseline_matrix(
                dataset_key,
                aggregation,
                model_names,
                str(args.pair_agg),
            )
            meta = {
                "metric_name": metric_name,
                "is_paired": True,
                "is_symmetric": bool(args.pair_agg in {"antisym", "sym"}),
                "pair_agg": str(args.pair_agg),
                "selector": selector,
                "single_metric_name": aggregation,
                "source_single_metrics_dir": str(
                    (Path("data") / "single_metrics" / dataset_key).resolve()
                ),
                "constructor_schema_version": 1,
            }
            _save_matrix(out_path, matrix, model_names, meta)
            print(f"Сохранено: {out_path}")
            continue

        if maps_dir is None and not maps_dirs_by_key:
            raise ValueError(
                f"Метрика {metric_name} требует хранилище локальных отображений."
            )
        matrix = np.full((len(model_names), len(model_names)), np.nan, dtype=np.float32)
        metric_artifacts: Dict[str, Any] = {}
        metric_store_keys = (
            _adaptive_store_keys_from_name(metric_name)
            if selector == "adaptive"
            else [f"k{fixed_k}" if selector == "fixed_k" else store_key]
        )
        for i, mi in enumerate(tqdm(model_names, desc=metric_name, unit="model")):
            for j, mj in enumerate(model_names):
                if i == j:
                    matrix[i, j] = np.float32(0.0)
                    continue
                if args.pair_agg == "directed":
                    pair = (
                        _load_pair(maps_dir, mi, mj)
                        if maps_dir is not None
                        else _load_pair_for_store_keys(
                            maps_dirs_by_key, mi, mj, metric_store_keys
                        )
                    )
                    matrix[i, j] = np.float32(
                        _direction_value_fixed(
                            pair,
                            store_key=(
                                f"k{fixed_k}" if selector == "fixed_k" else store_key
                            ),
                            aggregation=aggregation,
                            hard_rank_threshold=float(args.hard_rank_threshold),
                            weak_spectrum_count=int(args.weak_spectrum_count),
                        )
                        if selector in {"fixed_k", "fixed_store_key"}
                        else _direction_value_adaptive(
                            pair,
                            aggregation=aggregation,
                            hard_rank_threshold=float(args.hard_rank_threshold),
                            weak_spectrum_count=int(args.weak_spectrum_count),
                        )
                    )
                    _add_direction_artifacts(
                        metric_artifacts,
                        prefix=f"{mi}_to_{mj}",
                        direction_artifacts=_artifact_from_pair(
                            pair,
                            selector=selector,
                            k=fixed_k,
                            store_key=store_key,
                            aggregation=aggregation,
                            hard_rank_threshold=float(args.hard_rank_threshold),
                            weak_spectrum_count=int(args.weak_spectrum_count),
                        ),
                    )
                    continue
                if j < i:
                    continue
                if maps_dir is not None:
                    pair_ij = _load_pair(maps_dir, mi, mj)
                    pair_ji = _load_pair(maps_dir, mj, mi)
                else:
                    pair_ij = _load_pair_for_store_keys(
                        maps_dirs_by_key, mi, mj, metric_store_keys
                    )
                    pair_ji = _load_pair_for_store_keys(
                        maps_dirs_by_key, mj, mi, metric_store_keys
                    )
                if selector in {"fixed_k", "fixed_store_key"}:
                    effective_store_key = (
                        f"k{fixed_k}" if selector == "fixed_k" else store_key
                    )
                    mij = _direction_value_fixed(
                        pair_ij,
                        store_key=effective_store_key,
                        aggregation=aggregation,
                        hard_rank_threshold=float(args.hard_rank_threshold),
                        weak_spectrum_count=int(args.weak_spectrum_count),
                    )
                    mji = _direction_value_fixed(
                        pair_ji,
                        store_key=effective_store_key,
                        aggregation=aggregation,
                        hard_rank_threshold=float(args.hard_rank_threshold),
                        weak_spectrum_count=int(args.weak_spectrum_count),
                    )
                else:
                    mij = _direction_value_adaptive(
                        pair_ij,
                        aggregation=aggregation,
                        hard_rank_threshold=float(args.hard_rank_threshold),
                        weak_spectrum_count=int(args.weak_spectrum_count),
                    )
                    mji = _direction_value_adaptive(
                        pair_ji,
                        aggregation=aggregation,
                        hard_rank_threshold=float(args.hard_rank_threshold),
                        weak_spectrum_count=int(args.weak_spectrum_count),
                    )
                _add_direction_artifacts(
                    metric_artifacts,
                    prefix=f"{mi}_to_{mj}",
                    direction_artifacts=_artifact_from_pair(
                        pair_ij,
                        selector=selector,
                        k=fixed_k,
                        store_key=store_key,
                        aggregation=aggregation,
                        hard_rank_threshold=float(args.hard_rank_threshold),
                        weak_spectrum_count=int(args.weak_spectrum_count),
                    ),
                )
                _add_direction_artifacts(
                    metric_artifacts,
                    prefix=f"{mj}_to_{mi}",
                    direction_artifacts=_artifact_from_pair(
                        pair_ji,
                        selector=selector,
                        k=fixed_k,
                        store_key=store_key,
                        aggregation=aggregation,
                        hard_rank_threshold=float(args.hard_rank_threshold),
                        weak_spectrum_count=int(args.weak_spectrum_count),
                    ),
                )
                if args.pair_agg == "antisym":
                    val = np.float32(mij - mji)
                    matrix[i, j] = val
                    matrix[j, i] = np.float32(-val)
                else:
                    val = np.float32(0.5 * (mij + mji))
                    matrix[i, j] = val
                    matrix[j, i] = val

        np.fill_diagonal(matrix, 0.0)
        meta = {
            "metric_name": metric_name,
            "is_paired": True,
            "is_symmetric": bool(args.pair_agg in {"antisym", "sym"}),
            "pair_agg": str(args.pair_agg),
            "selector": selector,
            "fixed_k": fixed_k,
            "store_key": store_key,
            "aggregation": aggregation,
            "rank_aggregation": aggregation,
            "source_maps_dir": str(maps_dir.resolve()) if maps_dir is not None else "",
            "source_maps_dirs_by_key": {
                key: str(path.resolve()) for key, path in maps_dirs_by_key.items()
            },
            "source_manifest": manifest,
            "constructor_schema_version": 1,
        }
        _save_matrix(out_path, matrix, model_names, meta)
        _save_metric_artifacts(
            out_dir / "artifacts" / f"{metric_name}_artifacts.npz",
            metric_artifacts,
            meta,
        )
        print(f"Сохранено: {out_path}")

    if not args.skip_eval and not args.downstream_json:
        args.downstream_json = _infer_downstream_json(dataset_key)

    if args.downstream_json and not args.skip_eval:
        reports_dir = Path(args.reports_dir or (out_dir.parent / "reports"))
        reports_dir.mkdir(parents=True, exist_ok=True)
        dataset_base = _dataset_base_from_key(dataset_key)
        eval_suffix = "signed" if args.eval_protocol == "delta_signed" else "abs"
        eval_csv = reports_dir / f"{dataset_base}_eval_{eval_suffix}.csv"
        eval_md = reports_dir / f"{dataset_base}_eval_{eval_suffix}.md"
        eval_cmd = [
            sys.executable,
            "-m",
            "scripts.run_evaluate_metrics",
            "--metrics_dir",
            str(out_dir),
            "--downstream_json",
            str(args.downstream_json),
            "--out_csv",
            str(eval_csv),
            "--out_md",
            str(eval_md),
            "--eval_protocol",
            str(args.eval_protocol),
            "--plots_dir",
            str(args.plots_dir or ""),
            "--plots_ext",
            str(args.plots_ext),
            "--plots_mode",
            str(args.plots_mode),
        ]
        if args.tasks:
            eval_cmd.extend(["--tasks", str(args.tasks)])
        print("\nЗапуск оценки метрик:")
        print(" ".join(eval_cmd))
        subprocess.run(eval_cmd, check=True)

        if args.summary_plots_dir:
            summary_cmd = [
                sys.executable,
                "-m",
                "scripts.plot_metric_summary",
                "--eval_csv",
                str(eval_csv),
                "--out_dir",
                str(args.summary_plots_dir),
                "--dataset",
                dataset_base,
                "--protocol",
                str(args.eval_protocol),
                "--out_name",
                f"metrics_summary_constructor_{args.pair_agg}.png",
            ]
            print("\nЗапуск сводного графика:")
            print(" ".join(summary_cmd))
            subprocess.run(summary_cmd, check=True)

    print("Готово.")


if __name__ == "__main__":
    main()
