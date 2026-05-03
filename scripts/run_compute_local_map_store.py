"""
Build a reusable local spectrum store for local neighborhoods.

This script intentionally lives next to the legacy metric runner instead of
replacing it.  It solves the same local linear problems as
run_compute_embedding_metrics.py, but stores candidate spectra for every
requested neighborhood so fixed and adaptive metrics can reuse one shared
source without saving full local matrices by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import scripts.run_compute_embedding_metrics as legacy


NON_MODEL_STEMS = {
    "subset_indices",
    "labels",
    "targets",
    "embeddings_manifest",
    "subset_manifest",
}


def _parse_csv_ints(raw: str) -> Tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())
    if any(k < 1 for k in vals):
        raise ValueError("Все значения должны быть >= 1")
    return tuple(sorted(dict.fromkeys(vals)))


def _parse_k_list(raw: str) -> Tuple[int, ...]:
    vals = _parse_csv_ints(raw)
    if any(k < 2 for k in vals):
        raise ValueError("Все k должны быть >= 2")
    return vals


def _parse_csv_names(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _json_hash(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _dataset_key_from_embeddings_dir(path: str) -> str:
    return Path(path).name


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


def _effective_dataset_key(args: argparse.Namespace) -> str:
    key = str(args.dataset_key).strip()
    if int(args.sample_size) > 0:
        if not key:
            raise ValueError("Для --sample_size нужно указать базовый --dataset_key.")
        return _sampled_dataset_key(
            key,
            int(args.sample_size),
            int(args.sample_seed),
            str(args.sample_strategy),
        )
    return key


def _embeddings_dir_from_dataset_key(dataset_key: str, sampled: bool) -> str:
    root = Path("data") / "embeddings"
    if sampled:
        root = root / "samples"
    return str(root / str(dataset_key))


def _load_embeddings(
    embeddings_dir: str,
    models_raw: str,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    model_names, model_to_path = legacy._list_models(embeddings_dir)
    model_names = [m for m in model_names if m not in NON_MODEL_STEMS]
    model_to_path = {m: p for m, p in model_to_path.items() if m not in NON_MODEL_STEMS}
    requested = _parse_csv_names(models_raw)
    if requested:
        missing = [m for m in requested if m not in model_to_path]
        if missing:
            raise ValueError(
                f"--models: модели не найдены в embeddings_dir: {missing}. "
                f"Доступные: {sorted(model_to_path.keys())}"
            )
        model_names = [m for m in model_names if m in set(requested)]

    embeddings: Dict[str, np.ndarray] = {}
    n0: Optional[int] = None
    for model_name in model_names:
        arr = legacy._load_embeddings(model_to_path[model_name])
        embeddings[model_name] = arr
        if n0 is None:
            n0 = int(arr.shape[0])
        elif int(arr.shape[0]) != n0:
            raise RuntimeError(
                "У всех эмбеддингов должно совпадать N. "
                f"У {model_name}: {arr.shape[0]} против {n0}"
            )
    return model_names, embeddings


def _object_array(items: List[np.ndarray]) -> np.ndarray:
    arr = np.empty(len(items), dtype=object)
    for idx, item in enumerate(items):
        arr[idx] = item
    return arr


def _neighborhood_specs(args: argparse.Namespace, k_list: Tuple[int, ...]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = [
        {"key": f"k{k}", "kind": "knn", "k": int(k), "solver": str(args.solver)}
        for k in k_list
    ]
    for p in _parse_csv_ints(args.eps_percentiles):
        specs.append(
            {
                "key": f"eps_p{p}",
                "kind": "epsilon",
                "eps_percentile": int(p),
                "weighting": "uniform",
                "solver": str(args.solver),
            }
        )
    for p in _parse_csv_ints(args.weighted_eps_percentiles):
        specs.append(
            {
                "key": f"w_eps_p{p}",
                "kind": "weighted_epsilon",
                "sigma_percentile": int(p),
                "eps_scale": float(args.weighted_eps_scale),
                "weighting": "gaussian",
                "solver": str(args.solver),
            }
        )
    for p in _parse_csv_ints(args.ransac_weighted_eps_percentiles):
        specs.append(
            {
                "key": f"w_eps_p{p}_ransac",
                "kind": "weighted_epsilon",
                "sigma_percentile": int(p),
                "eps_scale": float(args.weighted_eps_scale),
                "weighting": "gaussian",
                "solver": "ransac",
                "ransac_n_iter": int(args.ransac_n_iter),
                "ransac_sample_frac": float(args.ransac_sample_frac),
                "ransac_min_inliers": int(args.ransac_min_inliers),
                "ransac_threshold_scale": float(args.ransac_threshold_scale),
            }
        )
    return specs


def _build_store_spec(args: argparse.Namespace, k_list: Tuple[int, ...]) -> Dict[str, Any]:
    neighborhood_specs = _neighborhood_specs(args, k_list)
    return {
        "schema_version": 1,
        "store_kind": "candidate_local_spectra",
        "dataset_key": args.dataset_key,
        "base_dataset_key": str(args.base_dataset_key),
        "sampled": bool(args.sample_size > 0),
        "sample_size": int(args.sample_size),
        "sample_seed": int(args.sample_seed),
        "sample_strategy": str(args.sample_strategy),
        "embeddings_dir": str(Path(args.embeddings_dir).resolve()),
        "models": _parse_csv_names(args.models),
        "seed": int(args.seed),
        "n_centers": int(args.n_centers),
        "k_list": list(k_list),
        "eps_percentiles": list(_parse_csv_ints(args.eps_percentiles)),
        "weighted_eps_percentiles": list(_parse_csv_ints(args.weighted_eps_percentiles)),
        "ransac_weighted_eps_percentiles": list(_parse_csv_ints(args.ransac_weighted_eps_percentiles)),
        "neighborhood_specs": neighborhood_specs,
        "local_geometry_mode": str(args.local_geometry_mode),
        "adaptive_selection": "center_prediction_error",
        "adaptive_selection_centering": str(args.adaptive_selection_centering),
        "solver": str(args.solver),
        "ransac_n_iter": int(args.ransac_n_iter),
        "ransac_sample_frac": float(args.ransac_sample_frac),
        "ransac_min_inliers": int(args.ransac_min_inliers),
        "ransac_threshold_scale": float(args.ransac_threshold_scale),
        "feature_mode": "zscore",
        "map_dtype": str(args.map_dtype),
        "store_maps": bool(args.store_maps),
        "backend": str(args.backend),
    }


def _resolve_store_dir(args: argparse.Namespace, spec_payload: Dict[str, Any]) -> Path:
    if args.store_dir:
        return Path(args.store_dir)
    store_id = args.store_id.strip() or _json_hash(spec_payload)
    root = Path(args.out_root)
    if int(args.sample_size) > 0 and root == Path("data") / "local_maps":
        root = root / "samples"
    return root / str(args.dataset_key) / store_id


def _center_prediction_error(
    Xc: np.ndarray,
    Yc: np.ndarray,
    X_center: np.ndarray,
    Y_center: np.ndarray,
    adaptive_selection_centering: str,
) -> float:
    if Xc.shape[0] < 2:
        return float("inf")
    Xc_work = np.asarray(Xc, dtype=np.float64)
    Yc_work = np.asarray(Yc, dtype=np.float64)
    X_center_work = np.asarray(X_center, dtype=np.float64).reshape(1, -1)
    Y_center_work = np.asarray(Y_center, dtype=np.float64).reshape(1, -1)

    if legacy._LOCAL_GEOMETRY_MODE in {"centered_offsets_v1", "centered_offsets_v2"}:
        if adaptive_selection_centering == "neighbors_mean":
            x0 = Xc_work.mean(axis=0, keepdims=True)
            y0 = Yc_work.mean(axis=0, keepdims=True)
        elif adaptive_selection_centering == "center":
            x0 = X_center_work
            y0 = Y_center_work
        else:
            raise ValueError(
                "Неизвестный adaptive_selection_centering: "
                f"{adaptive_selection_centering}"
            )
        M = legacy._fit_local_linear_map(Xc_work - x0, Yc_work - y0)
        y_pred = (X_center_work - x0) @ M + y0
    else:
        M = legacy._fit_local_linear_map(Xc_work, Yc_work)
        y_pred = X_center_work @ M
    return float(np.linalg.norm(y_pred.reshape(-1) - Y_center_work.reshape(-1)))


def _pair_path(store_dir: Path, model_i: str, model_j: str) -> Path:
    return store_dir / "pairs" / f"{model_i}_to_{model_j}.npz"


def _save_pair_store(
    path: Path,
    *,
    model_i: str,
    model_j: str,
    center_indices: np.ndarray,
    k_list: Tuple[int, ...],
    payload_by_k: Dict[int, Dict[str, Any]],
    payload_by_spec: Dict[str, Dict[str, Any]],
    center_prediction_errors: np.ndarray,
    selected_ks: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    arrays: Dict[str, Any] = {
        "model_i": np.array(model_i, dtype=object),
        "model_j": np.array(model_j, dtype=object),
        "center_indices": np.asarray(center_indices, dtype=np.int32),
        "k_candidates": np.asarray(k_list, dtype=np.int32),
        "center_prediction_errors": np.asarray(center_prediction_errors, dtype=np.float32),
        "selected_ks": np.asarray(selected_ks, dtype=np.int32),
        "meta_json": json.dumps(meta, ensure_ascii=False),
    }
    for k in k_list:
        prefix = f"k{k}"
        row = payload_by_k[k]
        if "maps" in row:
            arrays[f"{prefix}/maps"] = row["maps"]
        arrays[f"{prefix}/singular_values"] = row["singular_values"]
        arrays[f"{prefix}/metric_rankme"] = row["metric_rankme"]
        arrays[f"{prefix}/residuals"] = row["residuals"]
        arrays[f"{prefix}/relative_residuals"] = row["relative_residuals"]
        arrays[f"{prefix}/neighbor_indices"] = row["neighbor_indices"]
        arrays[f"{prefix}/neighbor_distances"] = row["neighbor_distances"]
        arrays[f"{prefix}/inlier_masks"] = row["inlier_masks"]
    for prefix, row in payload_by_spec.items():
        if "maps" in row:
            arrays[f"{prefix}/maps"] = row["maps"]
        arrays[f"{prefix}/singular_values"] = row["singular_values"]
        arrays[f"{prefix}/metric_rankme"] = row["metric_rankme"]
        arrays[f"{prefix}/residuals"] = row["residuals"]
        arrays[f"{prefix}/relative_residuals"] = row["relative_residuals"]
        arrays[f"{prefix}/neighbor_indices"] = row["neighbor_indices"]
        arrays[f"{prefix}/neighbor_distances"] = row["neighbor_distances"]
        arrays[f"{prefix}/neighbor_sizes"] = row["neighbor_sizes"]
        arrays[f"{prefix}/sample_weights"] = row["sample_weights"]
        arrays[f"{prefix}/inlier_masks"] = row["inlier_masks"]
        arrays[f"{prefix}/sigma_values"] = row["sigma_values"]
        arrays[f"{prefix}/eps_values"] = row["eps_values"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _compute_direction_store(
    *,
    Xn: np.ndarray,
    Yn: np.ndarray,
    cache_x: legacy.NeighborCache,
    k_list: Tuple[int, ...],
    args: argparse.Namespace,
    model_i: str,
    model_j: str,
) -> Dict[str, Any]:
    rng = np.random.RandomState(int(args.seed))
    center_indices = np.asarray(cache_x.center_indices, dtype=np.int32)
    c_count = int(center_indices.size)
    errors = np.full((c_count, len(k_list)), np.inf, dtype=np.float32)

    payload_by_k: Dict[int, Dict[str, Any]] = {}
    for k in k_list:
        maps: Optional[List[np.ndarray]] = [] if args.store_maps else None
        singular_values: List[np.ndarray] = []
        metric_rankme: List[float] = []
        residuals: List[float] = []
        relative_residuals: List[float] = []
        neighbor_indices: List[np.ndarray] = []
        neighbor_distances: List[np.ndarray] = []
        inlier_masks: List[np.ndarray] = []

        lookup_k = int(k) + 1 if args.exclude_center_from_fit else int(k)
        nn = cache_x.knn[lookup_k]
        nn_dist = cache_x.knn_distances[lookup_k]

        for center_pos, (center_idx, idxs_raw, dists_raw) in enumerate(
            zip(center_indices, nn, nn_dist)
        ):
            idxs = np.asarray(idxs_raw, dtype=np.int32)
            dists = np.asarray(dists_raw, dtype=np.float32)
            if args.exclude_center_from_fit:
                idxs, dists = legacy._exclude_center_from_indices(
                    int(center_idx), idxs, dists, k=int(k)
                )
            if idxs.size < 2:
                if maps is not None:
                    maps.append(np.zeros((Xn.shape[1], Yn.shape[1]), dtype=args.map_dtype))
                singular_values.append(np.zeros((0,), dtype=np.float64))
                metric_rankme.append(float("nan"))
                residuals.append(float("nan"))
                relative_residuals.append(float("nan"))
                neighbor_indices.append(idxs)
                neighbor_distances.append(dists)
                inlier_masks.append(np.zeros((idxs.size,), dtype=bool))
                continue

            Xc = Xn[idxs]
            Yc = Yn[idxs]
            errors[center_pos, k_list.index(k)] = _center_prediction_error(
                Xc,
                Yc,
                X_center=Xn[int(center_idx)],
                Y_center=Yn[int(center_idx)],
                adaptive_selection_centering=str(args.adaptive_selection_centering),
            )
            solved = legacy._solve_local_linear_map_and_rank(
                Xc,
                Yc,
                X_center=Xn[int(center_idx)],
                Y_center=Yn[int(center_idx)],
                solver=str(args.solver),
                rng=rng,
                rank_aggregation="rankme",
                hard_rank_threshold=1e-2,
                weak_spectrum_count=5,
            )
            if maps is not None:
                maps.append(np.asarray(solved.local_map, dtype=args.map_dtype))
            singular_values.append(np.asarray(solved.singular_values, dtype=np.float64))
            metric_rankme.append(float(solved.rank_value))
            residuals.append(float(solved.raw_residual))
            relative_residuals.append(float(solved.relative_residual))
            neighbor_indices.append(idxs)
            neighbor_distances.append(dists)
            inlier_masks.append(np.asarray(solved.inlier_mask, dtype=bool))

        payload = {
            "singular_values": np.stack(singular_values, axis=0),
            "metric_rankme": np.asarray(metric_rankme, dtype=np.float64),
            "residuals": np.asarray(residuals, dtype=np.float32),
            "relative_residuals": np.asarray(relative_residuals, dtype=np.float32),
            "neighbor_indices": np.stack(neighbor_indices, axis=0).astype(np.int32),
            "neighbor_distances": np.stack(neighbor_distances, axis=0).astype(np.float32),
            "inlier_masks": np.stack(inlier_masks, axis=0).astype(bool),
        }
        if maps is not None:
            payload["maps"] = np.stack(maps, axis=0)
        payload_by_k[int(k)] = payload

    if k_list:
        best_pos = np.argmin(errors, axis=1)
        selected_ks = np.asarray(k_list, dtype=np.int32)[best_pos]
    else:
        selected_ks = np.zeros((c_count,), dtype=np.int32)
    payload_by_spec: Dict[str, Dict[str, Any]] = {}
    eps_specs = [
        spec for spec in _neighborhood_specs(args, k_list)
        if spec["kind"] in {"epsilon", "weighted_epsilon"}
    ]
    for spec in eps_specs:
        key = str(spec["key"])
        percentile = int(spec.get("eps_percentile", spec.get("sigma_percentile")))
        neigh = cache_x.eps[percentile]
        neigh_distances = cache_x.eps_distances[percentile]
        sigma_val = float(cache_x.sigma_values.get(percentile, float("nan")))
        eps_val = float(cache_x.eps_values.get(percentile, float("nan")))
        sigma_safe = max(sigma_val, 1e-8) if np.isfinite(sigma_val) else 1.0
        maps: Optional[List[np.ndarray]] = [] if args.store_maps else None
        singular_values: List[np.ndarray] = []
        metric_rankme: List[float] = []
        residuals: List[float] = []
        relative_residuals: List[float] = []
        neighbor_indices: List[np.ndarray] = []
        neighbor_distances: List[np.ndarray] = []
        neighbor_sizes: List[int] = []
        sample_weights_list: List[np.ndarray] = []
        inlier_masks: List[np.ndarray] = []
        sigma_values: List[float] = []
        eps_values: List[float] = []
        for center_idx, idxs_raw, dists_raw in zip(center_indices, neigh, neigh_distances):
            idxs = np.asarray(idxs_raw, dtype=np.int32).reshape(-1)
            dists = np.asarray(dists_raw, dtype=np.float32).reshape(-1)
            if idxs.size < 2:
                if maps is not None:
                    maps.append(np.zeros((Xn.shape[1], Yn.shape[1]), dtype=args.map_dtype))
                singular_values.append(np.zeros((0,), dtype=np.float64))
                metric_rankme.append(float("nan"))
                residuals.append(float("nan"))
                relative_residuals.append(float("nan"))
                neighbor_indices.append(idxs)
                neighbor_distances.append(dists)
                neighbor_sizes.append(int(idxs.size))
                sample_weights_list.append(np.ones((idxs.size,), dtype=np.float32))
                inlier_masks.append(np.zeros((idxs.size,), dtype=bool))
                sigma_values.append(sigma_val)
                eps_values.append(eps_val)
                continue
            weights = None
            if spec.get("weighting") == "gaussian":
                weights = np.exp(-np.square(dists.astype(np.float64)) / (sigma_safe**2)).astype(np.float32)
            solved = legacy._solve_local_linear_map_and_rank(
                Xn[idxs],
                Yn[idxs],
                X_center=Xn[int(center_idx)],
                Y_center=Yn[int(center_idx)],
                sample_weights=weights,
                solver=str(spec.get("solver", args.solver)),
                rng=rng,
                ransac_n_iter=int(spec.get("ransac_n_iter", args.ransac_n_iter)),
                ransac_sample_frac=float(spec.get("ransac_sample_frac", args.ransac_sample_frac)),
                ransac_min_inliers=int(spec.get("ransac_min_inliers", args.ransac_min_inliers)),
                ransac_threshold_scale=float(spec.get("ransac_threshold_scale", args.ransac_threshold_scale)),
                rank_aggregation="rankme",
                hard_rank_threshold=1e-2,
                weak_spectrum_count=5,
            )
            if maps is not None:
                maps.append(np.asarray(solved.local_map, dtype=args.map_dtype))
            singular_values.append(np.asarray(solved.singular_values, dtype=np.float64))
            metric_rankme.append(float(solved.rank_value))
            residuals.append(float(solved.raw_residual))
            relative_residuals.append(float(solved.relative_residual))
            neighbor_indices.append(idxs)
            neighbor_distances.append(dists)
            neighbor_sizes.append(int(idxs.size))
            sample_weights_list.append(
                np.ones((idxs.size,), dtype=np.float32)
                if weights is None
                else np.asarray(weights, dtype=np.float32)
            )
            inlier_masks.append(np.asarray(solved.inlier_mask, dtype=bool))
            sigma_values.append(sigma_val)
            eps_values.append(eps_val)
        payload = {
            "singular_values": np.stack(singular_values, axis=0),
            "metric_rankme": np.asarray(metric_rankme, dtype=np.float64),
            "residuals": np.asarray(residuals, dtype=np.float32),
            "relative_residuals": np.asarray(relative_residuals, dtype=np.float32),
            "neighbor_indices": _object_array(neighbor_indices),
            "neighbor_distances": _object_array(neighbor_distances),
            "neighbor_sizes": np.asarray(neighbor_sizes, dtype=np.int32),
            "sample_weights": _object_array(sample_weights_list),
            "inlier_masks": _object_array(inlier_masks),
            "sigma_values": np.asarray(sigma_values, dtype=np.float32),
            "eps_values": np.asarray(eps_values, dtype=np.float32),
        }
        if maps is not None:
            payload["maps"] = np.stack(maps, axis=0)
        payload_by_spec[key] = payload
    return {
        "center_indices": center_indices,
        "payload_by_k": payload_by_k,
        "payload_by_spec": payload_by_spec,
        "center_prediction_errors": errors,
        "selected_ks": selected_ks,
        "meta": {
            "model_i": model_i,
            "model_j": model_j,
            "k_list": list(k_list),
            "local_geometry_mode": legacy._LOCAL_GEOMETRY_MODE,
            "exclude_center_from_fit": bool(args.exclude_center_from_fit),
            "adaptive_selection": "center_prediction_error",
            "adaptive_selection_centering": str(args.adaptive_selection_centering),
            "solver": str(args.solver),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reusable local linear map store for kNN candidate neighborhoods."
    )
    parser.add_argument("--embeddings_dir", default="")
    parser.add_argument("--out_root", default=str(Path("data") / "local_maps"))
    parser.add_argument("--store_dir", default="")
    parser.add_argument("--store_id", default="")
    parser.add_argument("--dataset_key", default="")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="Если >0, используется data/embeddings/samples/<dataset_key>_sN_seedS_<sample_strategy>.",
    )
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--sample_strategy", choices=["stratified", "random"], default="stratified")
    parser.add_argument("--models", default="")
    parser.add_argument("--k_list", default="")
    parser.add_argument("--eps_percentiles", default="")
    parser.add_argument("--weighted_eps_percentiles", default="")
    parser.add_argument("--ransac_weighted_eps_percentiles", default="")
    parser.add_argument("--weighted_eps_scale", type=float, default=3.0)
    parser.add_argument("--n_centers", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--local_geometry_mode",
        choices=list(legacy._LOCAL_GEOMETRY_MODE_CHOICES),
        default="centered_offsets_v2",
    )
    parser.add_argument("--adaptive_selection_centering", default="neighbors_mean")
    parser.add_argument("--solver", choices=["lstsq", "ransac"], default="lstsq")
    parser.add_argument("--ransac_n_iter", type=int, default=48)
    parser.add_argument("--ransac_sample_frac", type=float, default=0.5)
    parser.add_argument("--ransac_min_inliers", type=int, default=4)
    parser.add_argument("--ransac_threshold_scale", type=float, default=2.5)
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--map_dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument(
        "--store_maps",
        action="store_true",
        help="Диагностический режим: дополнительно сохранять полные локальные M. По умолчанию сохраняются только спектры.",
    )
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument(
        "--include_self",
        action="store_true",
        help="Also store model_i -> model_i directions. Pairwise metrics do not need this.",
    )
    args = parser.parse_args()

    k_list = _parse_k_list(args.k_list)
    if not _neighborhood_specs(args, k_list):
        raise ValueError(
            "Нужно указать хотя бы один тип окрестности: --k_list, "
            "--eps_percentiles, --weighted_eps_percentiles или "
            "--ransac_weighted_eps_percentiles."
        )
    args.base_dataset_key = str(args.dataset_key).strip()
    effective_key = _effective_dataset_key(args) if args.dataset_key else ""
    if not args.embeddings_dir:
        if not effective_key:
            raise ValueError("Нужно указать либо --embeddings_dir, либо --dataset_key.")
        args.embeddings_dir = _embeddings_dir_from_dataset_key(
            effective_key,
            sampled=bool(int(args.sample_size) > 0),
        )
    if not args.dataset_key:
        args.dataset_key = _dataset_key_from_embeddings_dir(args.embeddings_dir)
        args.base_dataset_key = str(args.dataset_key)
    elif effective_key:
        args.dataset_key = effective_key

    legacy._LOCAL_GEOMETRY_MODE = str(args.local_geometry_mode)
    legacy._COMPUTE_BACKEND = legacy._resolve_compute_backend(str(args.backend))

    # The old centered_offsets_v2 protocol excludes the center for kNN rows.
    args.exclude_center_from_fit = bool(args.local_geometry_mode == "centered_offsets_v2")

    model_names, embeddings = _load_embeddings(args.embeddings_dir, args.models)
    legacy._PRECOMPUTED_ZSCORES = {
        m: np.asarray(legacy._zscore_rows(x), dtype=np.float32)
        for m, x in embeddings.items()
    }

    store_spec = _build_store_spec(args, k_list)
    store_dir = _resolve_store_dir(args, store_spec)
    (store_dir / "pairs").mkdir(parents=True, exist_ok=True)

    manifest = {
        **store_spec,
        "store_id": store_dir.name,
        "model_names": model_names,
        "pair_file_format": "npz-per-directed-pair",
    }
    (store_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (store_dir / "model_names.json").write_text(
        json.dumps(model_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    percentile_values = list(_parse_csv_ints(args.eps_percentiles))
    percentile_values += list(_parse_csv_ints(args.weighted_eps_percentiles))
    percentile_values += list(_parse_csv_ints(args.ransac_weighted_eps_percentiles))
    percentile_values = sorted(set(percentile_values))

    cache_key = legacy.NeighborCacheKey(
        n_centers=int(args.n_centers),
        ks=tuple(int(k) + 1 if args.exclude_center_from_fit else int(k) for k in k_list),
        percentile=None,
        eps_scale=float(args.weighted_eps_scale),
    )
    cache_by_model: Dict[str, legacy.NeighborCache] = {}

    def cache_for(model_name: str) -> legacy.NeighborCache:
        if model_name not in cache_by_model:
            cache = legacy._build_neighbor_cache_from_key(
                embeddings[model_name],
                cache_key,
                seed=int(args.seed),
                X_norm=legacy._get_precomputed_zscore(model_name, embeddings[model_name]),
            )
            for percentile in percentile_values:
                eps_cache = legacy._build_neighbor_cache_from_key(
                    embeddings[model_name],
                    legacy.NeighborCacheKey(
                        n_centers=int(args.n_centers),
                        ks=(),
                        percentile=int(percentile),
                        eps_scale=float(args.weighted_eps_scale),
                    ),
                    seed=int(args.seed),
                    center_indices=cache.center_indices,
                    X_norm=legacy._get_precomputed_zscore(model_name, embeddings[model_name]),
                )
                cache.eps[int(percentile)] = eps_cache.eps[int(percentile)]
                cache.eps_distances[int(percentile)] = eps_cache.eps_distances[int(percentile)]
                cache.sigma_values[int(percentile)] = eps_cache.sigma_values[int(percentile)]
                cache.eps_values[int(percentile)] = eps_cache.eps_values[int(percentile)]
            cache_by_model[model_name] = cache
        return cache_by_model[model_name]

    print(f"Local map store: {store_dir}")
    print(f"Models: {len(model_names)} | directed pairs: {len(model_names) * (len(model_names) - 1)}")
    print(f"k candidates: {k_list} | n_centers={args.n_centers}")
    print(f"geometry={legacy._LOCAL_GEOMETRY_MODE} | backend={legacy._COMPUTE_BACKEND.name}")

    for model_i in tqdm(model_names, desc="model_i"):
        Xn = legacy._get_precomputed_zscore(model_i, embeddings[model_i])
        cache_i = cache_for(model_i)
        for model_j in model_names:
            if model_i == model_j and not args.include_self:
                continue
            out_path = _pair_path(store_dir, model_i, model_j)
            if args.incremental and out_path.exists():
                continue
            Yn = legacy._get_precomputed_zscore(model_j, embeddings[model_j])
            direction = _compute_direction_store(
                Xn=Xn,
                Yn=Yn,
                cache_x=cache_i,
                k_list=k_list,
                args=args,
                model_i=model_i,
                model_j=model_j,
            )
            _save_pair_store(
                out_path,
                model_i=model_i,
                model_j=model_j,
                center_indices=direction["center_indices"],
                k_list=k_list,
                payload_by_k=direction["payload_by_k"],
                payload_by_spec=direction["payload_by_spec"],
                center_prediction_errors=direction["center_prediction_errors"],
                selected_ks=direction["selected_ks"],
                meta=direction["meta"],
            )

    print(f"Готово: {store_dir}")


if __name__ == "__main__":
    main()
