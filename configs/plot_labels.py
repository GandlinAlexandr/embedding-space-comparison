from __future__ import annotations

import os
import re

from configs.metric_configs import short_metric_name


def display_dataset_name(name: str) -> str:
    """Canonical dataset spelling for figures and captions."""
    raw = str(name).strip()
    key = raw.lower().replace("-", "_")
    key = re.sub(r"_(train|test|validation|val)$", "", key)

    mapping = {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "food101": "Food101",
        "stl10": "STL10",
        "sun397": "SUN397",
        "ag_news": "AG News",
        "agnews": "AG News",
        "banking77": "Banking77",
        "emotion": "Emotion",
    }
    return mapping.get(key, raw)


def display_metric_name(name: str) -> str:
    """Human-readable metric labels used consistently in plots."""
    raw = os.path.basename(str(name))
    if raw.lower().endswith(".npz"):
        raw = raw[:-4]

    metric = short_metric_name(raw)

    adaptive_match = re.fullmatch(
        r"adaptive_k(?:\d+_)+(alpha_req|nesum|pseudo_condition_number|rankme|self_cluster|stable_rank)_(?:anti)?sym",
        metric,
    )
    if adaptive_match:
        aggregator = display_metric_name(adaptive_match.group(1))
        return f"adaptive kNN [{aggregator}]"

    if re.fullmatch(r"adaptive_k(?:\d+_)+(?:anti)?sym", metric):
        return "adaptive kNN [RankMe]"

    adaptive_weak_match = re.fullmatch(
        r"adaptive_weak_k(?:\d+_)+q(\d+)_(?:anti)?sym",
        metric,
    )
    if adaptive_weak_match:
        return rf"adaptive kNN [weak RankMe, $q={adaptive_weak_match.group(1)}$]"

    adaptive_tail_match = re.fullmatch(
        r"adaptive_tail_k(?:\d+_)+q(\d+)_(?:anti)?sym",
        metric,
    )
    if adaptive_tail_match:
        return rf"adaptive kNN [tail spectrum, $q={adaptive_tail_match.group(1)}$]"

    mapping = {
        "directed_k10": r"Directed, $k=10$",
        "lin_k5_antisym": r"kNN, $k=5$",
        "lin_k10_antisym": r"kNN, $k=10$",
        "lin_k20_antisym": r"kNN, $k=20$",
        "lin_k40_antisym": r"kNN, $k=40$",
        "lin_k60_antisym": r"kNN, $k=60$",
        "lin_k80_antisym": r"kNN, $k=80$",
        "lin_k100_antisym": r"kNN, $k=100$",
        "lin_k5_sym": r"kNN, $k=5$",
        "lin_k10_sym": r"kNN, $k=10$",
        "lin_k20_sym": r"kNN, $k=20$",
        "lin_k40_sym": r"kNN, $k=40$",
        "lin_k60_sym": r"kNN, $k=60$",
        "lin_k80_sym": r"kNN, $k=80$",
        "lin_k100_sym": r"kNN, $k=100$",
        "lin_eps_5_antisym": r"$\varepsilon$, 5",
        "lin_eps_10_antisym": r"$\varepsilon$, 10",
        "lin_eps_20_antisym": r"$\varepsilon$, 20",
        "lin_eps_5_sym": r"$\varepsilon$, 5",
        "lin_eps_10_sym": r"$\varepsilon$, 10",
        "lin_eps_20_sym": r"$\varepsilon$, 20",
        "multiscale_mean_antisym": "Multiscale kNN",
        "multiscale_mean_sym": "Multiscale kNN",
        "rff_k10_antisym": r"RFF, $k=10$",
        "rff_k10_sym": r"RFF, $k=10$",
        "alpha_req": r"$\alpha$-ReQ",
        "coherence": "Coherence",
        "nesum": "NESum",
        "pseudo_condition_number": "PCN",
        "rankme": "RankMe",
        "self_cluster": "Self-cluster",
        "stable_rank": "Stable rank",
    }

    if metric in mapping:
        return mapping[metric]

    id_match = re.fullmatch(r"id_diff_k(\d+)_mle_(?:anti)?sym", metric)
    if id_match:
        return rf"LID, $k={id_match.group(1)}$"

    return metric.replace("_antisym", "").replace("_sym", "")
