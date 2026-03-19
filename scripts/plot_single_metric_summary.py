from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ВАЖНО: запускаем как модуль: python -m scripts.plot_single_metric_summary
from configs.metric_configs import short_metric_name as short_pairwise_name


# ============================================================
# Utils
# ============================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def require_cols(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"{name}: отсутствуют столбцы {sorted(missing)}")


def load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name}: файл не найден: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{name}: файл пуст: {path}")
    return df


# ============================================================
# Labels by protocol
# ============================================================


def target_label(protocol: str) -> str:
    if protocol == "signed":
        return "Δacc"
    if protocol == "abs":
        return "|Δacc|"
    raise ValueError(f"Неизвестный protocol: {protocol}")


def pairwise_corr_label(protocol: str) -> str:
    return f"corr(metric(e₁, e₂), {target_label(protocol)})"


def single_corr_label(protocol: str) -> str:
    if protocol == "signed":
        return "corr(u(e₁) - u(e₂), Δacc)"
    if protocol == "abs":
        return "corr(|u(e₁) - u(e₂)|, |Δacc|)"
    raise ValueError(f"Неизвестный protocol: {protocol}")


def build_title_prefix(base_title: str, protocol: str) -> str:
    # Если пользователь уже явно указал нужную цель в title,
    # не дублируем её второй раз.
    tgt = target_label(protocol)
    if tgt in base_title:
        return base_title
    return f"{base_title} | {tgt}"


# ============================================================
# Short names
# ============================================================


def short_single_name(metric_name: str) -> str:
    mapping = {
        "pseudo_condition_number": "pseudo_cond",
        "rankme": "rankme",
        "coherence": "coherence",
        "stable_rank": "stable_rank",
        "nesum": "nesum",
        "self_cluster": "self_cluster",
        "alpha_req": "alpha_req",
    }
    return mapping.get(metric_name, metric_name)


# ============================================================
# Load pairwise table
# pairwise CSV from run_evaluate_metrics.py:
# metric_file, spearman_mean, pearson_mean, kendall_mean, ...
# ============================================================


def load_pairwise_table(pairwise_csv: Path) -> pd.DataFrame:
    df = load_csv(pairwise_csv, "pairwise_csv")
    require_cols(
        df,
        ["metric_file", "spearman_mean", "pearson_mean", "kendall_mean"],
        "pairwise_csv",
    )

    out = pd.DataFrame(
        {
            "pairwise_metric_file": df["metric_file"].astype(str),
            "pairwise_metric": df["metric_file"].astype(str).map(short_pairwise_name),
            "pairwise_spearman": pd.to_numeric(df["spearman_mean"], errors="coerce"),
            "pairwise_pearson": pd.to_numeric(df["pearson_mean"], errors="coerce"),
            "pairwise_kendall": pd.to_numeric(df["kendall_mean"], errors="coerce"),
        }
    )

    out = out.dropna(
        subset=[
            "pairwise_metric",
            "pairwise_spearman",
            "pairwise_pearson",
            "pairwise_kendall",
        ]
    ).reset_index(drop=True)

    return out


# ============================================================
# Load single table
# single CSV from run_evaluate_single_metrics.py:
# metric_name, protocol, task, spearman, pearson, kendall, ...
# ============================================================


def load_single_table(single_csv: Path, protocol: str) -> pd.DataFrame:
    df = load_csv(single_csv, "single_csv")
    require_cols(
        df,
        ["metric_name", "protocol", "spearman", "pearson", "kendall"],
        "single_csv",
    )

    work = df[df["protocol"].astype(str) == protocol].copy()
    if work.empty:
        raise ValueError(f"single_csv: нет строк protocol={protocol}")

    if "task" in work.columns:
        mean_rows = work[work["task"].astype(str) == "__mean__"].copy()
        if not mean_rows.empty:
            work = mean_rows
        else:
            work = work.groupby("metric_name", as_index=False)[
                ["spearman", "pearson", "kendall"]
            ].mean()

    out = pd.DataFrame(
        {
            "baseline_metric": work["metric_name"].astype(str).map(short_single_name),
            "single_spearman": pd.to_numeric(work["spearman"], errors="coerce"),
            "single_pearson": pd.to_numeric(work["pearson"], errors="coerce"),
            "single_kendall": pd.to_numeric(work["kendall"], errors="coerce"),
        }
    )

    out = out.dropna(
        subset=[
            "baseline_metric",
            "single_spearman",
            "single_pearson",
            "single_kendall",
        ]
    ).reset_index(drop=True)

    return out


# ============================================================
# Plot helpers
# ============================================================


def plot_grouped_three(
    labels: list[str],
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    l1: str,
    l2: str,
    l3: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.15), 6))

    ax.bar(x - width, v1, width=width, label=l1)
    ax.bar(x, v2, width=width, label=l2)
    ax.bar(x + width, v3, width=width, label=l3)

    ax.set_title(title)
    ax.set_xlabel("Метрика")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.axhline(0.0, linewidth=1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    save_fig(fig, out_path)


def plot_two_series(
    labels: list[str],
    pairwise_vals: np.ndarray,
    single_vals: np.ndarray,
    pairwise_label: str,
    single_label: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.3), 6))

    ax.bar(x - width / 2, pairwise_vals, width=width, label=pairwise_label)
    ax.bar(x + width / 2, single_vals, width=width, label=single_label)

    ax.set_title(title)
    ax.set_xlabel("Baseline-метрика u")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.axhline(0.0, linewidth=1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    save_fig(fig, out_path)


# ============================================================
# Best pairwise by each correlation
# ============================================================


def select_best_pairwise(pairwise_df: pd.DataFrame, corr_col: str) -> pd.Series:
    idx = pairwise_df[corr_col].astype(float).idxmax()
    return pairwise_df.loc[idx]


def build_best_vs_single_table(
    pairwise_df: pd.DataFrame,
    single_df: pd.DataFrame,
) -> pd.DataFrame:
    best_sp = select_best_pairwise(pairwise_df, "pairwise_spearman")
    best_pe = select_best_pairwise(pairwise_df, "pairwise_pearson")
    best_ke = select_best_pairwise(pairwise_df, "pairwise_kendall")

    out = single_df.copy()

    out["best_pairwise_metric_spearman"] = str(best_sp["pairwise_metric"])
    out["best_pairwise_metric_pearson"] = str(best_pe["pairwise_metric"])
    out["best_pairwise_metric_kendall"] = str(best_ke["pairwise_metric"])

    out["pairwise_spearman"] = float(best_sp["pairwise_spearman"])
    out["pairwise_pearson"] = float(best_pe["pairwise_pearson"])
    out["pairwise_kendall"] = float(best_ke["pairwise_kendall"])

    out["delta_spearman"] = out["pairwise_spearman"] - out["single_spearman"]
    out["delta_pearson"] = out["pairwise_pearson"] - out["single_pearson"]
    out["delta_kendall"] = out["pairwise_kendall"] - out["single_kendall"]

    return out


# ============================================================
# Main plotting routines
# ============================================================


def plot_pairwise_only(
    pairwise_df: pd.DataFrame,
    out_dir: Path,
    title_prefix: str,
    protocol: str,
) -> None:
    labels = pairwise_df["pairwise_metric"].astype(str).tolist()

    plot_grouped_three(
        labels=labels,
        v1=pairwise_df["pairwise_spearman"].to_numpy(dtype=np.float64),
        v2=pairwise_df["pairwise_pearson"].to_numpy(dtype=np.float64),
        v3=pairwise_df["pairwise_kendall"].to_numpy(dtype=np.float64),
        l1="Spearman",
        l2="Pearson",
        l3="Kendall",
        title=f"{title_prefix} | pairwise only",
        ylabel=pairwise_corr_label(protocol),
        out_path=out_dir / "pairwise_only_all.png",
    )


def plot_single_only(
    single_df: pd.DataFrame,
    out_dir: Path,
    title_prefix: str,
    protocol: str,
) -> None:
    labels = single_df["baseline_metric"].astype(str).tolist()

    plot_grouped_three(
        labels=labels,
        v1=single_df["single_spearman"].to_numpy(dtype=np.float64),
        v2=single_df["single_pearson"].to_numpy(dtype=np.float64),
        v3=single_df["single_kendall"].to_numpy(dtype=np.float64),
        l1="Spearman",
        l2="Pearson",
        l3="Kendall",
        title=f"{title_prefix} | single-diff only",
        ylabel=single_corr_label(protocol),
        out_path=out_dir / "single_diff_only_all.png",
    )


def plot_best_pairwise_vs_single(
    best_vs_single_df: pd.DataFrame,
    out_dir: Path,
    title_prefix: str,
    protocol: str,
) -> None:
    labels = best_vs_single_df["baseline_metric"].astype(str).tolist()

    best_sp_name = best_vs_single_df["best_pairwise_metric_spearman"].iloc[0]
    best_pe_name = best_vs_single_df["best_pairwise_metric_pearson"].iloc[0]
    best_ke_name = best_vs_single_df["best_pairwise_metric_kendall"].iloc[0]

    pairwise_label = pairwise_corr_label(protocol)
    single_label = single_corr_label(protocol)

    plot_two_series(
        labels=labels,
        pairwise_vals=best_vs_single_df["pairwise_spearman"].to_numpy(dtype=np.float64),
        single_vals=best_vs_single_df["single_spearman"].to_numpy(dtype=np.float64),
        pairwise_label=pairwise_label,
        single_label=single_label,
        title=f"{title_prefix} | Spearman | best pairwise = {best_sp_name}",
        ylabel="корреляция",
        out_path=out_dir / "best_pairwise_vs_single_spearman.png",
    )

    plot_two_series(
        labels=labels,
        pairwise_vals=best_vs_single_df["pairwise_pearson"].to_numpy(dtype=np.float64),
        single_vals=best_vs_single_df["single_pearson"].to_numpy(dtype=np.float64),
        pairwise_label=pairwise_label,
        single_label=single_label,
        title=f"{title_prefix} | Pearson | best pairwise = {best_pe_name}",
        ylabel="корреляция",
        out_path=out_dir / "best_pairwise_vs_single_pearson.png",
    )

    plot_two_series(
        labels=labels,
        pairwise_vals=best_vs_single_df["pairwise_kendall"].to_numpy(dtype=np.float64),
        single_vals=best_vs_single_df["single_kendall"].to_numpy(dtype=np.float64),
        pairwise_label=pairwise_label,
        single_label=single_label,
        title=f"{title_prefix} | Kendall | best pairwise = {best_ke_name}",
        ylabel="корреляция",
        out_path=out_dir / "best_pairwise_vs_single_kendall.png",
    )


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Нормальные графики для pairwise-метрик и single-diff baseline."
    )

    parser.add_argument(
        "--single_eval_csv",
        type=Path,
        required=True,
        help="CSV из run_evaluate_single_metrics.py",
    )

    parser.add_argument(
        "--pairwise_eval_csv",
        type=Path,
        required=True,
        help="CSV из run_evaluate_metrics.py",
    )

    parser.add_argument(
        "--single_protocol",
        type=str,
        required=True,
        choices=["signed", "abs"],
        help="Какой protocol брать из single_eval_csv.",
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Папка для графиков.",
    )

    parser.add_argument(
        "--title",
        type=str,
        default="CIFAR10",
        help="Базовый префикс заголовков. Протокол (Δacc или |Δacc|) будет добавлен автоматически.",
    )

    parser.add_argument(
        "--save_pairwise_table_csv",
        type=Path,
        default=None,
        help="Необязательный путь для сохранения таблицы pairwise-only.",
    )

    parser.add_argument(
        "--save_single_table_csv",
        type=Path,
        default=None,
        help="Необязательный путь для сохранения таблицы single-only.",
    )

    parser.add_argument(
        "--save_best_comparison_csv",
        type=Path,
        default=None,
        help="Необязательный путь для сохранения таблицы best-pairwise-vs-single.",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    args = parse_args()

    ensure_dir(args.out_dir)

    pairwise_df = load_pairwise_table(args.pairwise_eval_csv)
    single_df = load_single_table(args.single_eval_csv, args.single_protocol)
    best_vs_single_df = build_best_vs_single_table(pairwise_df, single_df)
    title_prefix = build_title_prefix(args.title, args.single_protocol)

    if args.save_pairwise_table_csv is not None:
        args.save_pairwise_table_csv.parent.mkdir(parents=True, exist_ok=True)
        pairwise_df.to_csv(
            args.save_pairwise_table_csv, index=False, encoding="utf-8-sig"
        )

    if args.save_single_table_csv is not None:
        args.save_single_table_csv.parent.mkdir(parents=True, exist_ok=True)
        single_df.to_csv(args.save_single_table_csv, index=False, encoding="utf-8-sig")

    if args.save_best_comparison_csv is not None:
        args.save_best_comparison_csv.parent.mkdir(parents=True, exist_ok=True)
        best_vs_single_df.to_csv(
            args.save_best_comparison_csv, index=False, encoding="utf-8-sig"
        )

    print("============================================================")
    print("ПОСТРОЕНИЕ ГРАФИКОВ ДЛЯ PAIRWISE И SINGLE-DIFF")
    print("============================================================")
    print(f"Pairwise CSV       : {args.pairwise_eval_csv}")
    print(f"Single CSV         : {args.single_eval_csv}")
    print(f"Single protocol    : {args.single_protocol}")
    print(f"Target label       : {target_label(args.single_protocol)}")
    print(f"Pairwise metrics   : {len(pairwise_df)}")
    print(f"Baseline metrics   : {len(single_df)}")
    print(f"Папка вывода       : {args.out_dir}")
    print("============================================================")

    plot_pairwise_only(pairwise_df, args.out_dir, title_prefix, args.single_protocol)
    plot_single_only(single_df, args.out_dir, title_prefix, args.single_protocol)
    plot_best_pairwise_vs_single(
        best_vs_single_df, args.out_dir, title_prefix, args.single_protocol
    )

    print("============================================================")
    print("ГОТОВО")
    print("============================================================")
    print(f"Графики сохранены в: {args.out_dir}")
    print("============================================================")


if __name__ == "__main__":
    main()
