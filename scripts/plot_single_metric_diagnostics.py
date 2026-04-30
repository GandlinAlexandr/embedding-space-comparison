from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def load_single_eval_table(path: Path, protocol: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"single_eval_csv не найден: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"single_eval_csv пуст: {path}")

    required = ["metric_name", "protocol", "task", "spearman", "pearson", "kendall"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"В single_eval_csv отсутствуют столбцы: {missing}")

    work = df[df["protocol"].astype(str) == protocol].copy()
    if work.empty:
        raise ValueError(f"В single_eval_csv нет строк protocol={protocol!r}")

    mean_rows = work[work["task"].astype(str) == "__mean__"].copy()
    if mean_rows.empty:
        agg_cols = ["spearman", "pearson", "kendall"]
        optional_cols = [
            "correct_ratio_mean",
            "correct_ratio_flip_mean",
            "correct_ratio_adjusted_mean",
        ]
        for col in optional_cols:
            if col in work.columns:
                agg_cols.append(col)
        mean_rows = work.groupby("metric_name", as_index=False)[agg_cols].mean()

    for col in [
        "spearman",
        "pearson",
        "kendall",
        "correct_ratio_mean",
        "correct_ratio_flip_mean",
        "correct_ratio_adjusted_mean",
    ]:
        if col in mean_rows.columns:
            mean_rows[col] = pd.to_numeric(mean_rows[col], errors="coerce")

    mean_rows["metric"] = mean_rows["metric_name"].astype(str).map(short_single_name)
    return mean_rows.reset_index(drop=True)


def choose_cr_column(df: pd.DataFrame) -> str | None:
    for col in ["correct_ratio_adjusted_mean", "correct_ratio_mean"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            if not np.all(np.isnan(values)):
                return col
    return None


def build_title(dataset: str, protocol: str, title: str) -> str:
    parts = [dataset.upper(), f"single-diff {protocol}"]
    if title.strip():
        parts.append(title.strip())
    return " | ".join(parts)


def plot_table(df: pd.DataFrame, out_path: Path, dataset: str, protocol: str, title: str) -> None:
    cr_col = choose_cr_column(df)
    show_cr = cr_col is not None

    work = df.copy()
    if show_cr:
        work = work.sort_values([cr_col, "spearman"], ascending=False)
    else:
        work["_abs_sp"] = work["spearman"].abs()
        work["_abs_ke"] = work["kendall"].abs()
        work["_abs_pe"] = work["pearson"].abs()
        work = work.sort_values(["_abs_sp", "_abs_ke", "_abs_pe"], ascending=False)

    metrics = work["metric"].astype(str).tolist()
    y = np.arange(len(metrics))
    title_text = build_title(dataset=dataset, protocol=protocol, title=title)

    if show_cr:
        fig = plt.figure(figsize=(12, max(6, 0.55 * len(metrics) + 2)))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.4], hspace=0.25)
        fig.suptitle(title_text, y=0.98)

        ax1 = fig.add_subplot(gs[0, 0])
        cr_vals = work[cr_col].to_numpy(dtype=float)
        ax1.barh(y, cr_vals)
        ax1.set_yticks(y)
        ax1.set_yticklabels(metrics)
        ax1.invert_yaxis()
        ax1.set_xlim(0.0, 1.0)
        ax1.set_xlabel(
            "Доля правильного ранжирования (с поправкой)"
            if cr_col == "correct_ratio_adjusted_mean"
            else "Доля правильного ранжирования"
        )
        ax1.set_title("Доля правильного ранжирования")
        ax1.grid(True, axis="x", alpha=0.3)
        for yi, value in zip(y, cr_vals):
            if np.isfinite(value):
                ax1.text(min(value + 0.01, 1.0), yi, f"{value:.3f}", va="center", fontsize=9)

        ax2 = fig.add_subplot(gs[1, 0])
    else:
        fig = plt.figure(figsize=(12, max(6, 0.55 * len(metrics) + 2)))
        fig.suptitle(title_text, y=0.98)
        ax2 = fig.add_subplot(111)

    width = 0.25
    sp = work["spearman"].to_numpy(dtype=float)
    ke = work["kendall"].to_numpy(dtype=float)
    pe = work["pearson"].to_numpy(dtype=float)

    ax2.barh(y - width, sp, height=width, label="Spearman")
    ax2.barh(y, ke, height=width, label="Kendall")
    ax2.barh(y + width, pe, height=width, label="Pearson")

    ax2.set_yticks(y)
    ax2.set_yticklabels(metrics)
    ax2.invert_yaxis()
    ax2.set_xlim(-1.0, 1.0)
    ax2.set_xlabel("Корреляция")
    ax2.set_title("Корреляции")
    ax2.grid(True, axis="x", alpha=0.3)
    ax2.legend(loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.9, hspace=0.35)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Итоговый график диагностики single-diff метрик: CR + корреляции."
    )
    parser.add_argument(
        "--single_eval_csv",
        type=Path,
        required=True,
        help="CSV из run_evaluate_single_metrics.py",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--protocol",
        type=str,
        default="signed",
        choices=["signed", "abs"],
        help="Protocol из single_eval_csv.",
    )
    parser.add_argument("--title", type=str, default="")
    parser.add_argument(
        "--out_name",
        type=str,
        default="single_metric_diagnostics.png",
        help="Имя итогового PNG-файла.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    df = load_single_eval_table(args.single_eval_csv, protocol=args.protocol)
    out_path = args.out_dir / args.out_name
    plot_table(
        df=df,
        out_path=out_path,
        dataset=args.dataset,
        protocol=args.protocol,
        title=args.title,
    )
    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()
