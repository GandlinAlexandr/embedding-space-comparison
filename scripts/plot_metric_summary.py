import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from configs.plot_labels import display_dataset_name, display_metric_name


VALID_PLOT_EXTS = ("png", "pdf", "svg")


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _parse_plots_exts(raw: str) -> list[str]:
    exts = [
        part.strip().lower().lstrip(".") for part in str(raw).split(",") if part.strip()
    ]
    if not exts:
        raise ValueError("--plots_ext должен содержать хотя бы одно расширение.")
    bad = [ext for ext in exts if ext not in VALID_PLOT_EXTS]
    if bad:
        raise ValueError(
            f"Неподдерживаемые расширения: {bad}. Допустимые: {list(VALID_PLOT_EXTS)}"
        )
    out: list[str] = []
    for ext in exts:
        if ext not in out:
            out.append(ext)
    return out


def _save_fig(fig: plt.Figure, out_path: Path, plots_exts: list[str]) -> list[Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for ext in plots_exts:
        path = out_path.with_suffix(f".{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    return saved


def _choose_cr_column(df: pd.DataFrame) -> str | None:
    """
    Возвращает имя столбца «Correct Ratio», если он присутствует, иначе — None.
    Примечание: для протокола delta_abs значение CR не определено, поэтому значения будут NaN.
    """
    for candidate in ["correct_ratio_adjusted_mean", "correct_ratio_mean"]:
        if candidate in df.columns:
            return candidate
    return None


def _is_all_nan(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce")
    return bool(np.all(np.isnan(s.to_numpy())))


def _display_protocol(raw: str) -> str:
    value = str(raw).strip()
    labels = {
        "delta_signed": "Δacc",
        "signed": "Δacc",
        "delta_abs": "|Δacc|",
        "abs": "|Δacc|",
    }
    return labels.get(value, value)


def _clean_metric_file_name(name: str) -> str:
    raw = str(name).replace("\\", "/").split("/")[-1]
    if raw.lower().endswith(".npz"):
        raw = raw[:-4]
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval_csv",
        required=True,
        help="CSV с оценкой, полученный из run_evaluate_metrics.py",
    )
    ap.add_argument("--out_dir", required=True, help="Куда сохранить график")
    ap.add_argument(
        "--dataset",
        required=True,
        help="Имя датасета в заголовке (например, cifar10, imagenet100)",
    )
    ap.add_argument(
        "--protocol", default="Δacc protocol", help="Подпись протокола в заголовке"
    )
    ap.add_argument(
        "--title", default="", help="Необязательный дополнительный суффикс заголовка"
    )
    ap.add_argument(
        "--out_name", default="metrics_summary.png", help="Базовое имя выходного файла"
    )
    ap.add_argument(
        "--plots_ext",
        default="png,svg",
        help="Расширения для сохранения через запятую: png,svg,pdf.",
    )
    ap.add_argument(
        "--exclude_metrics",
        default="",
        help=(
            "Метрики для исключения через запятую. Можно указывать исходные имена, "
            "имена файлов или отображаемые подписи."
        ),
    )
    args = ap.parse_args()

    plots_exts = _parse_plots_exts(args.plots_ext)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)
    df = pd.read_csv(args.eval_csv)

    required = ["metric_file", "spearman_mean", "pearson_mean", "kendall_mean"]
    for c in required:
        if c not in df.columns:
            raise KeyError(
                f"Столбец '{c}' не найден в CSV. Доступные столбцы: {list(df.columns)}"
            )

    df = df.copy()
    df["metric"] = df["metric_file"].apply(display_metric_name)
    if args.exclude_metrics.strip():
        exclude = {
            part.strip() for part in args.exclude_metrics.split(",") if part.strip()
        }
        metric_file_clean = df["metric_file"].astype(str).map(_clean_metric_file_name)
        df = df[
            ~df["metric"].isin(exclude)
            & ~df["metric_file"].astype(str).isin(exclude)
            & ~metric_file_clean.isin(exclude)
        ].copy()

    # - В протоколе delta_signed CR существует и имеет смысл.
    # - В протоколе delta_abs CR не определён -> это NaN.
    cr_col = _choose_cr_column(df)
    show_cr_panel = False
    if cr_col is not None and (not _is_all_nan(df[cr_col])):
        show_cr_panel = True

    # Сортировка:
    # - Для antisym/delta_signed: сохранить старое поведение ТОЧНО (CR по убыванию, затем Spearman по убыванию)
    # - Для sym/delta_abs: сортировка по abs(Spearman) по убыванию, затем abs(Kendall), затем abs(Pearson)
    if show_cr_panel:
        df = df.sort_values([cr_col, "spearman_mean"], ascending=False)
    else:
        df["_abs_sp"] = df["spearman_mean"].abs()
        df["_abs_ke"] = df["kendall_mean"].abs()
        df["_abs_pe"] = df["pearson_mean"].abs()
        df = df.sort_values(["_abs_sp", "_abs_ke", "_abs_pe"], ascending=False)

    metrics = df["metric"].astype(str).tolist()
    y = np.arange(len(metrics))

    # Заголовок
    title_parts = [display_dataset_name(args.dataset), _display_protocol(args.protocol)]
    if args.title.strip():
        title_parts.append(args.title.strip())
    title_text = " | ".join(title_parts)

    # Рисунок
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )

    if show_cr_panel:
        fig = plt.figure(figsize=(7.5, max(6.5, 0.58 * len(metrics) + 2.2)))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.2], hspace=0.38)
        fig.suptitle(title_text, y=0.98)

        # Панель 1: CR
        ax1 = fig.add_subplot(gs[0, 0])
        cr_vals = pd.to_numeric(df[cr_col], errors="coerce").to_numpy(dtype=float)
        ax1.barh(y, cr_vals)
        ax1.set_yticks(y)
        ax1.set_yticklabels(metrics)
        ax1.invert_yaxis()
        ax1.set_ylabel("Метрики")
        ax1.set_xlim(0.0, 1.0)
        ax1.set_xlabel("Доля правильного ранжирования")
        ax1.set_title("Доля правильного ранжирования")
        ax1.grid(True, axis="x", alpha=0.3)

        # Подписи значений на барах
        for yi, v in zip(y, cr_vals):
            if np.isnan(v):
                continue
            ax1.text(min(v + 0.01, 0.97), yi, f"{v:.3f}", va="center", fontsize=9)

        # Панель 2: корреляции
        ax2 = fig.add_subplot(gs[1, 0])
        width = 0.25
        sp = df["spearman_mean"].to_numpy(dtype=float)
        ke = df["kendall_mean"].to_numpy(dtype=float)
        pe = df["pearson_mean"].to_numpy(dtype=float)

        ax2.barh(y - width, sp, height=width, label="Spearman")
        ax2.barh(y, ke, height=width, label="Kendall")
        ax2.barh(y + width, pe, height=width, label="Pearson")

        ax2.set_yticks(y)
        ax2.set_yticklabels(metrics)
        ax2.invert_yaxis()
        ax2.set_xlim(-1.0, 1.0)
        ax2.set_ylabel("Метрики")
        ax2.set_xlabel("Корреляция")
        ax2.set_title("Корреляции")
        ax2.grid(True, axis="x", alpha=0.3)
        ax2.legend(loc="lower left")

        out_path = out_dir / args.out_name
        fig.subplots_adjust(left=0.24, right=0.98, top=0.90, bottom=0.08, hspace=0.38)
        for path in _save_fig(fig, out_path, plots_exts):
            print(f"Сохранено: {path}")
        return

    # ---- Sym / delta_abs: только корреляции (CR не определено)
    fig = plt.figure(figsize=(7.5, max(5.5, 0.42 * len(metrics) + 2.0)))
    ax = fig.add_subplot(111)
    fig.suptitle(title_text, y=0.98)

    width = 0.25
    sp = df["spearman_mean"].to_numpy(dtype=float)
    ke = df["kendall_mean"].to_numpy(dtype=float)
    pe = df["pearson_mean"].to_numpy(dtype=float)

    ax.barh(y - width, sp, height=width, label="Spearman")
    ax.barh(y, ke, height=width, label="Kendall")
    ax.barh(y + width, pe, height=width, label="Pearson")

    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.invert_yaxis()
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylabel("Метрики")
    ax.set_xlabel("Корреляция")
    ax.set_title("Корреляции")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower left")

    out_path = out_dir / args.out_name
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for path in _save_fig(fig, out_path, plots_exts):
        print(f"Сохранено: {path}")


if __name__ == "__main__":
    main()
