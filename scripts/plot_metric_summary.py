import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def short_metric_name(metric_file: str) -> str:
    """
    Удобочитаемые короткие названия для графиков.
    Расширяйте это сопоставление по мере добавления новых вариантов.

    ВАЖНО:

    - Существующие названия антисимметрии сохраняются без изменений (не изменяйте выходные данные для уже вычисленных экспериментов).

    - Добавлены симметричные (_sym) псевдонимы без влияния на антисимметрию.
    """
    name = os.path.basename(str(metric_file))
    if name.lower().endswith(".npz"):
        name = name[:-4]

    # Убираем общий префикс
    name = name.replace("local_map_rank_", "")

    # -------------------------
    # Варианты для EPS
    # -------------------------
    if "linear_eps_percentile_5_antisym" in name:
        return "lin_eps_5"
    if "linear_eps_percentile_10_antisym" in name:
        return "lin_eps_10"
    if "linear_eps_percentile_20_antisym" in name:
        return "lin_eps_20"

    if "linear_eps_percentile_5_sym" in name:
        return "lin_eps_5_sym"
    if "linear_eps_percentile_10_sym" in name:
        return "lin_eps_10_sym"
    if "linear_eps_percentile_20_sym" in name:
        return "lin_eps_20_sym"

    # -------------------------
    # Линейный kNN, антисимметричный вариант
    # -------------------------
    if "linear_knn_k5_antisym" in name:
        return "lin_k5"
    if "linear_knn_k10_antisym" in name:
        return "lin_k10"
    if "linear_knn_k20_antisym" in name:
        return "lin_k20"
    if "linear_knn_k40_antisym" in name:
        return "lin_k40"

    # Линейный kNN, симметричный вариант
    if "linear_knn_k5_sym" in name:
        return "lin_k5_sym"
    if "linear_knn_k10_sym" in name:
        return "lin_k10_sym"
    if "linear_knn_k20_sym" in name:
        return "lin_k20_sym"
    if "linear_knn_k40_sym" in name:
        return "lin_k40_sym"

    # -------------------------
    # Мульти-масштабный вариант
    # -------------------------
    if "multiscale_knn_mean_antisym" in name:
        return "multiscale_mean"
    if "multiscale_knn_mean_sym" in name:
        return "multiscale_mean_sym"

    # -------------------------
    # Вариант с RFF
    # -------------------------
    if "rff_knn_k10_antisym" in name:
        return "rff_k10"
    if "rff_knn_k10_sym" in name:
        return "rff_k10_sym"

    # -------------------------
    # Неантисимметричный вариант
    # -------------------------
    if name == "local_map_rank_linear_knn_k10":
        return "directed_k10"
    if "linear_knn_k10" in name and ("antisym" not in name) and ("_sym" not in name):
        return "directed_k10"

    return name


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_csv", required=True, help="CSV с оценкой, полученный из run_evaluate_metrics.py")
    ap.add_argument("--out_dir", required=True, help="Куда сохранить график")
    ap.add_argument("--dataset", required=True, help="Имя датасета в заголовке (например, cifar10, imagenet100)")
    ap.add_argument("--protocol", default="Δacc protocol", help="Подпись протокола в заголовке")
    ap.add_argument("--title", default="", help="Необязательный дополнительный суффикс заголовка")
    ap.add_argument("--out_name", default="metrics_summary.png", help="Имя выходного файла (png)")
    args = ap.parse_args()

    _ensure_dir(args.out_dir)
    df = pd.read_csv(args.eval_csv)

    # Обязательные столбцы в вашем текущем CSV-файле
    required = ["metric_file", "spearman_mean", "pearson_mean", "kendall_mean"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Столбец '{c}' не найден в CSV. Доступные столбцы: {list(df.columns)}")

    df = df.copy()
    df["metric"] = df["metric_file"].apply(short_metric_name)

    # Решаем, можем ли мы отображать панель CR:
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

    # ---- Заголовок
    title_parts = [str(args.dataset).upper(), str(args.protocol)]
    if args.title.strip():
        title_parts.append(args.title.strip())
    title_text = " | ".join(title_parts)

    # ---- Рисунок
    if show_cr_panel:
        fig = plt.figure(figsize=(12, max(6, 0.55 * len(metrics) + 2)))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.4], hspace=0.25)
        fig.suptitle(title_text, y=0.98)

        # Панель 1: CR
        ax1 = fig.add_subplot(gs[0, 0])
        cr_vals = pd.to_numeric(df[cr_col], errors="coerce").to_numpy(dtype=float)
        ax1.barh(y, cr_vals)
        ax1.set_yticks(y)
        ax1.set_yticklabels(metrics)
        ax1.invert_yaxis()
        ax1.set_xlim(0.0, 1.0)
        ax1.set_xlabel(
            "Доля правильного ранжирования (с поправкой)" if cr_col == "correct_ratio_adjusted_mean" else "Доля правильного ранжирования"
        )
        ax1.set_title("Доля правильного ранжирования")
        ax1.grid(True, axis="x", alpha=0.3)

        # Подписи значений на барах
        for yi, v in zip(y, cr_vals):
            if np.isnan(v):
                continue
            ax1.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=9)

        # Панель 2: корреляции
        ax2 = fig.add_subplot(gs[1, 0])
        width = 0.25
        sp = df["spearman_mean"].to_numpy(dtype=float)
        ke = df["kendall_mean"].to_numpy(dtype=float)
        pe = df["pearson_mean"].to_numpy(dtype=float)

        ax2.barh(y - width, sp, height=width, label="Spearman")
        ax2.barh(y,         ke, height=width, label="Kendall")
        ax2.barh(y + width, pe, height=width, label="Pearson")

        ax2.set_yticks(y)
        ax2.set_yticklabels(metrics)
        ax2.invert_yaxis()
        ax2.set_xlim(-1.0, 1.0)
        ax2.set_xlabel("Корреляция")
        ax2.set_title("Корреляции")
        ax2.grid(True, axis="x", alpha=0.3)
        ax2.legend(loc="lower right")

        out_path = os.path.join(args.out_dir, args.out_name)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Сохранено: {out_path}")
        return

    # ---- Sym / delta_abs: только корреляции (CR не определено)
    fig = plt.figure(figsize=(12, max(6, 0.55 * len(metrics) + 2)))
    ax = fig.add_subplot(111)
    fig.suptitle(title_text, y=0.98)

    width = 0.25
    sp = df["spearman_mean"].to_numpy(dtype=float)
    ke = df["kendall_mean"].to_numpy(dtype=float)
    pe = df["pearson_mean"].to_numpy(dtype=float)

    ax.barh(y - width, sp, height=width, label="Spearman")
    ax.barh(y,         ke, height=width, label="Kendall")
    ax.barh(y + width, pe, height=width, label="Pearson")

    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.invert_yaxis()
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Корреляция (CR не определено для протокола |Δacc|)")
    ax.set_title("Корреляции")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right")

    out_path = os.path.join(args.out_dir, args.out_name)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()