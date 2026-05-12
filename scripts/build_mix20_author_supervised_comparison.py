import csv
from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_ROOT = Path("analysis/mix_20_author_supervised_comparison")
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"
SUMMARY_DIR = OUTPUT_ROOT / "summary"
METRICS = ("RMSE", "MAE", "MAPE")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as fin:
        return list(csv.DictReader(fin))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main():
    for path in (TABLES_DIR, FIGURES_DIR, SUMMARY_DIR):
        path.mkdir(parents=True, exist_ok=True)

    author_repro_summary = read_csv(
        Path("analysis/mix_20_comparison/tables/summary_statistics.csv")
    )
    author_repro_seed = read_csv(
        Path("analysis/mix_20_comparison/tables/seed_level_comparison.csv")
    )
    supervised_summary = read_csv(
        Path("analysis/mix_20_supervised_weighted_comparison/tables/summary_statistics.csv")
    )
    supervised_seed = read_csv(
        Path("analysis/mix_20_supervised_weighted_comparison/tables/seed_level_comparison.csv")
    )

    author_summary = {
        row["metric"]: row for row in author_repro_summary
        if row["group"] == "author"
    }
    supervised_summary_map = {
        row["metric"]: row for row in supervised_summary
        if row["group"] == "batlinet_supervised_weighted"
    }
    supervised_by_seed = {int(row["seed"]): row for row in supervised_seed}

    summary_rows = []
    for metric in METRICS:
        author_mean = float(author_summary[metric]["mean"])
        supervised_mean = float(supervised_summary_map[metric]["mean"])
        delta = supervised_mean - author_mean
        summary_rows.append({
            "metric": metric,
            "author_mean": author_mean,
            "author_std": float(author_summary[metric]["std"]),
            "supervised_weighted_mean": supervised_mean,
            "supervised_weighted_std": float(supervised_summary_map[metric]["std"]),
            "delta": delta,
            "delta_percent": delta / author_mean * 100,
            "author_best_seed": int(author_summary[metric]["best_seed"]),
            "author_best_value": float(author_summary[metric]["best_value"]),
            "supervised_best_seed": int(supervised_summary_map[metric]["best_seed"]),
            "supervised_best_value": float(supervised_summary_map[metric]["best_value"]),
        })

    seed_rows = []
    for row in author_repro_seed:
        seed = int(row["seed"])
        supervised = supervised_by_seed[seed]
        output = {"seed": seed}
        for metric in METRICS:
            key = metric.lower()
            output[f"author_{key}"] = float(row[f"author_{key}"])
            output[f"supervised_weighted_{key}"] = float(
                supervised[f"candidate_{key}"]
            )
            output[f"delta_{key}"] = (
                output[f"supervised_weighted_{key}"]
                - output[f"author_{key}"]
            )
        seed_rows.append(output)

    write_csv(TABLES_DIR / "author_vs_supervised_summary.csv", summary_rows)
    write_csv(TABLES_DIR / "author_vs_supervised_seed_level.csv", seed_rows)

    summary_lines = [
        "| Metric | Author mean | Author std | Supervised weighted mean | Supervised weighted std | Delta | Delta percent |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        summary_lines.append(
            "| {metric} | {author_mean} | {author_std} | {supervised_mean} | {supervised_std} | {delta} | {delta_percent:.2f}% |".format(
                metric=row["metric"],
                author_mean=fmt(row["author_mean"]),
                author_std=fmt(row["author_std"]),
                supervised_mean=fmt(row["supervised_weighted_mean"]),
                supervised_std=fmt(row["supervised_weighted_std"]),
                delta=fmt(row["delta"]),
                delta_percent=row["delta_percent"],
            )
        )
    (TABLES_DIR / "author_vs_supervised_summary.md").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8-sig",
    )

    seed_lines = [
        "| Seed | Author RMSE | Supervised RMSE | Delta RMSE | Author MAE | Supervised MAE | Delta MAE | Author MAPE | Supervised MAPE | Delta MAPE |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in seed_rows:
        seed_lines.append(
            "| {seed} | {author_rmse} | {sup_rmse} | {delta_rmse} | {author_mae} | {sup_mae} | {delta_mae} | {author_mape} | {sup_mape} | {delta_mape} |".format(
                seed=row["seed"],
                author_rmse=fmt(row["author_rmse"]),
                sup_rmse=fmt(row["supervised_weighted_rmse"]),
                delta_rmse=fmt(row["delta_rmse"]),
                author_mae=fmt(row["author_mae"]),
                sup_mae=fmt(row["supervised_weighted_mae"]),
                delta_mae=fmt(row["delta_mae"]),
                author_mape=fmt(row["author_mape"]),
                sup_mape=fmt(row["supervised_weighted_mape"]),
                delta_mape=fmt(row["delta_mape"]),
            )
        )
    (TABLES_DIR / "author_vs_supervised_seed_level.md").write_text(
        "\n".join(seed_lines) + "\n",
        encoding="utf-8-sig",
    )

    colors = ("#4C78A8", "#C13829")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    for axis, metric in zip(axes, METRICS):
        row = next(item for item in summary_rows if item["metric"] == metric)
        values = [row["author_mean"], row["supervised_weighted_mean"]]
        bars = axis.bar(
            ["Author", "Supervised weighted"],
            values,
            width=0.55,
            color=colors,
        )
        ymax = max(values)
        axis.set_title(metric)
        axis.set_ylabel("Score")
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)
        axis.tick_params(axis="x", labelrotation=12)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + ymax * 0.025,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("mix_20 Author vs Supervised Weighted Mean Metrics")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "author_vs_supervised_metric_means_bar.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    seeds = [row["seed"] for row in seed_rows]
    for axis, metric in zip(axes, METRICS):
        key = metric.lower()
        axis.plot(
            seeds,
            [row[f"author_{key}"] for row in seed_rows],
            marker="o",
            label="Author",
            color=colors[0],
        )
        axis.plot(
            seeds,
            [row[f"supervised_weighted_{key}"] for row in seed_rows],
            marker="o",
            label="Supervised weighted",
            color=colors[1],
        )
        axis.set_title(metric)
        axis.set_xlabel("Seed")
        axis.set_ylabel("Score")
        axis.grid(linestyle="--", alpha=0.35)
    axes[0].legend()
    fig.suptitle("mix_20 Author vs Supervised Weighted Across Seeds")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "author_vs_supervised_seed_lines.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    for axis, metric in zip(axes, METRICS):
        key = metric.lower()
        deltas = [row[f"delta_{key}"] for row in seed_rows]
        bar_colors = ["#54A24B" if delta < 0 else "#E45756" for delta in deltas]
        axis.bar(seeds, deltas, color=bar_colors)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"{metric} delta")
        axis.set_xlabel("Seed")
        axis.set_ylabel("Supervised weighted - Author")
        axis.grid(axis="y", linestyle="--", alpha=0.35)
    fig.suptitle("mix_20 Seed-level Delta Against Author Results")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "author_vs_supervised_seed_deltas.png", dpi=220)
    plt.close(fig)

    text_lines = ["mix_20 author vs supervised weighted comparison", ""]
    for row in summary_rows:
        direction = "better" if row["delta"] < 0 else "worse" if row["delta"] > 0 else "equal"
        text_lines.append(
            f"{row['metric']}: author {row['author_mean']:.4f}, "
            f"supervised weighted {row['supervised_weighted_mean']:.4f}, "
            f"delta {row['delta']:+.4f} ({row['delta_percent']:+.2f}%), {direction}."
        )
    text_lines.extend([
        "",
        "Lower is better for all metrics.",
    ])
    (SUMMARY_DIR / "author_vs_supervised_summary.txt").write_text(
        "\n".join(text_lines) + "\n",
        encoding="utf-8-sig",
    )

    print(f"Saved outputs to: {OUTPUT_ROOT}")
    for row in summary_rows:
        print(
            row["metric"],
            f"author={row['author_mean']:.4f}",
            f"supervised={row['supervised_weighted_mean']:.4f}",
            f"delta={row['delta']:+.4f}",
            f"percent={row['delta_percent']:+.2f}%",
        )


if __name__ == "__main__":
    main()
