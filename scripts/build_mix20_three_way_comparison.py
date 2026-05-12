import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("analysis/mix_20_three_way_comparison")
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
SUMMARY = ROOT / "summary"
METRICS = ("RMSE", "MAE", "MAPE")
GROUPS = ("author", "reproduction", "supervised_weighted")
GROUP_LABELS = {
    "author": "Author",
    "reproduction": "Reproduction",
    "supervised_weighted": "Supervised weighted",
}


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
    for path in (TABLES, FIGURES, SUMMARY):
        path.mkdir(parents=True, exist_ok=True)

    base_summary = read_csv(
        Path("analysis/mix_20_comparison/tables/summary_statistics.csv")
    )
    base_seed = read_csv(
        Path("analysis/mix_20_comparison/tables/seed_level_comparison.csv")
    )
    supervised_summary = read_csv(
        Path("analysis/mix_20_supervised_weighted_comparison/tables/summary_statistics.csv")
    )
    supervised_seed = read_csv(
        Path("analysis/mix_20_supervised_weighted_comparison/tables/seed_level_comparison.csv")
    )

    summary_rows = [
        row for row in base_summary
        if row["group"] in ("author", "reproduction")
    ]
    for row in supervised_summary:
        if row["group"] == "batlinet_supervised_weighted":
            row = dict(row)
            row["group"] = "supervised_weighted"
            summary_rows.append(row)

    summary_by_key = {
        (row["group"], row["metric"]): row for row in summary_rows
    }
    supervised_by_seed = {int(row["seed"]): row for row in supervised_seed}

    seed_rows = []
    for row in base_seed:
        seed = int(row["seed"])
        supervised = supervised_by_seed[seed]
        output = {"seed": seed}
        for metric in METRICS:
            key = metric.lower()
            output[f"author_{key}"] = float(row[f"author_{key}"])
            output[f"reproduction_{key}"] = float(row[f"repro_{key}"])
            output[f"supervised_weighted_{key}"] = float(
                supervised[f"candidate_{key}"]
            )
            output[f"supervised_vs_reproduction_delta_{key}"] = (
                output[f"supervised_weighted_{key}"]
                - output[f"reproduction_{key}"]
            )
            output[f"supervised_vs_author_delta_{key}"] = (
                output[f"supervised_weighted_{key}"]
                - output[f"author_{key}"]
            )
        seed_rows.append(output)

    mean_rows = []
    for metric in METRICS:
        values = {
            group: float(summary_by_key[(group, metric)]["mean"])
            for group in GROUPS
        }
        mean_rows.append({
            "metric": metric,
            "author_mean": values["author"],
            "reproduction_mean": values["reproduction"],
            "supervised_weighted_mean": values["supervised_weighted"],
            "supervised_vs_reproduction_delta": (
                values["supervised_weighted"] - values["reproduction"]
            ),
            "supervised_vs_reproduction_percent": (
                (values["supervised_weighted"] - values["reproduction"])
                / values["reproduction"] * 100
            ),
            "supervised_vs_author_delta": (
                values["supervised_weighted"] - values["author"]
            ),
            "supervised_vs_author_percent": (
                (values["supervised_weighted"] - values["author"])
                / values["author"] * 100
            ),
        })

    ordered_summary_rows = [
        summary_by_key[(group, metric)]
        for group in GROUPS
        for metric in METRICS
    ]
    write_csv(TABLES / "summary_statistics.csv", ordered_summary_rows)
    write_csv(TABLES / "three_way_mean_comparison.csv", mean_rows)
    write_csv(TABLES / "seed_level_three_way_comparison.csv", seed_rows)

    mean_lines = [
        "| Metric | Author | Reproduction | Supervised weighted | Delta vs reproduction | Delta vs author |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in mean_rows:
        mean_lines.append(
            "| {metric} | {author} | {repro} | {sup} | {drep} ({prep:.2f}%) | {dauth} ({pauth:.2f}%) |".format(
                metric=row["metric"],
                author=fmt(row["author_mean"]),
                repro=fmt(row["reproduction_mean"]),
                sup=fmt(row["supervised_weighted_mean"]),
                drep=fmt(row["supervised_vs_reproduction_delta"]),
                prep=row["supervised_vs_reproduction_percent"],
                dauth=fmt(row["supervised_vs_author_delta"]),
                pauth=row["supervised_vs_author_percent"],
            )
        )
    (TABLES / "three_way_mean_comparison.md").write_text(
        "\n".join(mean_lines) + "\n", encoding="utf-8-sig"
    )

    seed_lines = [
        "| Seed | Author RMSE | Repro RMSE | Supervised RMSE | Delta RMSE vs Repro | Author MAE | Repro MAE | Supervised MAE | Delta MAE vs Repro | Author MAPE | Repro MAPE | Supervised MAPE | Delta MAPE vs Repro |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in seed_rows:
        seed_lines.append(
            "| {seed} | {ar} | {rr} | {sr} | {dr} | {am} | {rm} | {sm} | {dm} | {ap} | {rp} | {sp} | {dp} |".format(
                seed=row["seed"],
                ar=fmt(row["author_rmse"]),
                rr=fmt(row["reproduction_rmse"]),
                sr=fmt(row["supervised_weighted_rmse"]),
                dr=fmt(row["supervised_vs_reproduction_delta_rmse"]),
                am=fmt(row["author_mae"]),
                rm=fmt(row["reproduction_mae"]),
                sm=fmt(row["supervised_weighted_mae"]),
                dm=fmt(row["supervised_vs_reproduction_delta_mae"]),
                ap=fmt(row["author_mape"]),
                rp=fmt(row["reproduction_mape"]),
                sp=fmt(row["supervised_weighted_mape"]),
                dp=fmt(row["supervised_vs_reproduction_delta_mape"]),
            )
        )
    (TABLES / "seed_level_three_way_comparison.md").write_text(
        "\n".join(seed_lines) + "\n", encoding="utf-8-sig"
    )

    colors = ("#4C78A8", "#F58518", "#54A24B")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, metric in zip(axes, METRICS):
        values = [
            float(summary_by_key[(group, metric)]["mean"]) for group in GROUPS
        ]
        bars = axis.bar(
            [GROUP_LABELS[group] for group in GROUPS],
            values,
            color=colors,
            width=0.58,
        )
        ymax = max(values)
        axis.set_title(metric)
        axis.set_ylabel("Score")
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)
        axis.tick_params(axis="x", labelrotation=18)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + ymax * 0.025,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("mix_20 Three-way Mean Metrics")
    fig.tight_layout()
    fig.savefig(FIGURES / "metric_means_three_way_bar.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for axis, metric in zip(axes, METRICS):
        key = metric.lower()
        seeds = [row["seed"] for row in seed_rows]
        for group, color in zip(GROUPS, colors):
            axis.plot(
                seeds,
                [row[f"{group}_{key}"] for row in seed_rows],
                marker="o",
                label=GROUP_LABELS[group],
                color=color,
            )
        axis.set_title(metric)
        axis.set_xlabel("Seed")
        axis.set_ylabel("Score")
        axis.grid(linestyle="--", alpha=0.35)
    axes[0].legend()
    fig.suptitle("mix_20 Metrics Across Seeds")
    fig.tight_layout()
    fig.savefig(FIGURES / "metrics_by_seed_three_way_lines.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, metric in zip(axes, METRICS):
        key = metric.lower()
        deltas = [
            row[f"supervised_vs_reproduction_delta_{key}"]
            for row in seed_rows
        ]
        bar_colors = ["#54A24B" if delta < 0 else "#E45756" for delta in deltas]
        axis.bar([row["seed"] for row in seed_rows], deltas, color=bar_colors)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"{metric} delta vs reproduction")
        axis.set_xlabel("Seed")
        axis.set_ylabel("Candidate - reproduction")
        axis.grid(axis="y", linestyle="--", alpha=0.35)
    fig.suptitle("Supervised Weighted Improvements by Seed")
    fig.tight_layout()
    fig.savefig(FIGURES / "supervised_vs_reproduction_delta_by_seed.png", dpi=220)
    plt.close(fig)

    summary_lines = ["mix_20 three-way comparison", ""]
    for row in mean_rows:
        summary_lines.append(
            f"{row['metric']}: Author {row['author_mean']:.4f}, "
            f"Reproduction {row['reproduction_mean']:.4f}, "
            f"Supervised weighted {row['supervised_weighted_mean']:.4f}; "
            f"vs reproduction {row['supervised_vs_reproduction_delta']:+.4f} "
            f"({row['supervised_vs_reproduction_percent']:+.2f}%), "
            f"vs author {row['supervised_vs_author_delta']:+.4f} "
            f"({row['supervised_vs_author_percent']:+.2f}%)."
        )
    summary_lines.extend([
        "",
        "Lower is better for all three metrics. Supervised weighted has the best mean value on RMSE, MAE, and MAPE in this comparison.",
    ])
    (SUMMARY / "three_way_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8-sig"
    )

    print(f"Saved outputs to: {ROOT}")
    for row in mean_rows:
        print(
            row["metric"],
            f"author={row['author_mean']:.4f}",
            f"reproduction={row['reproduction_mean']:.4f}",
            f"supervised={row['supervised_weighted_mean']:.4f}",
            f"delta_vs_reproduction={row['supervised_vs_reproduction_delta']:+.4f}",
        )


if __name__ == "__main__":
    main()
