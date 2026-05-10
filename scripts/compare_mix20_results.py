import argparse
import csv
import io
import math
import pickle
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BATLINET_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_AUTHOR_WORKSPACE = (
    BATLINET_ROOT / "ds340.batlinet-main" / "code" / "workspaces" / "batlinet" / "mix_20"
)
DEFAULT_REPRO_WORKSPACE = REPO_ROOT / "workspaces" / "mix_20_0424"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "mix_20_comparison"
METRICS = ("RMSE", "MAE", "MAPE")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare BatLiNet mix_20 results from author and reproduction workspaces."
    )
    parser.add_argument(
        "--author-workspace",
        default=str(DEFAULT_AUTHOR_WORKSPACE),
        help="Path to the author workspace that contains predictions_seed_*.pkl files.",
    )
    parser.add_argument(
        "--repro-workspace",
        default=str(DEFAULT_REPRO_WORKSPACE),
        help="Path to the reproduction workspace that contains predictions_seed_*.pkl files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where tables, figures, and summary files will be saved.",
    )
    return parser.parse_args()


def ensure_output_dirs(output_dir: Path):
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    summary_dir = output_dir / "summary"
    for path in (tables_dir, figures_dir, summary_dir):
        path.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir, summary_dir


class PlaceholderObject:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["state"] = state


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda data: torch.load(
                io.BytesIO(data),
                map_location=torch.device("cpu"),
                weights_only=False,
            )
        if module.startswith("src."):
            return PlaceholderObject
        return super().find_class(module, name)


def load_prediction_file(path: Path):
    with path.open("rb") as fin:
        obj = CPUUnpickler(fin).load()

    if "scores" not in obj:
        raise KeyError(f"Missing 'scores' field in {path}")
    if "seed" not in obj:
        raise KeyError(f"Missing 'seed' field in {path}")

    scores = obj["scores"]
    row = {"seed": int(obj["seed"]), "path": str(path)}
    for metric in METRICS:
        if metric not in scores:
            raise KeyError(f"Missing metric '{metric}' in {path}")
        row[metric] = float(scores[metric])
    return row


def collect_workspace_results(workspace: Path):
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace does not exist: {workspace}")

    files = sorted(workspace.glob("predictions_seed_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No predictions_seed_*.pkl files found in {workspace}")

    rows = [load_prediction_file(file) for file in files]
    rows.sort(key=lambda item: item["seed"])
    return rows


def index_by_seed(rows):
    return {row["seed"]: row for row in rows}


def build_seed_comparison(author_rows, repro_rows):
    author_by_seed = index_by_seed(author_rows)
    repro_by_seed = index_by_seed(repro_rows)
    all_seeds = sorted(set(author_by_seed) | set(repro_by_seed))

    comparison_rows = []
    missing = []
    for seed in all_seeds:
        author = author_by_seed.get(seed)
        repro = repro_by_seed.get(seed)
        if author is None:
            missing.append(f"author seed {seed}")
        if repro is None:
            missing.append(f"reproduction seed {seed}")

        row = {"seed": seed}
        for metric in METRICS:
            author_value = author[metric] if author else None
            repro_value = repro[metric] if repro else None
            row[f"author_{metric.lower()}"] = author_value
            row[f"repro_{metric.lower()}"] = repro_value
            row[f"delta_{metric.lower()}"] = (
                None if author_value is None or repro_value is None else repro_value - author_value
            )
        comparison_rows.append(row)

    return comparison_rows, missing


def summarize_group(rows):
    summary = {}
    for metric in METRICS:
        values = [row[metric] for row in rows]
        best_row = min(rows, key=lambda item: item[metric])
        worst_row = max(rows, key=lambda item: item[metric])
        summary[metric] = {
            "count": len(values),
            "mean": sum(values) / len(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "best_seed": best_row["seed"],
            "best_value": best_row[metric],
            "worst_seed": worst_row["seed"],
            "worst_value": worst_row[metric],
        }
    return summary


def build_summary_rows(group_name, summary):
    rows = []
    for metric in METRICS:
        item = summary[metric]
        rows.append(
            {
                "group": group_name,
                "metric": metric,
                "count": item["count"],
                "mean": item["mean"],
                "std": item["std"],
                "best_seed": item["best_seed"],
                "best_value": item["best_value"],
                "worst_seed": item["worst_seed"],
                "worst_value": item["worst_value"],
            }
        )
    return rows


def format_float(value):
    if value is None:
        return "NA"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "NA"
    return f"{value:.4f}"


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: Path, content: str):
    with path.open("w", encoding="utf-8-sig") as fout:
        fout.write(content)


def plot_metric_means(author_summary, repro_summary, figures_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    for axis, metric in zip(axes, METRICS):
        author_mean = author_summary[metric]["mean"]
        repro_mean = repro_summary[metric]["mean"]
        bars = axis.bar(
            ["Author", "Reproduction"],
            [author_mean, repro_mean],
            color=["#4C78A8", "#F58518"],
            width=0.55,
        )
        axis.set_title(metric)
        axis.set_ylabel("Score")
        axis.grid(axis="y", linestyle="--", alpha=0.4)
        ymax = max(author_mean, repro_mean)
        axis.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)
        for bar, value in zip(bars, (author_mean, repro_mean)):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + ymax * 0.03,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("mix_20 Mean Metrics Comparison")
    fig.tight_layout()
    fig.savefig(figures_dir / "metric_means_bar.png", dpi=200)
    plt.close(fig)


def plot_seed_lines(author_rows, repro_rows, figures_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for axis, metric in zip(axes, METRICS):
        axis.plot(
            [row["seed"] for row in author_rows],
            [row[metric] for row in author_rows],
            marker="o",
            label="Author",
        )
        axis.plot(
            [row["seed"] for row in repro_rows],
            [row[metric] for row in repro_rows],
            marker="o",
            label="Reproduction",
        )
        axis.set_title(metric)
        axis.set_xlabel("Seed")
        axis.set_ylabel("Score")
        axis.grid(linestyle="--", alpha=0.4)
    axes[0].legend()
    fig.suptitle("mix_20 Metrics Across Seeds")
    fig.tight_layout()
    fig.savefig(figures_dir / "metrics_by_seed_lines.png", dpi=200)
    plt.close(fig)


def plot_metric_boxplots(author_rows, repro_rows, figures_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, metric in zip(axes, METRICS):
        axis.boxplot(
            [
                [row[metric] for row in author_rows],
                [row[metric] for row in repro_rows],
            ],
            tick_labels=["Author", "Reproduction"],
        )
        axis.set_title(metric)
        axis.set_ylabel("Score")
        axis.grid(axis="y", linestyle="--", alpha=0.4)
    fig.suptitle("mix_20 Metric Distribution by Group")
    fig.tight_layout()
    fig.savefig(figures_dir / "metric_distributions_boxplot.png", dpi=200)
    plt.close(fig)


def infer_overall_conclusion(author_summary, repro_summary):
    comparisons = {}
    for metric in METRICS:
        author_mean = author_summary[metric]["mean"]
        repro_mean = repro_summary[metric]["mean"]
        delta = repro_mean - author_mean
        relative_gap = abs(delta) / author_mean if author_mean else 0.0
        comparisons[metric] = {
            "author_mean": author_mean,
            "repro_mean": repro_mean,
            "delta": delta,
            "relative_gap": relative_gap,
        }

    close_metrics = sum(item["relative_gap"] <= 0.10 for item in comparisons.values())
    if close_metrics == len(METRICS):
        verdict = "你的复现结果与作者结果整体接近，可以认为 mix_20 已经较好复现。"
    elif close_metrics >= 2:
        verdict = "你的复现结果总体较接近作者结果，但仍有个别指标存在中等差异。"
    else:
        verdict = "你的复现结果与作者结果存在明显差异，建议进一步排查。"

    better_metrics = [metric for metric, item in comparisons.items() if item["delta"] < 0]
    worse_metrics = [metric for metric, item in comparisons.items() if item["delta"] > 0]
    if not worse_metrics:
        drift = "当前复现结果在三项指标上都不差于作者均值。"
    elif not better_metrics:
        drift = "当前复现结果在三项指标上都差于作者均值。"
    else:
        drift = "三项指标呈现有优有劣的混合状态，更像是 seed 波动而不是单一系统性错误。"
    return comparisons, verdict, drift


def metric_meaning_text(metric: str):
    mapping = {
        "RMSE": "RMSE 对大误差更敏感，用来观察模型是否存在少量偏差特别大的样本。",
        "MAE": "MAE 表示平均绝对误差，用来衡量整体预测偏差的平均水平。",
        "MAPE": "MAPE 表示平均相对误差，用来衡量误差相对于真实寿命的比例。",
    }
    return mapping[metric]


def metric_conclusion_text(metric: str, author_mean: float, repro_mean: float):
    delta = repro_mean - author_mean
    relative_gap = abs(delta) / author_mean if author_mean else 0.0

    if metric == "RMSE":
        if delta < 0:
            base = (
                f"你的复现 RMSE 比作者更低 {abs(delta):.4f}，"
                "说明大误差样本整体控制得更好。"
            )
        elif delta > 0:
            base = (
                f"你的复现 RMSE 比作者更高 {abs(delta):.4f}，"
                "说明大误差样本略多或更极端。"
            )
        else:
            base = "你的复现 RMSE 与作者基本相同，说明大误差样本控制水平接近。"
    elif metric == "MAE":
        if delta < 0:
            base = (
                f"你的复现 MAE 比作者更低 {abs(delta):.4f}，"
                "说明平均绝对偏差略优于作者结果。"
            )
        elif delta > 0:
            base = (
                f"你的复现 MAE 比作者更高 {abs(delta):.4f}，"
                "说明整体平均误差几乎持平但略高。"
            )
        else:
            base = "你的复现 MAE 与作者基本相同，说明整体平均误差非常接近。"
    else:
        if delta < 0:
            base = (
                f"你的复现 MAPE 比作者更低 {abs(delta):.4f}，"
                "说明相对误差水平更低。"
            )
        elif delta > 0:
            base = (
                f"你的复现 MAPE 比作者更高 {abs(delta):.4f}，"
                "说明相对误差水平略高。"
            )
        else:
            base = "你的复现 MAPE 与作者基本相同，说明相对误差水平接近。"

    tail = (
        "这一项与作者结果已经比较接近。"
        if relative_gap <= 0.10
        else "这一项和作者结果仍有较明显差距，后续可以继续排查。"
    )
    return f"{base}{tail}"


def build_seed_markdown_table(rows):
    lines = [
        "| Seed | 作者 RMSE | 复现 RMSE | 差值 | 作者 MAE | 复现 MAE | 差值 | 作者 MAPE | 复现 MAPE | 差值 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {seed} | {author_rmse} | {repro_rmse} | {delta_rmse} | {author_mae} | {repro_mae} | {delta_mae} | {author_mape} | {repro_mape} | {delta_mape} |".format(
                seed=row["seed"],
                author_rmse=format_float(row["author_rmse"]),
                repro_rmse=format_float(row["repro_rmse"]),
                delta_rmse=format_float(row["delta_rmse"]),
                author_mae=format_float(row["author_mae"]),
                repro_mae=format_float(row["repro_mae"]),
                delta_mae=format_float(row["delta_mae"]),
                author_mape=format_float(row["author_mape"]),
                repro_mape=format_float(row["repro_mape"]),
                delta_mape=format_float(row["delta_mape"]),
            )
        )
    return "\n".join(lines) + "\n"


def build_summary_markdown_table(rows):
    group_names = {"author": "作者", "reproduction": "复现"}
    lines = [
        "| 分组 | 指标 | 数量 | 均值 | 标准差 | 最优 Seed | 最优值 | 最差 Seed | 最差值 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {group} | {metric} | {count} | {mean} | {std} | {best_seed} | {best_value} | {worst_seed} | {worst_value} |".format(
                group=group_names.get(row["group"], row["group"]),
                metric=row["metric"],
                count=row["count"],
                mean=format_float(row["mean"]),
                std=format_float(row["std"]),
                best_seed=row["best_seed"],
                best_value=format_float(row["best_value"]),
                worst_seed=row["worst_seed"],
                worst_value=format_float(row["worst_value"]),
            )
        )
    return "\n".join(lines) + "\n"


def build_three_line_html(seed_rows, summary_rows):
    group_names = {"author": "作者", "reproduction": "复现"}
    lines = [
        "<html><head><meta charset='utf-8'>",
        "<style>",
        "body { font-family: 'Microsoft YaHei', sans-serif; margin: 24px; color: #222; }",
        "h1, h2 { margin-bottom: 12px; }",
        "table { border-collapse: collapse; width: 100%; margin-bottom: 28px; font-size: 14px; }",
        "thead tr:first-child th { border-top: 2px solid #222; }",
        "thead th { border-bottom: 1.5px solid #222; padding: 8px 10px; text-align: center; }",
        "tbody td { padding: 8px 10px; text-align: right; }",
        "tbody td:first-child, tbody td:nth-child(2) { text-align: center; }",
        "tbody tr:last-child td { border-bottom: 2px solid #222; }",
        "tbody tr:nth-child(even) td { background: #f7f7f7; }",
        "</style>",
        "</head><body>",
        "<h1>BatLiNet mix_20 结果对比表</h1>",
        "<h2>Seed 级别对比</h2>",
        "<table>",
        "<thead><tr><th>Seed</th><th>作者 RMSE</th><th>复现 RMSE</th><th>差值</th><th>作者 MAE</th><th>复现 MAE</th><th>差值</th><th>作者 MAPE</th><th>复现 MAPE</th><th>差值</th></tr></thead>",
        "<tbody>",
    ]
    for row in seed_rows:
        lines.append(
            "<tr>"
            f"<td>{row['seed']}</td>"
            f"<td>{format_float(row['author_rmse'])}</td>"
            f"<td>{format_float(row['repro_rmse'])}</td>"
            f"<td>{format_float(row['delta_rmse'])}</td>"
            f"<td>{format_float(row['author_mae'])}</td>"
            f"<td>{format_float(row['repro_mae'])}</td>"
            f"<td>{format_float(row['delta_mae'])}</td>"
            f"<td>{format_float(row['author_mape'])}</td>"
            f"<td>{format_float(row['repro_mape'])}</td>"
            f"<td>{format_float(row['delta_mape'])}</td>"
            "</tr>"
        )
    lines.extend(
        [
            "</tbody></table>",
            "<h2>汇总统计</h2>",
            "<table>",
            "<thead><tr><th>分组</th><th>指标</th><th>数量</th><th>均值</th><th>标准差</th><th>最优 Seed</th><th>最优值</th><th>最差 Seed</th><th>最差值</th></tr></thead>",
            "<tbody>",
        ]
    )
    for row in summary_rows:
        lines.append(
            "<tr>"
            f"<td>{group_names.get(row['group'], row['group'])}</td>"
            f"<td>{row['metric']}</td>"
            f"<td>{row['count']}</td>"
            f"<td>{format_float(row['mean'])}</td>"
            f"<td>{format_float(row['std'])}</td>"
            f"<td>{row['best_seed']}</td>"
            f"<td>{format_float(row['best_value'])}</td>"
            f"<td>{row['worst_seed']}</td>"
            f"<td>{format_float(row['worst_value'])}</td>"
            "</tr>"
        )
    lines.extend(["</tbody></table>", "</body></html>"])
    return "\n".join(lines) + "\n"


def build_summary_text(author_workspace, repro_workspace, author_summary, repro_summary, missing):
    comparisons, verdict, drift = infer_overall_conclusion(author_summary, repro_summary)

    lines = [
        "BatLiNet mix_20 结果对比总结",
        "",
        f"作者结果目录: {author_workspace}",
        f"复现结果目录: {repro_workspace}",
        "",
        "指标含义与对比结论:",
    ]

    for metric in METRICS:
        item = comparisons[metric]
        lines.append(f"- {metric} 含义: {metric_meaning_text(metric)}")
        lines.append(
            f"  作者均值 {item['author_mean']:.4f}，复现均值 {item['repro_mean']:.4f}，"
            f"差值 {item['delta']:+.4f}，相对差距 {item['relative_gap'] * 100:.2f}%"
        )
        lines.append(
            f"  结论: {metric_conclusion_text(metric, item['author_mean'], item['repro_mean'])}"
        )

    lines.extend(
        [
            "",
            f"总体判断: {verdict}",
            f"解释: {drift}",
            "",
            "Seed 覆盖情况:",
        ]
    )
    if missing:
        for item in missing:
            lines.append(f"- 缺失: {item}")
    else:
        lines.append("- 两个目录中的 8 个 seed 结果都已成功读取。")

    lines.extend(["", "两组结果的离散程度:"])
    for group_name, summary in (("作者", author_summary), ("复现", repro_summary)):
        lines.append(f"- {group_name}:")
        for metric in METRICS:
            item = summary[metric]
            lines.append(
                f"  {metric}: 均值 {item['mean']:.4f}，标准差 {item['std']:.4f}，"
                f"最佳 seed 为 {item['best_seed']} ({item['best_value']:.4f})，"
                f"最差 seed 为 {item['worst_seed']} ({item['worst_value']:.4f})"
            )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    author_workspace = Path(args.author_workspace)
    repro_workspace = Path(args.repro_workspace)
    output_dir = Path(args.output_dir)
    tables_dir, figures_dir, summary_dir = ensure_output_dirs(output_dir)

    author_rows = collect_workspace_results(author_workspace)
    repro_rows = collect_workspace_results(repro_workspace)
    comparison_rows, missing = build_seed_comparison(author_rows, repro_rows)

    author_summary = summarize_group(author_rows)
    repro_summary = summarize_group(repro_rows)
    summary_rows = build_summary_rows("author", author_summary) + build_summary_rows(
        "reproduction", repro_summary
    )

    write_csv(
        tables_dir / "seed_level_comparison.csv",
        [
            "seed",
            "author_rmse",
            "repro_rmse",
            "delta_rmse",
            "author_mae",
            "repro_mae",
            "delta_mae",
            "author_mape",
            "repro_mape",
            "delta_mape",
        ],
        comparison_rows,
    )
    write_csv(
        tables_dir / "summary_statistics.csv",
        [
            "group",
            "metric",
            "count",
            "mean",
            "std",
            "best_seed",
            "best_value",
            "worst_seed",
            "worst_value",
        ],
        summary_rows,
    )
    write_text(tables_dir / "seed_level_comparison.md", build_seed_markdown_table(comparison_rows))
    write_text(tables_dir / "summary_statistics.md", build_summary_markdown_table(summary_rows))
    write_text(tables_dir / "comparison_tables.html", build_three_line_html(comparison_rows, summary_rows))

    plot_metric_means(author_summary, repro_summary, figures_dir)
    plot_seed_lines(author_rows, repro_rows, figures_dir)
    plot_metric_boxplots(author_rows, repro_rows, figures_dir)

    write_text(
        summary_dir / "comparison_summary.txt",
        build_summary_text(
            author_workspace,
            repro_workspace,
            author_summary,
            repro_summary,
            missing,
        ),
    )

    print(f"Saved outputs to: {output_dir}")
    print(f"Author seeds loaded: {len(author_rows)}")
    print(f"Reproduction seeds loaded: {len(repro_rows)}")
    print(
        "Author means:",
        ", ".join(f"{metric}={format_float(author_summary[metric]['mean'])}" for metric in METRICS),
    )
    print(
        "Reproduction means:",
        ", ".join(f"{metric}={format_float(repro_summary[metric]['mean'])}" for metric in METRICS),
    )


if __name__ == "__main__":
    main()
