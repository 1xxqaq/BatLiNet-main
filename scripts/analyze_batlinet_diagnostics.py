import argparse
import csv
import io
import math
import pickle
import statistics
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METRICS = ("RMSE", "MAE", "MAPE")


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda data: torch.load(
                io.BytesIO(data),
                map_location=torch.device("cpu"),
                weights_only=False,
            )
        return super().find_class(module, name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze BatLiNet diagnostic outputs saved in predictions_seed_*.pkl files."
    )
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--experiment-name", default="batlinet_diagnostics")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "analysis" / "batlinet_diagnostics"),
    )
    return parser.parse_args()


def ensure_output_dirs(output_dir):
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    summary_dir = output_dir / "summary"
    for path in (tables_dir, figures_dir, summary_dir):
        path.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir, summary_dir


def load_pickle(path):
    with path.open("rb") as fin:
        return CPUUnpickler(fin).load()


def select_latest_prediction_files(workspace):
    workspace = Path(workspace)
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace does not exist: {workspace}")

    rows_by_seed = {}
    for file in sorted(workspace.glob("predictions_seed_*.pkl")):
        obj = load_pickle(file)
        seed = int(obj["seed"])
        previous = rows_by_seed.get(seed)
        if previous is None or file.stat().st_mtime > Path(previous["path"]).stat().st_mtime:
            rows_by_seed[seed] = {"path": str(file), "seed": seed}

    if not rows_by_seed:
        raise FileNotFoundError(f"No predictions_seed_*.pkl files found in {workspace}")
    return [Path(rows_by_seed[seed]["path"]) for seed in sorted(rows_by_seed)]


def inverse_label_tensor(dataset, tensor):
    label_transformation = getattr(dataset, "label_transformation", None)
    if label_transformation is not None:
        return label_transformation.inverse_transform(tensor)
    return tensor


def metric_value(target, prediction, metric):
    if metric == "RMSE":
        score = torch.mean((target - prediction) ** 2) ** 0.5
    elif metric == "MAE":
        score = torch.mean((target - prediction).abs())
    elif metric == "MAPE":
        score = torch.abs((target - prediction) / target).mean()
    else:
        raise ValueError(metric)
    return float(score)


def per_sample_absolute_error(target, prediction):
    return (prediction - target).abs()


def flatten_feature(x):
    return x.reshape(x.size(0), -1)


def safe_corr(x, y):
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def mean_or_nan(values):
    valid = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not valid:
        return float("nan")
    return float(sum(valid) / len(valid))


def summarize_metric_rows(rows, value_key):
    summary = {}
    for metric in METRICS:
        metric_rows = [row for row in rows if row["metric"] == metric]
        values = [row[value_key] for row in metric_rows]
        summary[metric] = {
            "count": len(values),
            "mean": sum(values) / len(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "best_seed": min(metric_rows, key=lambda row: row[value_key])["seed"],
            "best_value": min(values),
            "worst_seed": max(metric_rows, key=lambda row: row[value_key])["seed"],
            "worst_value": max(values),
        }
    return summary


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, text):
    with path.open("w", encoding="utf-8-sig") as fout:
        fout.write(text)


def analyze_prediction_file(path):
    obj = load_pickle(path)
    diagnostics = obj.get("diagnostics")
    if diagnostics is None:
        raise KeyError(
            f"Missing 'diagnostics' in {path}. Please rerun evaluation with the updated pipeline."
        )

    dataset = obj["data"].to("cpu")
    target = inverse_label_tensor(dataset, dataset.test_data.label).view(-1).cpu()
    final_prediction = inverse_label_tensor(dataset, obj["prediction"]).view(-1).cpu()
    y_ori = inverse_label_tensor(dataset, diagnostics["y_ori"]).view(-1).cpu()
    y_sup_agg = inverse_label_tensor(dataset, diagnostics["y_sup_agg"]).view(-1).cpu()
    y_sup = inverse_label_tensor(dataset, diagnostics["y_sup"]).cpu()
    support_index = diagnostics["support_index"].long().cpu()
    support_weight = diagnostics.get("support_weight")
    if support_weight is not None:
        support_weight = support_weight.cpu()

    prediction_rows = []
    for branch_name, prediction in (
        ("final_prediction", final_prediction),
        ("self_branch", y_ori),
        ("support_branch_aggregated", y_sup_agg),
    ):
        for metric in METRICS:
            prediction_rows.append(
                {
                    "seed": int(obj["seed"]),
                    "branch": branch_name,
                    "metric": metric,
                    "value": metric_value(target, prediction, metric),
                }
            )

    abs_error_pointwise = (y_sup - target.view(-1, 1)).abs()
    sample_mean_abs_error = abs_error_pointwise.mean(1)
    sample_best_abs_error = abs_error_pointwise.min(1).values
    sample_worst_abs_error = abs_error_pointwise.max(1).values
    sample_median_abs_error = abs_error_pointwise.median(1).values
    agg_abs_error = per_sample_absolute_error(target, y_sup_agg)
    final_abs_error = per_sample_absolute_error(target, final_prediction)
    ori_abs_error = per_sample_absolute_error(target, y_ori)

    single_support_rows = [
        {
            "seed": int(obj["seed"]),
            "num_test_samples": int(target.numel()),
            "support_size": int(y_sup.size(1)),
            "mean_single_support_abs_error": float(sample_mean_abs_error.mean()),
            "best_single_support_abs_error": float(sample_best_abs_error.mean()),
            "median_single_support_abs_error": float(sample_median_abs_error.mean()),
            "worst_single_support_abs_error": float(sample_worst_abs_error.mean()),
            "aggregated_support_abs_error": float(agg_abs_error.mean()),
            "self_branch_abs_error": float(ori_abs_error.mean()),
            "final_prediction_abs_error": float(final_abs_error.mean()),
            "share_best_single_beats_aggregated": float(
                (sample_best_abs_error < agg_abs_error).float().mean()
            ),
            "share_aggregated_beats_mean_single": float(
                (agg_abs_error < sample_mean_abs_error).float().mean()
            ),
            "share_final_beats_self_branch": float(
                (final_abs_error < ori_abs_error).float().mean()
            ),
        }
    ]

    train_feature = flatten_feature(dataset.train_data.feature.cpu())
    test_feature = flatten_feature(dataset.test_data.feature.cpu())
    selected_train_feature = train_feature[support_index]
    similarity_distance = torch.norm(
        test_feature.unsqueeze(1) - selected_train_feature,
        dim=-1,
    )

    weight_rows = []
    if support_weight is not None:
        top_k = max(1, support_weight.size(1) // 4)
        top_idx = torch.topk(support_weight, k=top_k, dim=1).indices
        bottom_idx = torch.topk(support_weight, k=top_k, dim=1, largest=False).indices

        top_error = abs_error_pointwise.gather(1, top_idx).mean(1)
        bottom_error = abs_error_pointwise.gather(1, bottom_idx).mean(1)
        top_distance = similarity_distance.gather(1, top_idx).mean(1)
        bottom_distance = similarity_distance.gather(1, bottom_idx).mean(1)

        weight_rows.append(
            {
                "seed": int(obj["seed"]),
                "support_size": int(support_weight.size(1)),
                "weight_error_corr": safe_corr(
                    support_weight.reshape(-1).numpy(),
                    (-abs_error_pointwise).reshape(-1).numpy(),
                ),
                "weight_similarity_corr": safe_corr(
                    support_weight.reshape(-1).numpy(),
                    (-similarity_distance).reshape(-1).numpy(),
                ),
                "top_weight_abs_error": float(top_error.mean()),
                "bottom_weight_abs_error": float(bottom_error.mean()),
                "top_weight_distance": float(top_distance.mean()),
                "bottom_weight_distance": float(bottom_distance.mean()),
                "share_top_weight_beats_bottom_weight_error": float(
                    (top_error < bottom_error).float().mean()
                ),
                "share_top_weight_is_closer": float(
                    (top_distance < bottom_distance).float().mean()
                ),
            }
        )

    return {
        "seed": int(obj["seed"]),
        "path": str(path),
        "prediction_rows": prediction_rows,
        "single_support_rows": single_support_rows,
        "weight_rows": weight_rows,
    }


def build_prediction_summary_rows(prediction_rows):
    summary = summarize_metric_rows(prediction_rows, "value")
    rows = []
    for metric in METRICS:
        item = summary[metric]
        rows.append(
            {
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


def build_summary_text(args, prediction_rows, single_support_rows, weight_rows):
    lines = [
        f"Experiment: {args.experiment_name}",
        f"Workspace: {args.workspace}",
        "",
        "1. 自身分支 vs 参考分支聚合 vs 最终融合",
    ]

    for branch in ("self_branch", "support_branch_aggregated", "final_prediction"):
        rows = [row for row in prediction_rows if row["branch"] == branch]
        lines.append(f"- {branch}:")
        for metric in METRICS:
            values = [row["value"] for row in rows if row["metric"] == metric]
            lines.append(
                f"  {metric} mean={sum(values) / len(values):.4f}, "
                f"std={statistics.pstdev(values) if len(values) > 1 else 0.0:.4f}"
            )

    single_mean = single_support_rows
    lines.extend(
        [
            "",
            "2. 单个参考电池预测误差",
            (
                f"- mean single-support abs error: "
                f"{mean_or_nan([row['mean_single_support_abs_error'] for row in single_mean]):.4f}"
            ),
            (
                f"- best single-support abs error: "
                f"{mean_or_nan([row['best_single_support_abs_error'] for row in single_mean]):.4f}"
            ),
            (
                f"- aggregated support abs error: "
                f"{mean_or_nan([row['aggregated_support_abs_error'] for row in single_mean]):.4f}"
            ),
            (
                f"- share(best single beats aggregated): "
                f"{mean_or_nan([row['share_best_single_beats_aggregated'] for row in single_mean]):.4f}"
            ),
            (
                f"- share(aggregated beats mean single): "
                f"{mean_or_nan([row['share_aggregated_beats_mean_single'] for row in single_mean]):.4f}"
            ),
        ]
    )

    lines.append("")
    lines.append("3. 学到的参考权重是否合理")
    if weight_rows:
        lines.extend(
            [
                (
                    f"- mean corr(weight, -abs_error): "
                    f"{mean_or_nan([row['weight_error_corr'] for row in weight_rows]):.4f}"
                ),
                (
                    f"- mean corr(weight, -distance): "
                    f"{mean_or_nan([row['weight_similarity_corr'] for row in weight_rows]):.4f}"
                ),
                (
                    f"- top-weight abs error: "
                    f"{mean_or_nan([row['top_weight_abs_error'] for row in weight_rows]):.4f}"
                ),
                (
                    f"- bottom-weight abs error: "
                    f"{mean_or_nan([row['bottom_weight_abs_error'] for row in weight_rows]):.4f}"
                ),
                (
                    f"- share(top-weight beats bottom-weight error): "
                    f"{mean_or_nan([row['share_top_weight_beats_bottom_weight_error'] for row in weight_rows]):.4f}"
                ),
                (
                    f"- share(top-weight is closer): "
                    f"{mean_or_nan([row['share_top_weight_is_closer'] for row in weight_rows]):.4f}"
                ),
            ]
        )
    else:
        lines.append("- 当前 workspace 没有可用的 learned weights；support_weight 全部为空。")

    return "\n".join(lines) + "\n"


def plot_branch_metrics(prediction_rows, figures_dir, experiment_name):
    import matplotlib.pyplot as plt

    branches = ["self_branch", "support_branch_aggregated", "final_prediction"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, metric in zip(axes, METRICS):
        values = []
        for branch in branches:
            branch_values = [
                row["value"]
                for row in prediction_rows
                if row["branch"] == branch and row["metric"] == metric
            ]
            values.append(sum(branch_values) / len(branch_values))
        bars = axis.bar(branches, values, width=0.6)
        ymax = max(values)
        axis.set_title(metric)
        axis.set_ylabel("Score")
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", linestyle="--", alpha=0.4)
        axis.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + ymax * 0.03,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle(f"{experiment_name} Branch Metric Means")
    fig.tight_layout()
    fig.savefig(figures_dir / "branch_metric_means.png", dpi=200)
    plt.close(fig)


def plot_single_support_analysis(single_support_rows, figures_dir, experiment_name):
    import matplotlib.pyplot as plt

    seeds = [row["seed"] for row in single_support_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))

    axes[0].plot(
        seeds,
        [row["best_single_support_abs_error"] for row in single_support_rows],
        marker="o",
        label="best_single",
    )
    axes[0].plot(
        seeds,
        [row["mean_single_support_abs_error"] for row in single_support_rows],
        marker="o",
        label="mean_single",
    )
    axes[0].plot(
        seeds,
        [row["aggregated_support_abs_error"] for row in single_support_rows],
        marker="o",
        label="aggregated_support",
    )
    axes[0].set_title("Absolute Error by Seed")
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Absolute Error")
    axes[0].grid(linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(
        seeds,
        [row["share_best_single_beats_aggregated"] for row in single_support_rows],
        marker="o",
        label="best_single_beats_agg",
    )
    axes[1].plot(
        seeds,
        [row["share_aggregated_beats_mean_single"] for row in single_support_rows],
        marker="o",
        label="agg_beats_mean_single",
    )
    axes[1].plot(
        seeds,
        [row["share_final_beats_self_branch"] for row in single_support_rows],
        marker="o",
        label="final_beats_self",
    )
    axes[1].set_title("Share by Seed")
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Share")
    axes[1].set_ylim(0, 1)
    axes[1].grid(linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.suptitle(f"{experiment_name} Single Support Diagnostics")
    fig.tight_layout()
    fig.savefig(figures_dir / "single_support_diagnostics.png", dpi=200)
    plt.close(fig)


def plot_weight_analysis(weight_rows, figures_dir, experiment_name):
    if not weight_rows:
        return

    import matplotlib.pyplot as plt

    seeds = [row["seed"] for row in weight_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))

    axes[0].plot(
        seeds,
        [row["top_weight_abs_error"] for row in weight_rows],
        marker="o",
        label="top_weight_error",
    )
    axes[0].plot(
        seeds,
        [row["bottom_weight_abs_error"] for row in weight_rows],
        marker="o",
        label="bottom_weight_error",
    )
    axes[0].set_title("Top vs Bottom Weight Error")
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Absolute Error")
    axes[0].grid(linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(
        seeds,
        [row["weight_error_corr"] for row in weight_rows],
        marker="o",
        label="corr(weight,-error)",
    )
    axes[1].plot(
        seeds,
        [row["weight_similarity_corr"] for row in weight_rows],
        marker="o",
        label="corr(weight,-distance)",
    )
    axes[1].set_title("Weight Correlations")
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Correlation")
    axes[1].set_ylim(-1, 1)
    axes[1].grid(linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.suptitle(f"{experiment_name} Weight Diagnostics")
    fig.tight_layout()
    fig.savefig(figures_dir / "weight_diagnostics.png", dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    tables_dir, figures_dir, summary_dir = ensure_output_dirs(output_dir)

    prediction_rows = []
    single_support_rows = []
    weight_rows = []

    for file in select_latest_prediction_files(args.workspace):
        result = analyze_prediction_file(file)
        prediction_rows.extend(result["prediction_rows"])
        single_support_rows.extend(result["single_support_rows"])
        weight_rows.extend(result["weight_rows"])

    branch_summary_rows = []
    for branch in ("self_branch", "support_branch_aggregated", "final_prediction"):
        branch_rows = [row for row in prediction_rows if row["branch"] == branch]
        for row in build_prediction_summary_rows(branch_rows):
            row["branch"] = branch
            branch_summary_rows.append(row)

    write_csv(
        tables_dir / "branch_metrics_by_seed.csv",
        ["seed", "branch", "metric", "value"],
        prediction_rows,
    )
    write_csv(
        tables_dir / "branch_metric_summary.csv",
        ["branch", "metric", "count", "mean", "std", "best_seed", "best_value", "worst_seed", "worst_value"],
        branch_summary_rows,
    )
    write_csv(
        tables_dir / "single_support_summary_by_seed.csv",
        [
            "seed",
            "num_test_samples",
            "support_size",
            "mean_single_support_abs_error",
            "best_single_support_abs_error",
            "median_single_support_abs_error",
            "worst_single_support_abs_error",
            "aggregated_support_abs_error",
            "self_branch_abs_error",
            "final_prediction_abs_error",
            "share_best_single_beats_aggregated",
            "share_aggregated_beats_mean_single",
            "share_final_beats_self_branch",
        ],
        single_support_rows,
    )
    write_csv(
        tables_dir / "weight_analysis_by_seed.csv",
        [
            "seed",
            "support_size",
            "weight_error_corr",
            "weight_similarity_corr",
            "top_weight_abs_error",
            "bottom_weight_abs_error",
            "top_weight_distance",
            "bottom_weight_distance",
            "share_top_weight_beats_bottom_weight_error",
            "share_top_weight_is_closer",
        ],
        weight_rows,
    )
    write_text(
        summary_dir / "diagnostic_summary.txt",
        build_summary_text(args, prediction_rows, single_support_rows, weight_rows),
    )

    plot_branch_metrics(prediction_rows, figures_dir, args.experiment_name)
    plot_single_support_analysis(single_support_rows, figures_dir, args.experiment_name)
    plot_weight_analysis(weight_rows, figures_dir, args.experiment_name)

    print(f"Saved outputs to: {output_dir}")
    print(f"Seeds analyzed: {len(single_support_rows)}")


if __name__ == "__main__":
    main()
