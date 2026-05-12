import argparse
import csv
import io
import math
import pickle
import statistics
import sys
from pathlib import Path
from functools import partial

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


METRICS = ("RMSE", "MAE", "MAPE")


class Dataset:
    def __init__(self, feature=None, label=None):
        self.feature = feature
        self.label = label

    def to(self, device):
        if self.label is not None:
            self.label = self.label.to(device)
        if self.feature is not None:
            self.feature = self.feature.to(device)
        return self


class DataBundle:
    def __init__(self,
                 train_feature=None,
                 train_label=None,
                 test_feature=None,
                 test_label=None,
                 feature_transformation=None,
                 label_transformation=None):
        self.train_data = Dataset(train_feature, train_label)
        self.test_data = Dataset(test_feature, test_label)
        self.feature_transformation = feature_transformation
        self.label_transformation = label_transformation

    def to(self, device):
        self.train_data = self.train_data.to(device)
        self.test_data = self.test_data.to(device)
        if self.feature_transformation is not None:
            self.feature_transformation = self.feature_transformation.to(device)
        if self.label_transformation is not None:
            self.label_transformation = self.label_transformation.to(device)
        return self


class BaseDataTransformation:
    def fit(self, data):
        return None

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

    def to(self, device):
        return self


def _log_forward(base, x):
    return torch.log(x) / math.log(base)


class LogScaleDataTransformation(BaseDataTransformation):
    def __init__(self, base=None):
        self.base = base or math.e
        if base is None:
            self._func = torch.log
            self._inv_func = torch.exp
        else:
            self._func = partial(_log_forward, base)
            self._inv_func = partial(torch.pow, base)

    def transform(self, data):
        return self._func(data)

    def inverse_transform(self, data):
        return self._inv_func(data)


class ZScoreDataTransformation(BaseDataTransformation):
    def __init__(self):
        self._mean = None
        self._std = None

    def fit(self, data):
        self._mean = data.mean(0, keepdim=True)
        self._std = torch.clamp(data.std(0, keepdim=True), min=1e-8)

    def inverse_transform(self, data):
        return data * self._std + self._mean

    def to(self, device):
        if self._mean is not None:
            self._mean = self._mean.to(device)
        if self._std is not None:
            self._std = self._std.to(device)
        return self


class SequentialDataTransformation(BaseDataTransformation):
    def __init__(self, transformations=None):
        self.transformations = transformations or []

    def inverse_transform(self, data):
        for trans in self.transformations[::-1]:
            data = trans.inverse_transform(data)
        return data

    def to(self, device):
        self.transformations = [t.to(device) for t in self.transformations]
        return self


SAFE_SRC_CLASSES = {
    ("src.data.databundle", "DataBundle"): DataBundle,
    ("src.data.databundle", "Dataset"): Dataset,
    ("src.data.transformation.base", "BaseDataTransformation"): BaseDataTransformation,
    ("src.data.transformation.log_scale", "LogScaleDataTransformation"): LogScaleDataTransformation,
    ("src.data.transformation.sequential", "SequentialDataTransformation"): SequentialDataTransformation,
    ("src.data.transformation.z_score", "ZScoreDataTransformation"): ZScoreDataTransformation,
}


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
        if (module, name) in SAFE_SRC_CLASSES:
            return SAFE_SRC_CLASSES[(module, name)]
        if module.startswith("src."):
            return PlaceholderObject
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
    per_seed_figures_dir = figures_dir / "per_seed"
    summary_dir = output_dir / "summary"
    for path in (tables_dir, figures_dir, per_seed_figures_dir, summary_dir):
        path.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir, per_seed_figures_dir, summary_dir


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


def configure_matplotlib():
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "Noto Sans SC",
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


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
            "share_aggregated_beats_median_single": float(
                (agg_abs_error < sample_median_abs_error).float().mean()
            ),
            "share_aggregated_beats_self_branch": float(
                (agg_abs_error < ori_abs_error).float().mean()
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
    pointwise_rows = []
    self_point_rows = []
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

    support_weight_for_rows = support_weight
    if support_weight_for_rows is None:
        support_weight_for_rows = torch.full_like(y_sup, float("nan"))

    for sample_idx in range(target.numel()):
        self_point_rows.append(
            {
                "seed": int(obj["seed"]),
                "sample_index": sample_idx,
                "target_rul": float(target[sample_idx]),
                "self_prediction_rul": float(y_ori[sample_idx]),
                "self_abs_error": float(ori_abs_error[sample_idx]),
                "aggregated_support_prediction_rul": float(y_sup_agg[sample_idx]),
                "aggregated_support_abs_error": float(agg_abs_error[sample_idx]),
                "final_prediction_rul": float(final_prediction[sample_idx]),
                "final_abs_error": float(final_abs_error[sample_idx]),
            }
        )
        for support_rank in range(y_sup.size(1)):
            pointwise_rows.append(
                {
                    "seed": int(obj["seed"]),
                    "sample_index": sample_idx,
                    "support_rank": support_rank,
                    "support_index": int(support_index[sample_idx, support_rank]),
                    "target_rul": float(target[sample_idx]),
                    "support_prediction_rul": float(y_sup[sample_idx, support_rank]),
                    "support_abs_error": float(abs_error_pointwise[sample_idx, support_rank]),
                    "support_weight": float(support_weight_for_rows[sample_idx, support_rank]),
                    "support_distance": float(similarity_distance[sample_idx, support_rank]),
                    "self_prediction_rul": float(y_ori[sample_idx]),
                    "self_abs_error": float(ori_abs_error[sample_idx]),
                    "aggregated_support_prediction_rul": float(y_sup_agg[sample_idx]),
                    "aggregated_support_abs_error": float(agg_abs_error[sample_idx]),
                    "final_prediction_rul": float(final_prediction[sample_idx]),
                    "final_abs_error": float(final_abs_error[sample_idx]),
                }
            )

    return {
        "seed": int(obj["seed"]),
        "path": str(path),
        "prediction_rows": prediction_rows,
        "single_support_rows": single_support_rows,
        "weight_rows": weight_rows,
        "pointwise_rows": pointwise_rows,
        "self_point_rows": self_point_rows,
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
    branch_labels = {
        "self_branch": "自身分支",
        "support_branch_aggregated": "参考分支聚合",
        "final_prediction": "最终融合预测",
    }
    lines = [
        f"实验名称：{args.experiment_name}",
        f"结果目录：{args.workspace}",
        "",
        "1. 自身分支 vs 参考分支聚合 vs 最终融合",
    ]

    for branch in ("self_branch", "support_branch_aggregated", "final_prediction"):
        rows = [row for row in prediction_rows if row["branch"] == branch]
        lines.append(f"- {branch_labels[branch]}：")
        for metric in METRICS:
            values = [row["value"] for row in rows if row["metric"] == metric]
            lines.append(
                f"  {metric} 平均值={sum(values) / len(values):.4f}，"
                f"标准差={statistics.pstdev(values) if len(values) > 1 else 0.0:.4f}"
            )

    single_mean = single_support_rows
    lines.extend(
        [
            "",
            "2. 单个参考电池预测误差",
            (
                f"- 单个参考预测绝对误差的平均值（mean single-support abs error）："
                f"{mean_or_nan([row['mean_single_support_abs_error'] for row in single_mean]):.4f}"
            ),
            (
                f"- 单个参考中“最好那个参考”的绝对误差平均值（best single-support abs error）："
                f"{mean_or_nan([row['best_single_support_abs_error'] for row in single_mean]):.4f}"
            ),
            (
                f"- 参考分支聚合后的绝对误差平均值（aggregated support abs error）："
                f"{mean_or_nan([row['aggregated_support_abs_error'] for row in single_mean]):.4f}"
            ),
            (
                f"- 最佳单参考优于当前聚合结果的样本占比（share(best single beats aggregated)）："
                f"{mean_or_nan([row['share_best_single_beats_aggregated'] for row in single_mean]):.4f}"
            ),
            (
                f"- 当前聚合结果优于“单参考中位数水平”的样本占比（share(aggregated beats median single)）："
                f"{mean_or_nan([row['share_aggregated_beats_median_single'] for row in single_mean]):.4f}"
            ),
            (
                f"- 当前聚合结果优于自身分支的样本占比（share(aggregated beats self branch)）："
                f"{mean_or_nan([row['share_aggregated_beats_self_branch'] for row in single_mean]):.4f}"
            ),
        ]
    )

    lines.append("")
    lines.append("3. 学到的参考权重是否合理")
    if weight_rows:
        lines.extend(
            [
                (
                    f"- 权重与“负绝对误差”的平均相关性（mean corr(weight, -abs_error)）："
                    f"{mean_or_nan([row['weight_error_corr'] for row in weight_rows]):.4f}"
                ),
                (
                    f"- 权重与“负距离”的平均相关性（mean corr(weight, -distance)）："
                    f"{mean_or_nan([row['weight_similarity_corr'] for row in weight_rows]):.4f}"
                ),
                (
                    f"- 高权重参考组的平均绝对误差（top-weight abs error）："
                    f"{mean_or_nan([row['top_weight_abs_error'] for row in weight_rows]):.4f}"
                ),
                (
                    f"- 低权重参考组的平均绝对误差（bottom-weight abs error）："
                    f"{mean_or_nan([row['bottom_weight_abs_error'] for row in weight_rows]):.4f}"
                ),
                (
                    f"- 高权重参考组优于低权重参考组的样本占比（share(top-weight beats bottom-weight error)）："
                    f"{mean_or_nan([row['share_top_weight_beats_bottom_weight_error'] for row in weight_rows]):.4f}"
                ),
                (
                    f"- 高权重参考组比低权重参考组更接近目标电池的样本占比（share(top-weight is closer)）："
                    f"{mean_or_nan([row['share_top_weight_is_closer'] for row in weight_rows]):.4f}"
                ),
            ]
        )
    else:
        lines.append("- 当前 workspace 没有可用的 learned weights；support_weight 全部为空。")

    return "\n".join(lines) + "\n"


def plot_branch_metrics(prediction_rows, figures_dir, experiment_name):
    configure_matplotlib()
    import matplotlib.pyplot as plt

    branches = ["self_branch", "support_branch_aggregated", "final_prediction"]
    branch_labels = ["自身分支", "参考分支聚合", "最终融合预测"]
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
        bars = axis.bar(branch_labels, values, width=0.6)
        ymax = max(values)
        axis.set_title(metric)
        axis.set_ylabel("指标值")
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
    fig.suptitle(f"{experiment_name} 三个分支的平均指标")
    fig.tight_layout()
    fig.savefig(figures_dir / "branch_metric_means.png", dpi=200)
    plt.close(fig)


def plot_single_support_analysis(single_support_rows, figures_dir, experiment_name):
    configure_matplotlib()
    import matplotlib.pyplot as plt

    seeds = [row["seed"] for row in single_support_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))

    axes[0].plot(
        seeds,
        [row["best_single_support_abs_error"] for row in single_support_rows],
        marker="o",
        label="最好单参考误差",
    )
    axes[0].plot(
        seeds,
        [row["mean_single_support_abs_error"] for row in single_support_rows],
        marker="o",
        label="单参考平均误差",
    )
    axes[0].plot(
        seeds,
        [row["aggregated_support_abs_error"] for row in single_support_rows],
        marker="o",
        label="聚合后参考分支误差",
    )
    axes[0].set_title("不同 seed 的绝对误差")
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("绝对误差")
    axes[0].grid(linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(
        seeds,
        [row["share_best_single_beats_aggregated"] for row in single_support_rows],
        marker="o",
        label="最好单参考优于聚合结果",
    )
    axes[1].plot(
        seeds,
        [row["share_aggregated_beats_median_single"] for row in single_support_rows],
        marker="o",
        label="聚合结果优于单参考中位数",
    )
    axes[1].plot(
        seeds,
        [row["share_aggregated_beats_self_branch"] for row in single_support_rows],
        marker="o",
        label="聚合结果优于自身分支",
    )
    axes[1].plot(
        seeds,
        [row["share_final_beats_self_branch"] for row in single_support_rows],
        marker="o",
        label="最终融合优于自身分支",
    )
    axes[1].set_title("不同 seed 的样本占比")
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("占比")
    axes[1].set_ylim(0, 1)
    axes[1].grid(linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.suptitle(f"{experiment_name} 单参考诊断")
    fig.tight_layout()
    fig.savefig(figures_dir / "single_support_diagnostics.png", dpi=200)
    plt.close(fig)


def plot_weight_analysis(weight_rows, figures_dir, experiment_name):
    if not weight_rows:
        return

    configure_matplotlib()
    import matplotlib.pyplot as plt

    seeds = [row["seed"] for row in weight_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))

    axes[0].plot(
        seeds,
        [row["top_weight_abs_error"] for row in weight_rows],
        marker="o",
        label="高权重组误差",
    )
    axes[0].plot(
        seeds,
        [row["bottom_weight_abs_error"] for row in weight_rows],
        marker="o",
        label="低权重组误差",
    )
    axes[0].set_title("高低权重参考组误差对比")
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("绝对误差")
    axes[0].grid(linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(
        seeds,
        [row["weight_error_corr"] for row in weight_rows],
        marker="o",
        label="权重-误差相关性",
    )
    axes[1].plot(
        seeds,
        [row["weight_similarity_corr"] for row in weight_rows],
        marker="o",
        label="权重-距离相关性",
    )
    axes[1].set_title("权重相关性")
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("相关系数")
    axes[1].set_ylim(-1, 1)
    axes[1].grid(linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.suptitle(f"{experiment_name} 权重诊断")
    fig.tight_layout()
    fig.savefig(figures_dir / "weight_diagnostics.png", dpi=200)
    plt.close(fig)


def _select_representative_samples(support_rows, self_rows):
    by_sample_support = {}
    for row in support_rows:
        by_sample_support.setdefault(row["sample_index"], []).append(row)
    self_by_sample = {row["sample_index"]: row for row in self_rows}

    sample_stats = []
    for sample_index, rows in by_sample_support.items():
        rows_sorted = sorted(rows, key=lambda x: x["support_abs_error"])
        best_support_error = rows_sorted[0]["support_abs_error"]
        support_gap = (
            self_by_sample[sample_index]["aggregated_support_abs_error"]
            - best_support_error
        )
        sample_stats.append(
            {
                "sample_index": sample_index,
                "aggregated_error": self_by_sample[sample_index]["aggregated_support_abs_error"],
                "self_error": self_by_sample[sample_index]["self_abs_error"],
                "support_gap": support_gap,
            }
        )

    chosen = []
    if sample_stats:
        chosen.append(
            ("聚合效果最好", min(sample_stats, key=lambda x: x["aggregated_error"])["sample_index"])
        )
        chosen.append(
            ("聚合最没用好最好单参考", max(sample_stats, key=lambda x: x["support_gap"])["sample_index"])
        )
        chosen.append(
            ("自身分支误差最大", max(sample_stats, key=lambda x: x["self_error"])["sample_index"])
        )

    dedup = []
    seen = set()
    for label, sample_index in chosen:
        if sample_index not in seen:
            dedup.append((label, sample_index))
            seen.add(sample_index)
    return dedup


def plot_per_seed_support_scatter(pointwise_rows, self_point_rows, per_seed_figures_dir, experiment_name):
    configure_matplotlib()
    import matplotlib.pyplot as plt

    seeds = sorted({row["seed"] for row in pointwise_rows})
    for seed in seeds:
        support_rows = [row for row in pointwise_rows if row["seed"] == seed]
        self_rows = [row for row in self_point_rows if row["seed"] == seed]
        if not support_rows:
            continue

        representatives = _select_representative_samples(support_rows, self_rows)
        if not representatives:
            continue

        fig, axes = plt.subplots(len(representatives), 2, figsize=(14, 4.6 * len(representatives)))
        if len(representatives) == 1:
            axes = np.array([axes])

        for row_idx, (sample_label, sample_index) in enumerate(representatives):
            sample_support_rows = [
                row for row in support_rows if row["sample_index"] == sample_index
            ]
            sample_support_rows = sorted(sample_support_rows, key=lambda x: x["support_rank"])
            sample_self_row = next(row for row in self_rows if row["sample_index"] == sample_index)

            support_ranks = [row["support_rank"] for row in sample_support_rows]
            support_errors = [row["support_abs_error"] for row in sample_support_rows]
            support_weights = [row["support_weight"] for row in sample_support_rows]

            ax_left = axes[row_idx, 0]
            bars = ax_left.bar(
                support_ranks,
                support_errors,
                color=plt.cm.viridis(np.nan_to_num(support_weights, nan=0.0)),
                alpha=0.85,
            )
            ax_left.axhline(
                y=sample_self_row["self_abs_error"],
                color="#d55e00",
                linestyle="--",
                linewidth=1.8,
                label="自身分支误差",
            )
            ax_left.axhline(
                y=sample_self_row["aggregated_support_abs_error"],
                color="#0072b2",
                linestyle="--",
                linewidth=1.8,
                label="参考分支聚合误差",
            )
            ax_left.axhline(
                y=sample_self_row["final_abs_error"],
                color="#7b3294",
                linestyle="--",
                linewidth=1.8,
                label="最终融合误差",
            )
            ax_left.set_title(f"Seed {seed} 样本 {sample_index}：{sample_label}")
            ax_left.set_xlabel("参考电池序号（0-31）")
            ax_left.set_ylabel("单参考绝对误差")
            ax_left.grid(axis="y", linestyle="--", alpha=0.3)
            ax_left.legend(loc="upper right")

            ax_right = axes[row_idx, 1]
            ax_right.plot(
                support_ranks,
                support_weights,
                color="#009e73",
                marker="o",
                linewidth=1.8,
                label="参考权重",
            )
            ax_right.set_ylim(0, max(max(support_weights) * 1.15, 0.05))
            ax_right.set_xlabel("参考电池序号（0-31）")
            ax_right.set_ylabel("参考权重")
            ax_right.grid(axis="y", linestyle="--", alpha=0.3)

            ax_right_twin = ax_right.twinx()
            ax_right_twin.plot(
                support_ranks,
                support_errors,
                color="#444444",
                marker="x",
                linewidth=1.2,
                alpha=0.8,
                label="单参考绝对误差",
            )
            ax_right_twin.set_ylabel("单参考绝对误差")

            handles_left, labels_left = ax_right.get_legend_handles_labels()
            handles_right, labels_right = ax_right_twin.get_legend_handles_labels()
            ax_right.legend(handles_left + handles_right, labels_left + labels_right, loc="upper right")

        fig.suptitle(f"{experiment_name} Seed {seed} 代表样本的参考误差-权重图")
        fig.tight_layout()
        fig.savefig(per_seed_figures_dir / f"seed_{seed}_support_scatter.png", dpi=220)
        plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    tables_dir, figures_dir, per_seed_figures_dir, summary_dir = ensure_output_dirs(output_dir)

    prediction_rows = []
    single_support_rows = []
    weight_rows = []
    pointwise_rows = []
    self_point_rows = []

    for file in select_latest_prediction_files(args.workspace):
        result = analyze_prediction_file(file)
        prediction_rows.extend(result["prediction_rows"])
        single_support_rows.extend(result["single_support_rows"])
        weight_rows.extend(result["weight_rows"])
        pointwise_rows.extend(result["pointwise_rows"])
        self_point_rows.extend(result["self_point_rows"])

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
            "share_aggregated_beats_median_single",
            "share_aggregated_beats_self_branch",
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
    write_csv(
        tables_dir / "support_prediction_details_by_point.csv",
        [
            "seed",
            "sample_index",
            "support_rank",
            "support_index",
            "target_rul",
            "support_prediction_rul",
            "support_abs_error",
            "support_weight",
            "support_distance",
            "self_prediction_rul",
            "self_abs_error",
            "aggregated_support_prediction_rul",
            "aggregated_support_abs_error",
            "final_prediction_rul",
            "final_abs_error",
        ],
        pointwise_rows,
    )
    write_csv(
        tables_dir / "branch_prediction_details_by_sample.csv",
        [
            "seed",
            "sample_index",
            "target_rul",
            "self_prediction_rul",
            "self_abs_error",
            "aggregated_support_prediction_rul",
            "aggregated_support_abs_error",
            "final_prediction_rul",
            "final_abs_error",
        ],
        self_point_rows,
    )
    write_text(
        summary_dir / "diagnostic_summary.txt",
        build_summary_text(args, prediction_rows, single_support_rows, weight_rows),
    )

    plot_branch_metrics(prediction_rows, figures_dir, args.experiment_name)
    plot_single_support_analysis(single_support_rows, figures_dir, args.experiment_name)
    plot_weight_analysis(weight_rows, figures_dir, args.experiment_name)
    plot_per_seed_support_scatter(
        pointwise_rows, self_point_rows, per_seed_figures_dir, args.experiment_name)

    print(f"Saved outputs to: {output_dir}")
    print(f"Seeds analyzed: {len(single_support_rows)}")


if __name__ == "__main__":
    main()
