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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METRICS = ("RMSE", "MAE", "MAPE")


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two BatLiNet experiment workspaces from prediction pkl files."
    )
    parser.add_argument("--baseline-workspace", required=True)
    parser.add_argument("--candidate-workspace", required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--experiment-name", default="experiment")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "analysis" / "experiment_comparison"),
    )
    return parser.parse_args()


def ensure_output_dirs(output_dir):
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    summary_dir = output_dir / "summary"
    for path in (tables_dir, figures_dir, summary_dir):
        path.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir, summary_dir


def load_prediction_file(path):
    with path.open("rb") as fin:
        obj = CPUUnpickler(fin).load()

    row = {"seed": int(obj["seed"]), "path": str(path)}
    for metric in METRICS:
        row[metric] = float(obj["scores"][metric])
    return row


def collect_workspace_results(workspace):
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace does not exist: {workspace}")

    rows_by_seed = {}
    for file in sorted(workspace.glob("predictions_seed_*.pkl")):
        row = load_prediction_file(file)
        seed = row["seed"]
        previous = rows_by_seed.get(seed)
        if previous is None or file.stat().st_mtime > Path(previous["path"]).stat().st_mtime:
            rows_by_seed[seed] = row

    if not rows_by_seed:
        raise FileNotFoundError(f"No predictions_seed_*.pkl files found in {workspace}")
    return [rows_by_seed[seed] for seed in sorted(rows_by_seed)]


def summarize(rows):
    summary = {}
    for metric in METRICS:
        values = [row[metric] for row in rows]
        best = min(rows, key=lambda row: row[metric])
        worst = max(rows, key=lambda row: row[metric])
        summary[metric] = {
            "count": len(values),
            "mean": sum(values) / len(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "best_seed": best["seed"],
            "best_value": best[metric],
            "worst_seed": worst["seed"],
            "worst_value": worst[metric],
        }
    return summary


def compare_by_seed(baseline_rows, candidate_rows):
    baseline_by_seed = {row["seed"]: row for row in baseline_rows}
    candidate_by_seed = {row["seed"]: row for row in candidate_rows}
    seeds = sorted(set(baseline_by_seed) | set(candidate_by_seed))
    rows = []
    missing = []

    for seed in seeds:
        baseline = baseline_by_seed.get(seed)
        candidate = candidate_by_seed.get(seed)
        if baseline is None:
            missing.append(f"baseline seed {seed}")
        if candidate is None:
            missing.append(f"candidate seed {seed}")

        row = {"seed": seed}
        for metric in METRICS:
            base_value = baseline[metric] if baseline else None
            cand_value = candidate[metric] if candidate else None
            row[f"baseline_{metric.lower()}"] = base_value
            row[f"candidate_{metric.lower()}"] = cand_value
            row[f"delta_{metric.lower()}"] = (
                None if base_value is None or cand_value is None else cand_value - base_value
            )
        rows.append(row)
    return rows, missing


def format_float(value):
    if value is None:
        return "NA"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "NA"
    return f"{value:.4f}"


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, text):
    with path.open("w", encoding="utf-8-sig") as fout:
        fout.write(text)


def summary_rows(name, summary):
    rows = []
    for metric in METRICS:
        item = summary[metric]
        rows.append(
            {
                "group": name,
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


def build_summary_text(args, baseline_summary, candidate_summary, missing):
    lines = [
        f"Experiment comparison: {args.experiment_name}",
        "",
        f"Baseline: {args.baseline_name}",
        f"Baseline workspace: {args.baseline_workspace}",
        f"Candidate: {args.candidate_name}",
        f"Candidate workspace: {args.candidate_workspace}",
        "",
        "Metric means:",
    ]
    for metric in METRICS:
        base_mean = baseline_summary[metric]["mean"]
        cand_mean = candidate_summary[metric]["mean"]
        delta = cand_mean - base_mean
        rel = delta / base_mean * 100 if base_mean else 0.0
        direction = "better" if delta < 0 else "worse" if delta > 0 else "equal"
        lines.append(
            f"- {metric}: baseline {base_mean:.4f}, candidate {cand_mean:.4f}, "
            f"delta {delta:+.4f} ({rel:+.2f}%), {direction}"
        )

    lines.extend(["", "Seed coverage:"])
    if missing:
        lines.extend(f"- missing {item}" for item in missing)
    else:
        lines.append("- all compared seeds are available in both workspaces")
    return "\n".join(lines) + "\n"


def plot_metric_means(args, baseline_summary, candidate_summary, figures_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    for axis, metric in zip(axes, METRICS):
        values = [baseline_summary[metric]["mean"], candidate_summary[metric]["mean"]]
        bars = axis.bar([args.baseline_name, args.candidate_name], values, width=0.55)
        ymax = max(values)
        axis.set_title(metric)
        axis.set_ylabel("Score")
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
    fig.suptitle(f"{args.experiment_name} Mean Metrics")
    fig.tight_layout()
    fig.savefig(figures_dir / "metric_means_bar.png", dpi=200)
    plt.close(fig)


def plot_seed_lines(args, baseline_rows, candidate_rows, figures_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for axis, metric in zip(axes, METRICS):
        axis.plot(
            [row["seed"] for row in baseline_rows],
            [row[metric] for row in baseline_rows],
            marker="o",
            label=args.baseline_name,
        )
        axis.plot(
            [row["seed"] for row in candidate_rows],
            [row[metric] for row in candidate_rows],
            marker="o",
            label=args.candidate_name,
        )
        axis.set_title(metric)
        axis.set_xlabel("Seed")
        axis.set_ylabel("Score")
        axis.grid(linestyle="--", alpha=0.4)
    axes[0].legend()
    fig.suptitle(f"{args.experiment_name} Metrics Across Seeds")
    fig.tight_layout()
    fig.savefig(figures_dir / "metrics_by_seed_lines.png", dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    baseline_workspace = Path(args.baseline_workspace)
    candidate_workspace = Path(args.candidate_workspace)
    output_dir = Path(args.output_dir)
    tables_dir, figures_dir, summary_dir = ensure_output_dirs(output_dir)

    baseline_rows = collect_workspace_results(baseline_workspace)
    candidate_rows = collect_workspace_results(candidate_workspace)
    comparison_rows, missing = compare_by_seed(baseline_rows, candidate_rows)
    baseline_summary = summarize(baseline_rows)
    candidate_summary = summarize(candidate_rows)
    all_summary_rows = (
        summary_rows(args.baseline_name, baseline_summary)
        + summary_rows(args.candidate_name, candidate_summary)
    )

    write_csv(
        tables_dir / "seed_level_comparison.csv",
        [
            "seed",
            "baseline_rmse",
            "candidate_rmse",
            "delta_rmse",
            "baseline_mae",
            "candidate_mae",
            "delta_mae",
            "baseline_mape",
            "candidate_mape",
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
        all_summary_rows,
    )
    write_text(
        summary_dir / "comparison_summary.txt",
        build_summary_text(args, baseline_summary, candidate_summary, missing),
    )
    plot_metric_means(args, baseline_summary, candidate_summary, figures_dir)
    plot_seed_lines(args, baseline_rows, candidate_rows, figures_dir)

    print(f"Saved outputs to: {output_dir}")
    print(f"{args.baseline_name} seeds loaded: {len(baseline_rows)}")
    print(f"{args.candidate_name} seeds loaded: {len(candidate_rows)}")
    for metric in METRICS:
        delta = candidate_summary[metric]["mean"] - baseline_summary[metric]["mean"]
        print(
            f"{metric}: {args.baseline_name}={baseline_summary[metric]['mean']:.4f} "
            f"{args.candidate_name}={candidate_summary[metric]['mean']:.4f} "
            f"delta={delta:+.4f}"
        )


if __name__ == "__main__":
    main()
