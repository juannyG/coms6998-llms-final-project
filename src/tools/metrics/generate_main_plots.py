"""
For use with single-tp-dpp-pp CSV results file.
"""

import argparse

import pandas as pd
import matplotlib.pyplot as plt

PLOT_CHOICE_KWARGS_MAP = {
    "throughput": {
        "metric": "throughput_tokens_sec",
        "ylabel": "Throughput (tokens/sec)",
        "title": "Throughput vs GPU Count",
    },
    "throughput_efficiency": {
        "metric": "throughput_efficiency_percent",
        "ylabel": "Efficiency (%)",
        "title": "Throughput Efficiency",
    },
    "relative_runtime_overhead": {
        "metric": "relative_runtime_overhead_percent",
        "ylabel": "Runtime Overhead (%)",
        "title": "Relative Runtime Overhead",
    },
    "peak_gpu": {
        "metric": "peak_gpu_mem_mb",
        "ylabel": "Peak GPU Memory (MB)",
        "title": "Peak Memory Usage",
    },
}

MODEL_SIZE_CHOICES = ["10m", "100m", "300m", "500m", "1b"]


def plot_metric_vs_gpus(df, model_size, metric, ylabel, title):
    subset = df[df["model_size"] == model_size]

    plt.figure(figsize=(6, 4))
    for strategy in subset["strategy"].unique():
        s = subset[subset["strategy"] == strategy]
        plt.plot(s["num_devices"], s[metric], marker="o", label=strategy)

    plt.xlabel("Number of GPUs")
    plt.ylabel(ylabel)
    plt.title(f"{title} — {model_size}")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots for a CSV generated via `python summary.py all ...`"
    )
    parser.add_argument("source", type=str, help="Source CSV for plots.")
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=list(PLOT_CHOICE_KWARGS_MAP.keys()),
        help="The type of plot you want produced",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.source)
    if args.plot_type:
        for model_size in MODEL_SIZE_CHOICES:
            plot_metric_vs_gpus(df, model_size, **PLOT_CHOICE_KWARGS_MAP[args.plot_type])
    else:
        for _, kwargs in PLOT_CHOICE_KWARGS_MAP.items():
            for model_size in MODEL_SIZE_CHOICES:
                plot_metric_vs_gpus(df, model_size, **kwargs)
