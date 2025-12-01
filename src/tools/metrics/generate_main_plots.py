"""
For use with single-tp-dpp-pp CSV results file.

python tools/metrics/generate_main_plots.py ../results/single-tp-dpp-pp-experiment-results.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PLOTS_PATH = Path(__file__).parent.parent.parent.parent.joinpath(
    "results/plots/tp-pp-ddp"
)


PLOT_CHOICE_KWARGS_MAP = {
    "throughput": {
        "metric": "throughput_tokens_sec",
        "ylabel": "Throughput (tokens/sec)",
        "title": "Throughput vs GPU Count",
    },
    "throughput_eff": {
        "metric": "throughput_efficiency_percent",
        "ylabel": "Efficiency (%)",
        "title": "Throughput Efficiency",
    },
    "relative_runtime_overhead": {
        "metric": "relative_runtime_overhead_percent",
        "ylabel": "Runtime Overhead (%)",
        "title": "Relative Runtime Overhead",
    },
    "peak_mem": {
        "metric": "peak_gpu_mem_mb",
        "ylabel": "Peak GPU Memory (MB)",
        "title": "Peak Memory Usage",
    },
}

MODEL_SIZE_CHOICES = ["10m", "100m", "300m", "500m", "1b"]


def plot_metric_vs_gpus(df, model_size, plot_type, metric, ylabel, title):
    subset = df[df["model_size"] == model_size]
    fig, ax = plt.subplots()

    for strategy in subset["strategy"].unique():
        s = subset[subset["strategy"] == strategy]
        ax.plot(s["num_devices"], s[metric], marker="o", label=strategy)

    ax.grid()
    ax.set(xlabel="Number of GPUs", ylabel=ylabel, title=f"{title} — {model_size}")
    ax.legend()
    fig.savefig(f"{PLOTS_PATH}/{plot_type}/{model_size}_{plot_type}.png")


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
            plot_metric_vs_gpus(
                df, model_size, args.plot_type, **PLOT_CHOICE_KWARGS_MAP[args.plot_type]
            )
    else:
        for plot_type, kwargs in PLOT_CHOICE_KWARGS_MAP.items():
            for model_size in MODEL_SIZE_CHOICES:
                plot_metric_vs_gpus(df, model_size, plot_type, **kwargs)
