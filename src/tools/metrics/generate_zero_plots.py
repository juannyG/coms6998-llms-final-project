"""
For use with simple-single-and-zero CSV results file.

python tools/metrics/generate_zero_plots.py ../results/simple-single-and-zero-experiment-results.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PLOTS_PATH = Path(__file__).parent.parent.parent.parent.joinpath("results/plots/zero")


PLOT_CHOICE_KWARGS_MAP = {
    "peak_mem_savings": {
        "metric": "peak_gpu_mem_mb",
        "ylabel": "Peak GPU Memory (MB)",
        "title": "Peak Memory Savings vs Zero Stage",
    },
    "throughput_penalty": {
        "metric": "throughput_tokens_sec",
        "ylabel": "Throughput (tokens/sec)",
        "title": "Throughput Penalty vs ZeRO Stage",
    },
}

MODEL_SIZE_CHOICES = ["zero-10m", "zero-100m", "zero-300m", "zero-500m", "zero-1b"]

LABEL_TO_STAGE_MAPPING = {
    "simple_single_gpu": "Baseline",
    "simple_zero_stage1": "Stage 1",
    "simple_zero_stage2": "Stage 2",
    "simple_zero_stage3": "Stage 3",
    "simple_zero_stage3_offload": "Stage 3 Offload",
}


def plot_metric_vs_stages(df, model_size, plot_type, metric, ylabel, title):
    subset = df[df["model_size"] == model_size].copy()
    subset["stage"] = subset["strategy"].map(LABEL_TO_STAGE_MAPPING)

    fig, ax = plt.subplots()
    for d in sorted(subset["num_devices"].unique()):
        s = subset[subset["num_devices"] == d]
        ax.plot(s["stage"], s[metric], marker="o", label=f"{d} GPU")

    ax.grid()
    ax.set(xlabel="ZeRO Stage", ylabel=ylabel, title=f"{title} — {model_size}")
    ax.legend()
    fig.savefig(f"{PLOTS_PATH}/{plot_type}/{model_size}_{plot_type}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ZeRO specific plots for a CSV generated via `python summary.py all ...`"
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
            plot_metric_vs_stages(
                df, model_size, args.plot_type, **PLOT_CHOICE_KWARGS_MAP[args.plot_type]
            )
    else:
        for plot_type, kwargs in PLOT_CHOICE_KWARGS_MAP.items():
            for model_size in MODEL_SIZE_CHOICES:
                plot_metric_vs_stages(df, model_size, plot_type, **kwargs)
