"""
For use with pp-microbatch-side-quest CSV results file.

python tools/metrics/generate_pp_side_quest_plot.py ../results/pp-microbatch-side-quest.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOTS_PATH = Path(__file__).parent.parent.parent.parent.joinpath(
    "results/plots/pp-microbatch-side-quest"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots for a CSV generated via `python summary.py all ...`"
    )
    parser.add_argument("source", type=str, help="Source CSV for plots.")
    args = parser.parse_args()

    df = pd.read_csv(args.source)
    model_sizes = ["10m", "100m", "300m"]
    for n_gpus in [2, 4]:
        by_n_gpus = df[df["# of GPUs"] == n_gpus]
        rec = by_n_gpus[by_n_gpus["# of microbatches"] == "Recommended"]
        rec_vals = [
            rec[rec["model size"] == model_size]["throughput"].values[0]
            for model_size in model_sizes
        ]
        worst = by_n_gpus[by_n_gpus["# of microbatches"] == "worst case"]
        worst_vals = [
            worst[worst["model size"] == model_size]["throughput"].values[0]
            for model_size in model_sizes
        ]

        width = 0.25
        fig, ax = plt.subplots()
        x = np.arange(len(model_sizes))
        ax.bar(x - width / 2, rec_vals, width, label="Recommended Microbatches")
        ax.bar(x + width / 2, worst_vals, width, label="Worst Case Microbatches")
        ax.grid()
        ax.set_xticks(x, model_sizes)
        ax.set(
            xlabel="Model Size",
            ylabel="Throughput (tokens/sec)",
            title=f"{n_gpus} GPU Pipeline Parallelism Recommended vs Worst Case Microbatches",
        )
        ax.legend()
        fig.savefig(f"{PLOTS_PATH}/{n_gpus}_rec_vs_worst_microbatches.png")
