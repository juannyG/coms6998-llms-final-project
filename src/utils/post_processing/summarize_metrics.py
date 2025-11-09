"""
This is a simple script meant to take a log file or an entire directory and produce
summaries of different type of measurements from our experiments.

Usage:
    python post_processing/summarize_metrics.py training --dir ../logs/

    python post_processing/summarize_metrics.py profiler --files ../logs/run_single_gpu_10m*
"""

import argparse
import json
import glob
from pathlib import Path

from tabulate import tabulate

from experiments import single_gpu, torch_ddp, tensor_parallel
from utils.post_processing.metrics_dataclasses import TrainingResults, ProfilerSummary


TRAINING_RESULTS_METRIC_TYPE = "training"
PROFILER_METRICS_TYPE = "profiler"
METRICS_MESSAGE_MAP = {
    # TODO: The values feels fragile - we can probably tighten up the structural connection with the JSONL log files...
    TRAINING_RESULTS_METRIC_TYPE: "Training results",
    PROFILER_METRICS_TYPE: "Profiler metrics",
}


"""
This provides a mapping of what profiler labels should be included the in the profiler summary
based on the type of experiment. For example: ddp_communication is not recorded nor does it make 
sense to include in the `single_gpu` experiment summary
"""
EXPERIMENT_PROFILER_OPERATION_LABELS = {
    "single_gpu": single_gpu.EXPERIMENT_PROFILER_LABELS,
    "ddp": torch_ddp.EXPERIMENT_PROFILER_LABELS,
    "tensor_parallel": tensor_parallel.EXPERIMENT_PROFILER_LABELS,
}


def format_training_results(training_results):
    table = [
        ["Avg Tokens/sec", f"{training_results.avg_tokens_per_s:,.0f}"],
        ["Avg Samples/sec", f"{training_results.avg_samples_per_s:,.1f}"],
        ["Avg Loss", f"{training_results.avg_loss:.4f}"],
        ["Total Tokens", f"{training_results.total_tokens:,}"],
        ["Total Time", f"{training_results.total_time_s:.2f} sec"],
        ["Peak GPU Mem", f"{training_results.peak_gpu_mem_mb:.1f} MB"],
        ["GPU Utilization", f"{training_results.gpu_util_percent}%"],
    ]
    return tabulate(table, headers=["Metric", "Value"], tablefmt="github")


def format_profiler_summary(profiler_summary):
    table = [
        [
            op_label,
            profiler_op_summary.calls,
            f"{profiler_op_summary.cpu_time_ms:.2f}",
            f"{profiler_op_summary.gpu_time_ms:.2f}",
            f"{profiler_op_summary.cpu_mem_mb:.2f}",
            f"{profiler_op_summary.gpu_mem_mb:.2f}",
        ]
        for op_label, profiler_op_summary in profiler_summary.operations.items()
    ]
    return tabulate(
        table,
        headers=[
            "Operation",
            "Calls",
            "CPU Time (ms)",
            "GPU Time (ms)",
            "CPU Memory (MB)",
            "GPU Memory (MB)",
        ],
        tablefmt="github",
    )


def get_metrics(file_path, metric_msg):
    with open(file_path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("message") == metric_msg:
                    return obj
            except json.JSONDecodeError:
                print("failed to load json")
                continue  # skip malformed lines
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Provide per-device metrics summary in a more concise and readable manner."
    )
    parser.add_argument(
        "metric_type",
        type=str,
        choices=list(METRICS_MESSAGE_MAP.keys()),
        help="The type of metric to extract and summarize.",
    )
    parser.add_argument("--files", nargs="*", default=[], help="List of files")
    parser.add_argument(
        "--dir", type=str, required=False, help="Directory containing rank JSONL files."
    )
    args = parser.parse_args()

    if not args.files and not args.dir:
        print("ERROR: You must specify either --files or --dir")
        parser.print_help()
        exit(1)

    # Load the log file(s)
    all_files = []
    if args.dir:
        all_files.extend(sorted(Path(args.dir).rglob("*.log")))
    for f in args.files:
        all_files.extend(sorted(glob.glob(f"{f}")))
    if not all_files:
        raise FileNotFoundError(
            f"No JSONL files found in files: {args.files} / dir: {args.dir}"
        )

    # Print summary tables for experiment type
    for fpath in all_files:
        run_path = Path(fpath).parent  # logs/torch_ddp/10m/1762628113/
        strategy = run_path.parts[-3]  # e.g., torch_ddp
        model_size = run_path.parts[-2]  # e.g., 10m
        run_id = run_path.parts[-1]  # e.g., 1762628113
        device_id = Path(fpath).stem  # e.g., cuda_1
        experiment = f"{strategy}/{model_size}/{run_id}/{device_id}"

        metrics = get_metrics(fpath, METRICS_MESSAGE_MAP[args.metric_type])
        if not metrics:
            print(f"No metrics found for {fpath} of type {args.metric_type}")
            continue

        print(f"\n=== Results for experiment: {experiment} ===")
        if args.metric_type == TRAINING_RESULTS_METRIC_TYPE:
            training_results = TrainingResults.from_dict(metrics)
            print(format_training_results(training_results))
        elif args.metric_type == PROFILER_METRICS_TYPE:
            operation_labels = []
            if "single_gpu" in experiment:
                operation_labels = EXPERIMENT_PROFILER_OPERATION_LABELS["single_gpu"]
            elif "ddp" in experiment:
                operation_labels = EXPERIMENT_PROFILER_OPERATION_LABELS["ddp"]
            elif "tensor_parallel" in experiment:
                operation_labels = EXPERIMENT_PROFILER_OPERATION_LABELS[
                    "tensor_parallel"
                ]
            profiler_summary = ProfilerSummary()
            profiler_summary.update_from_profiler_metrics(metrics["profiler_metrics"], operation_labels)
            print(format_profiler_summary(profiler_summary))


if __name__ == "__main__":
    main()
