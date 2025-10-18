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
    # TODO: Move the label definitions into the experiments and pull them in here
    "single_gpu": [
        "model_forward",
        "model_loss",
        "model_backward",
        "model_optimizer_step",
    ],
    "ddp": [
        "model_forward",
        "model_loss",
        "model_backward",
        "ddp_communication",
        "model_optimizer_step",
    ],
}


def format_training_results(result):
    # TODO: Eventually support diff formats (i.e. CSV, etc)
    table = [
        ["Avg Tokens/sec", f"{result['avg_tokens_per_s']:,.0f}"],
        ["Avg Samples/sec", f"{result['avg_samples_per_s']:,.1f}"],
        ["Avg Loss", f"{result['avg_loss']:.4f}"],
        ["Total Tokens", f"{result['total_tokens']:,}"],
        ["Total Time", f"{result['total_time_s']:.2f} sec"],
        ["Peak GPU Mem", f"{result['peak_gpu_mem_mb']:.1f} MB"],
        ["GPU Utilization", f"{result['gpu_util_percent']}%"],
    ]
    return tabulate(table, headers=["Metric", "Value"], tablefmt="github")


def format_profile_metrics(results, operation_labels):
    operation_metrics = {
        op_label: {
            "calls": 0,
            "cpu_time_ms": 0.0,
            "gpu_time_ms": 0.0,
            "cpu_mem_mb": 0.0,
            "gpu_mem_mb": 0.0,
        }
        for op_label in operation_labels
    }

    for r in results["profiler_metrics"]:
        if r.get("operation") in operation_labels:
            op_label = r["operation"]
            d_type = r["device_type"]
            operation_metrics[op_label]["calls"] = r["count"]

            if d_type == "DeviceType.CPU":
                operation_metrics[op_label]["cpu_time_ms"] += r["cpu_time_total"] / 1000
                operation_metrics[op_label]["cpu_mem_mb"] += r["cpu_memory_usage"] / (
                    1024**2
                )

                # Super weird - but, it appears pytorch profiling puts the meaningful GPU info in the device type CPU records
                operation_metrics[op_label]["gpu_time_ms"] += (
                    r["device_time_total"] / 1000
                )
                operation_metrics[op_label]["gpu_mem_mb"] += max(
                    r["device_memory_usage"], 0
                ) / (1024**2)

            elif d_type == "DeviceType.CUDA":
                # Use only "self" time; total time was already counted in CPU block (see above comment)
                operation_metrics[op_label]["gpu_time_ms"] += (
                    r["self_device_time_total"] / 1000
                )

    table = [
        [
            op_label,
            metric["calls"],
            f"{metric['cpu_time_ms']:.2f}",
            f"{metric['gpu_time_ms']:.2f}",
            f"{metric['cpu_mem_mb']:.2f}",
            f"{metric['gpu_mem_mb']:.2f}",
        ]
        for op_label, metric in operation_metrics.items()
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
        all_files.extend(sorted(glob.glob(f"{args.dir}/*.log")))
    for f in args.files:
        all_files.extend(sorted(glob.glob(f"{f}")))
    if not all_files:
        raise FileNotFoundError(
            f"No JSONL files found in files: {args.files} / dir: {args.dir}"
        )

    # Print summary tables for experiment type
    for fpath in all_files:
        experiment = Path(fpath).stem
        metrics = get_metrics(fpath, METRICS_MESSAGE_MAP[args.metric_type])
        if not metrics:
            print(f"No metrics found for {fpath} of type {args.metric_type}")
            continue

        print(f"\n=== Results for experiment: {experiment} ===")
        if args.metric_type == TRAINING_RESULTS_METRIC_TYPE:
            print(format_training_results(metrics))
        elif args.metric_type == PROFILER_METRICS_TYPE:
            operation_labels = []
            if "single_gpu" in experiment:
                operation_labels = EXPERIMENT_PROFILER_OPERATION_LABELS["single_gpu"]
            elif "ddp" in experiment:
                operation_labels = EXPERIMENT_PROFILER_OPERATION_LABELS["ddp"]
            print(format_profile_metrics(metrics, operation_labels))


if __name__ == "__main__":
    main()
