"""
Main entry point meant to take a log file or an entire directory and produce
summaries of different type of measurements from our experiments at different
levels of aggregation.

Usage:
    python tools/metrics/summary.py device --dir ../logs/

    python tools/metrics/summary.py experiment --files ../logs/single_gpu/10m/1234567890/*

    python tools/metrics/summary.py compare --baseline ../logs/single_gpu/100m/7890123456/cuda_0.log--files ../logs/torch_ddp/100m/1234567890/
"""

import argparse
import csv
import json
import glob
from dataclasses import dataclass
from pathlib import Path

from tools.metrics.experiment_summary import generate_experiment_summary
from tools.metrics.metrics_dataclasses import ComparisonSummmary, TrainingResults

DEVICE_LEVEL = "device"
EXPERIMENT_LEVEL = "experiment"
COMPARISON_LEVEL = "compare"
ALL_LEVEL = "all"
LEVELS = [DEVICE_LEVEL, EXPERIMENT_LEVEL, COMPARISON_LEVEL, ALL_LEVEL]


@dataclass
class DeviceSummary:
    TRAINING_RESULTS_LOG_MESSAGE = "Training results"

    path: Path
    device_id: str
    strategy: str
    model_size: str
    run_id: str
    training_results: TrainingResults

    @property
    def experiment_key(self):
        return f"{self.strategy}/{self.model_size}/{self.run_id}"

    @property
    def device_experiment(self):
        return f"{self.experiment_key}/{self.device_id}"


class DeviceLogIterator:
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for fpath in self.files:
            run_path = Path(fpath).parent
            strategy = run_path.parts[-3]
            model_size = run_path.parts[-2]
            run_id = run_path.parts[-1]
            device_id = Path(fpath).stem

            training_results = TrainingResults()
            training_metrics = get_metrics(
                fpath, DeviceSummary.TRAINING_RESULTS_LOG_MESSAGE
            )
            if training_metrics:
                training_results = TrainingResults.from_dict(training_metrics)

            yield DeviceSummary(
                path=fpath,
                device_id=device_id,
                strategy=strategy,
                model_size=model_size,
                run_id=run_id,
                training_results=training_results,
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
        description="Provide metrics summary for a given leven in a more concise and readable manner."
    )
    parser.add_argument(
        "level",
        type=str,
        choices=LEVELS,
        help="The type of summary you want produced",
    )
    parser.add_argument("--files", nargs="*", default=[], help="List of files")
    parser.add_argument(
        "--dir", type=str, required=False, help="Directory containing rank JSONL files."
    )
    parser.add_argument("--baseline", type=str, required=False, help="")
    parser.add_argument("--target_file", type=str, required=False, help="")
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

    dev_log_iterator = list(DeviceLogIterator(all_files))
    if args.level == DEVICE_LEVEL:
        for device_summary in dev_log_iterator:
            print(
                f"\n=== Results for experiment: {device_summary.device_experiment} ==="
            )
            print(device_summary.training_results.to_table())
    elif args.level == EXPERIMENT_LEVEL:
        summary_by_run_key = generate_experiment_summary(dev_log_iterator)
        for run_key, summary in summary_by_run_key.items():
            print(f"\n=== Aggregated Results for {run_key} ===")
            print(summary.to_table())
    elif args.level == COMPARISON_LEVEL:
        # Nice to have: pull the single gpu run for each strategy's model size dynamically...
        if not args.baseline:
            print(
                "ERROR: You must provide a baseline file (--baseline) to compare against"
            )
            exit(1)

        # This should throw execptions if the file does not exist
        run_path = Path(args.baseline).parent
        model_size = run_path.parts[-2]
        strategy = run_path.parts[-3]
        # TODO: With megatron - we can run the "tensor_parallel" experiment with nprocs=1 and get a "single GPU" baseline
        # if strategy != "single_gpu":
        #    print(f"ERROR: Baseilne files must be of strategy single_gpu: given {strategy}")
        #    exit(1)

        training_metrics = get_metrics(
            args.baseline, DeviceSummary.TRAINING_RESULTS_LOG_MESSAGE
        )
        if not training_metrics:
            print(f"ERROR: No training metrics found in {args.baseilne}")
            exit(1)

        baseline_results = TrainingResults.from_dict(training_metrics)
        print(f"\n=== Baseline metrics of {strategy}/{model_size}")
        print(baseline_results.to_table())

        summary_by_run_key = generate_experiment_summary(dev_log_iterator)
        for run_key, experiment_summary in summary_by_run_key.items():
            comparison = ComparisonSummmary(baseline_results, experiment_summary)
            print(
                f"\n=== Comparison Results of {strategy}/{model_size} against {run_key} ({experiment_summary.n_devices} devices) ==="
            )
            print(comparison.to_table())
    elif args.level == "all":
        if not args.target_file:
            print("ERROR: No target file defined.")
            exit(1)

        summary_by_run_key = generate_experiment_summary(dev_log_iterator)
        baseline_summaries = {}
        for run_key, experiment_summary in summary_by_run_key.items():
            if "single_gpu" not in run_key:
                continue
            baseline_summaries[experiment_summary.model_size] = experiment_summary

        with open(args.target_file, "w", newline="") as f:
            columns = [
                "model_size",
                "strategy",
                "num_devices",
                "total_tokens",
                "total_time_sec",
                "throughput_tokens_sec",
                "avg_gpu_mem_mb",
                "total_gpu_mem_mb",
                "peak_gpu_mem_mb",
                "avg_gpu_util_percent",
                "relative_runtime_overhead_percent",
                "ideal_scaling_throughput",
                "throughput_scaling_factor",
                "throughput_efficiency_percent",
                "memory_scaling_factor",
                "total_memory_scaling_factor",
                "peak_memory_scaling_factor",
            ]
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for run_key, experiment_summary in summary_by_run_key.items():
                baseline_summary = baseline_summaries[experiment_summary.model_size]
                baseline_training_results = baseline_summary.training_results[0]
                comparison = ComparisonSummmary(
                    baseline_training_results, experiment_summary
                )
                row = {
                    "model_size": experiment_summary.model_size,
                    "strategy": experiment_summary.strategy,
                    "num_devices": experiment_summary.n_devices,
                    "total_tokens": baseline_training_results.total_tokens,
                    "total_time_sec": experiment_summary.total_time_s,
                    "throughput_tokens_sec": experiment_summary.total_throughput,
                    "avg_gpu_mem_mb": experiment_summary.avg_gpu_mem_mb,
                    "total_gpu_mem_mb": comparison.total_gpu_mem_mb,
                    "peak_gpu_mem_mb": experiment_summary.peak_gpu_mem_mb,
                    "avg_gpu_util_percent": experiment_summary.avg_gpu_util_percent,
                    "relative_runtime_overhead_percent": comparison.relative_runtime_overhead_percent,
                    "ideal_scaling_throughput": comparison.ideal_scaling_throughput,
                    "throughput_scaling_factor": comparison.throughput_scaling_factor,
                    "throughput_efficiency_percent": comparison.throughput_efficiency_percent,
                    "memory_scaling_factor": comparison.memory_scaling_factor,
                    "total_memory_scaling_factor": comparison.total_memory_scaling_factor,
                    "peak_memory_scaling_factor": comparison.peak_memory_scaling_factor,
                }
                writer.writerow(row)


if __name__ == "__main__":
    main()
