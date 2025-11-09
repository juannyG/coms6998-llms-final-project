"""
Main entry point meant to take a log file or an entire directory and produce
summaries of different type of measurements from our experiments at different
levels of aggregation.

Usage:
    python tools/metrics/summary.py device training --dir ../logs/

    python tools/metrics/summary.py experiment profiler --files ../logs/single_gpu/10m/1234567890/*
"""

import argparse
from dataclasses import dataclass
import json
import glob
from pathlib import Path
from typing import Any

from experiments import single_gpu, torch_ddp, tensor_parallel
from tools.metrics_dataclasses import TrainingResults, ProfilerSummary


@dataclass
class ExperimentLogEntry:
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

    TRAINING_RESULTS_METRIC_TYPE = "training"
    PROFILER_METRICS_TYPE = "profiler"
    METRICS_MESSAGE_MAP = {
        # TODO: The values feels fragile - we can probably tighten up the structural connection with the JSONL log files...
        TRAINING_RESULTS_METRIC_TYPE: "Training results",
        PROFILER_METRICS_TYPE: "Profiler metrics",
    }

    path: Path
    device_id: str
    strategy: str
    model_size: str
    run_id: str
    metric_type: str
    summarized_metrics: Any


class ExperimentLogIterator:
    def __init__(self, metric_type, files):
        self.metric_type = metric_type
        self.files = files

    def __iter__(self):
        for fpath in self.files:
            run_path = Path(fpath).parent
            strategy = run_path.parts[-3]
            model_size = run_path.parts[-2]
            run_id = run_path.parts[-1]
            device_id = Path(fpath).stem

            metrics = get_metrics(fpath, ExperimentLogEntry.METRICS_MESSAGE_MAP[self.metric_type])
            if not metrics:
                continue

            summarized_metrics = None
            if self.metric_type == ExperimentLogEntry.TRAINING_RESULTS_METRIC_TYPE:
                summarized_metrics = TrainingResults.from_dict(metrics)
            elif self.metric_type == ExperimentLogEntry.PROFILER_METRICS_TYPE:
                pass

            yield ExperimentLogEntry(
                path=fpath,
                device_id=device_id,
                strategy=strategy,
                model_size=model_size,
                run_id=run_id,
                metric_type=self.metric_type,
                summarized_metrics=summarized_metrics,
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
        description="Provide metrics summary for a given level in a more concise and readable manner."
    )
    parser.add_argument(
        "level",
        type=str,
        choices=["device", "experiment"],  # TODO: comparison
        help="The type of summary you want produced",
    )
    parser.add_argument(
        "metric_type",
        type=str,
        choices=list(ExperimentLogEntry.METRICS_MESSAGE_MAP.keys()),
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
    #for fpath in all_files:
    #    run_path = Path(fpath).parent  # logs/torch_ddp/10m/1762628113/
    #    strategy = run_path.parts[-3]  # e.g., torch_ddp
    #    model_size = run_path.parts[-2]  # e.g., 10m
    #    run_id = run_path.parts[-1]  # e.g., 1762628113
    #    device_id = Path(fpath).stem  # e.g., cuda_1
    #    experiment = f"{strategy}/{model_size}/{run_id}/{device_id}"

    #    metrics = get_metrics(fpath, METRICS_MESSAGE_MAP[args.metric_type])
    #    if not metrics:
    #        print(f"No metrics found for {fpath} of type {args.metric_type}")
    #        continue

    #    print(f"\n=== Results for experiment: {experiment} ===")
    #    if args.metric_type == TRAINING_RESULTS_METRIC_TYPE:
    #        training_results = TrainingResults.from_dict(metrics)
    #        print(training_results.to_table())
    #    elif args.metric_type == PROFILER_METRICS_TYPE:
    #        operation_labels = []
    #        if "single_gpu" in experiment:
    #            operation_labels = EXPERIMENT_PROFILER_OPERATION_LABELS["single_gpu"]
    #        elif "ddp" in experiment:
    #            operation_labels = EXPERIMENT_PROFILER_OPERATION_LABELS["ddp"]
    #        elif "tensor_parallel" in experiment:
    #            operation_labels = EXPERIMENT_PROFILER_OPERATION_LABELS[
    #                "tensor_parallel"
    #            ]
    #        profiler_summary = ProfilerSummary()
    #        profiler_summary.update_from_profiler_metrics(
    #            metrics["profiler_metrics"], operation_labels
    #        )
    #        print(profiler_summary.to_table())


if __name__ == "__main__":
    main()
