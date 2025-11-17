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

from experiments import single_gpu, torch_ddp, torch_gpipe, tensor_parallel
from tools.metrics.experiment_summary import generate_experiment_summary
from tools.metrics.metrics_dataclasses import TrainingResults, ProfilerSummary

DEVICE_LEVEL = "device"
EXPERIMENT_LEVEL = "experiment"


@dataclass
class DeviceSummary:
    """
    This provides a mapping of what profiler labels should be included the in the profiler summary
    based on the type of experiment. For example: ddp_communication is not recorded nor does it make
    sense to include in the `single_gpu` experiment summary
    """

    # TODO: Instead of importing these here, flip it around: define the labels here and have the experiments import them
    EXPERIMENT_PROFILER_OPERATION_LABELS = {
        "single_gpu": single_gpu.EXPERIMENT_PROFILER_LABELS,
        "ddp": torch_ddp.EXPERIMENT_PROFILER_LABELS,
        "gpipe": torch_gpipe.EXPERIMENT_PROFILER_LABELS,
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
    summary: Any

    @property
    def experiment_key(self):
        return f"{self.strategy}/{self.model_size}/{self.run_id}"

    @property
    def device_experiment(self):
        return f"{self.experiment_key}/{self.device_id}"


class DeviceLogIterator:
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

            metrics = get_metrics(
                fpath, DeviceSummary.METRICS_MESSAGE_MAP[self.metric_type]
            )
            if not metrics:
                continue

            device_summary = None
            if self.metric_type == DeviceSummary.TRAINING_RESULTS_METRIC_TYPE:
                device_summary = TrainingResults.from_dict(metrics)
            elif self.metric_type == DeviceSummary.PROFILER_METRICS_TYPE:
                operation_labels = []
                if "single_gpu" in strategy:
                    operation_labels = (
                        DeviceSummary.EXPERIMENT_PROFILER_OPERATION_LABELS["single_gpu"]
                    )
                elif "ddp" in strategy:
                    operation_labels = (
                        DeviceSummary.EXPERIMENT_PROFILER_OPERATION_LABELS["ddp"]
                    )
                elif "tensor_parallel" in strategy:
                    operation_labels = (
                        DeviceSummary.EXPERIMENT_PROFILER_OPERATION_LABELS[
                            "tensor_parallel"
                        ]
                    )

                profiler_summary = ProfilerSummary(operation_labels=operation_labels)
                profiler_summary.update_from_profiler_metrics(
                    metrics["profiler_metrics"]
                )
                device_summary = profiler_summary

            yield DeviceSummary(
                path=fpath,
                device_id=device_id,
                strategy=strategy,
                model_size=model_size,
                run_id=run_id,
                metric_type=self.metric_type,
                summary=device_summary,
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
        choices=[DEVICE_LEVEL, EXPERIMENT_LEVEL],  # TODO: comparison
        help="The type of summary you want produced",
    )
    parser.add_argument(
        "metric_type",
        type=str,
        choices=list(DeviceSummary.METRICS_MESSAGE_MAP.keys()),
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

    dev_log_iterator = DeviceLogIterator(args.metric_type, all_files)
    if args.level == DEVICE_LEVEL:
        for device_summary in dev_log_iterator:
            print(
                f"\n=== Results for experiment: {device_summary.device_experiment} ==="
            )
            print(device_summary.summary.to_table())
    elif args.level == EXPERIMENT_LEVEL:
        generate_experiment_summary(dev_log_iterator)


if __name__ == "__main__":
    main()
