"""
This is a simple script meant to take a log file or an entire directory 


Usage:
    python post_processing/summarize_metrics.py training_results --log-dir ../logs/

    python post_processing/summarize_metrics.py profiler_metrics --log-files ../logs/run_single_gpu_10m_cuda_0.log
"""

import argparse
import json
import glob
from pathlib import Path

from tabulate import tabulate
from torch.cuda import device_memory_used

TRAINING_RESULTS_METRIC_TYPE = 'training_results'
PROFILER_METRICS_TYPE = 'profiler_metrics'
METRICS_MESSAGE_MAP = {
    # TODO: The values feels fragile - we can probably tighten up the structural connection with the JSONL log files...
    TRAINING_RESULTS_METRIC_TYPE: "Training results",
    PROFILER_METRICS_TYPE: "Profiler metrics"
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
        ["GPU Utilization", f"{result['gpu_util_percent']}%"]
    ]
    return tabulate(table, headers=["Metric", "Value"], tablefmt="github")


def format_profile_metrics(results):
    # We need to massage this a bit first...see experiments for labels
    operation_labels = ['model_forward', 'model_loss', 'model_backward', 'ddp_communication', 'model_optimizer_step']
    operation_metrics = {
        op_label: {
            "calls": 0,
            "device_time_ms": 0.0,
            "device_mem_mb": 0.0,
        } for op_label in operation_labels
    }

    print(results)
    for r in results['profiler_metrics']:
        if r.get('operation') in operation_labels:
            op_label = r['operation']
            operation_metrics[op_label]['calls'] = r['count']
            operation_metrics[op_label]['device_time_ms'] = r['device_time_total'] / 1000
            operation_metrics[op_label]['device_mem_mb'] = r['device_memory_usage'] / (1024**2)

    table = [
        [op_label, metric["calls"], f"{metric['device_time_ms']:.2f}", f"{metric['device_mem_mb']:.2f}"]
        for op_label, metric in operation_metrics.items()
    ]
    return tabulate(table, headers=["Operation", "Calls", "Time (ms)", "Memory (MB)"], tablefmt="github")


def get_metrics(file_path, metric_msg):
    with open(file_path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get('message') == metric_msg:
                    return obj
            except json.JSONDecodeError:
                print("failed to load json")
                continue  # skip malformed lines
    return None


def main():
    parser = argparse.ArgumentParser(description="Provide per-device metrics summary in a more concise and readable manner.")
    parser.add_argument('metric_type', type=str, choices=list(METRICS_MESSAGE_MAP.keys()), help="The type of metric to extract and summarize.")
    parser.add_argument("--log-files", nargs="+", required=False, help="Specific JSONL file.")
    parser.add_argument("--log-dir", type=str, required=False, help="Directory containing rank JSONL files.")
    args = parser.parse_args()

    if not args.log_files and not args.log_dir:
        print("ERROR: You must specify either --log-files or --log-dir")
        parser.print_help()
        exit(1)

    # Load the log file(s)
    all_files = []
    if args.log_dir:
        all_files.extend(sorted(glob.glob(f"{args.log_dir}/*.log")))
    for f in args.log_files:
        all_files.extend(sorted(glob.glob(f"{f}")))
    if not all_files:
        raise FileNotFoundError(f"No JSONL files found in {args.log_file if args.log_file else args.log_dir}")

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
            print(format_profile_metrics(metrics))


if __name__ == "__main__":
    main()
