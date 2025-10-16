"""
Usage:
    python post_processing/summarize_training_results.py --log-dir ../logs/
    python post_processing/summarize_training_results.py --log-file ../logs/run_single_gpu_10m_cuda_0.log
"""

import argparse
import json
import glob
from pathlib import Path

import pandas as pd
from tabulate import tabulate


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


def get_training_results(file_path):
    with open(file_path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get('message') == 'Training results':
                    return obj
            except json.JSONDecodeError:
                print("failed to load json")
                continue  # skip malformed lines
    return None


def main():
    parser = argparse.ArgumentParser(description="Pretty print PyTorch profiler metrics as a table.")
    parser.add_argument("--log-file", type=str, required=False, help="Specific JSONL file.")
    parser.add_argument("--log-dir", type=str, required=False, help="Directory containing rank JSONL files.")
    args = parser.parse_args()

    if not args.log_file and not args.log_dir:
        print("ERROR: You must specify either a --log-file or --log-dir")
        parser.print_help()
        exit(1)
    elif args.log_file and args.log_dir:
        print("Both a --log-dir and --log-file detected: --log-file will take precedence")

    # Load all log files
    all_files = []
    if args.log_dir:
        all_files = sorted(glob.glob(f"{args.log_dir}/*.log"))
    if args.log_file:
        all_files = sorted(glob.glob(f"{args.log_file}"))

    if not all_files:
        raise FileNotFoundError(f"No JSONL files found in {args.log_file if args.log_file else args.log_dir}")

    for fpath in all_files:
        experiment = Path(fpath).stem
        training_results = get_training_results(fpath)
        if not training_results:
            print(f"No training_results found for {fpath}")

        print(f"\n=== Results for experiment: {experiment} ===")
        print(format_training_results(training_results))


if __name__ == "__main__":
    main()
