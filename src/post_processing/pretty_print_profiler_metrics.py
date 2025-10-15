"""
Usage:
    python post_processing/pretty_print_profiler_metrics.py --log-dir ../logs/
    python post_processing/pretty_print_profiler_metrics.py --log-file ../logs/run_single_gpu_10m_cuda_0.log --top 50
"""

import argparse
import json
import glob
from pathlib import Path

import pandas as pd
from tabulate import tabulate


def load_profiler_metrics(file_path):
    """Load all profiler_metrics entries from a JSONL file."""
    with open(file_path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get('profiler_metrics'):
                    return obj['profiler_metrics']
            except json.JSONDecodeError:
                print("failed to load json")
                continue  # skip malformed lines
    return None


def main():
    parser = argparse.ArgumentParser(description="Pretty print PyTorch profiler metrics as a table.")
    parser.add_argument("--log-file", type=str, required=False, help="Specific JSONL file.")
    parser.add_argument("--log-dir", type=str, required=False, help="Directory containing rank JSONL files.")
    parser.add_argument("--sort-by", default="cpu_time_total", help="Column to sort by")
    parser.add_argument("--top", type=int, default=20, help="Number of rows to show")
    args = parser.parse_args()

    if not args.log_file and not args.log_dir:
        print("ERROR: You must specify either a --log-file or --log-dir")
        parser.print_help()
        exit(1)
    elif args.log_file and args.log_dir:
        print("Both a --log-dir and --log-file detected: --log-file will take precedence")

    # Load all rank JSONL files
    all_files = []
    if args.log_dir:
        all_files = sorted(glob.glob(f"{args.log_dir}/*.log"))
    if args.log_file:
        all_files = sorted(glob.glob(f"{args.log_file}"))

    if not all_files:
        raise FileNotFoundError(f"No JSONL files found in {args.log_file if args.log_file else args.log_dir}")

    all_data = []
    for fpath in all_files:
        rank = Path(fpath).stem
        metrics = load_profiler_metrics(fpath)
        if not metrics:
            print(f"No metrics found for {fpath}")
        df = pd.DataFrame(metrics).sort_values(args.sort_by, ascending=False).head(args.top)

        print(f"\n=== Rank: {rank} ===")
        print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))


if __name__ == "__main__":
    main()

