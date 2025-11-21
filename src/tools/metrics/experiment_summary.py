import collections

from tools.metrics.metrics_dataclasses import ExperimentSummary


def group_by_experiments(device_log_iterator):
    experiment_groups = collections.defaultdict(list)
    for device_summary in device_log_iterator:
        experiment_groups[device_summary.experiment_key].append(device_summary)
    return experiment_groups


def generate_experiment_summary(device_log_iterator):
    grouped = group_by_experiments(device_log_iterator)

    for run_key, device_entries in grouped.items():
        strategy = device_entries[0].strategy
        n_devices = len(device_entries)
        all_training_results = [d.training_results for d in device_entries]

        summary = ExperimentSummary(strategy, n_devices, all_training_results)
        summary.build_summary()

        print(f"\n=== Aggregated Results for {run_key} ===")
        print(summary.to_table())

