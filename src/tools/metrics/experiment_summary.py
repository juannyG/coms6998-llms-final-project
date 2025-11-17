import collections

from tools.metrics.metrics_dataclasses import ProfilerSummary, TrainingResults


def group_by_experiments(device_log_iterator):
    experiment_groups = collections.defaultdict(list)
    for device_summary in device_log_iterator:
        experiment_groups[device_summary.experiment_key].append(device_summary)
    return experiment_groups


def generate_experiment_summary(device_log_iterator):
    grouped = group_by_experiments(device_log_iterator)

    for run_key, device_entries in grouped.items():
        metric_type = device_entries[0].metric_type
        print(f"\n=== Aggregated Results for {run_key} ({metric_type}) ===")

        # TODO: Move the "metric type" constants some place else
        if metric_type == "training":
            aggregate_training = TrainingResults.aggregate(
                [d.summary for d in device_entries]
            )
            print(aggregate_training.to_table())

        elif metric_type == "profiler":
            merged_profiler_summary = ProfilerSummary.aggregate(
                [d.summary for d in device_entries]
            )
            print(merged_profiler_summary.to_table())
