"""
These dataclasses are meant to act as datamodel schemas for converting our
different result types into things we can use across aggregation scripts.
"""

import abc
from dataclasses import dataclass, field

from tabulate import tabulate


class TabularMetric(abc.ABC):
    @abc.abstractmethod
    def to_table(self):
        pass


@dataclass
class TrainingResults(TabularMetric):
    avg_tokens_per_s: float = 0.0
    avg_samples_per_s: float = 0.0
    avg_loss: float = 0.0
    total_tokens: int = 0
    total_time_s: float = 0.0
    avg_gpu_mem_mb: float = 0.0
    peak_gpu_mem_mb: float = 0.0
    avg_gpu_util_percent: float = 0.0

    def to_dict(self):
        return {
            "avg_tokens_per_s": self.avg_tokens_per_s,
            "avg_samples_per_s": self.avg_samples_per_s,
            "avg_loss": self.avg_loss,
            "total_tokens": self.total_tokens,
            "total_time_s": self.total_time_s,
            "avg_gpu_mem_mb": self.avg_gpu_mem_mb,
            "peak_gpu_mem_mb": self.peak_gpu_mem_mb,
            "avg_gpu_util_percent": self.avg_gpu_util_percent,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            avg_tokens_per_s=d["avg_tokens_per_s"],
            avg_samples_per_s=d["avg_samples_per_s"],
            avg_loss=d["avg_loss"],
            total_tokens=d["total_tokens"],
            total_time_s=d["total_time_s"],
            avg_gpu_mem_mb=d["avg_gpu_mem_mb"],
            peak_gpu_mem_mb=d["peak_gpu_mem_mb"],
            avg_gpu_util_percent=d["avg_gpu_util_percent"],
        )

    @classmethod
    def aggregate(cls, training_results):
        # TODO: MOVE THIS TO A DIFFERENT DATA CLASS
        """
        Produce an aggregate TrainingResults instance based on a list of TrainingResults

        Avg tokens/sec per device is summed across devices
        Avg samples/sec per device is summed across devices
        Avg loss per device is averaged across devices - it's going to be the same anyway
        Total tokens per device is summed across devices
        Total time is the maximum across devices
        Peak GPU memory per device is the max across devices
        Total GPU memory per device is summed across devices
        GPU util % per device can be averaged across devices
        """

        agg = cls()
        for tr in training_results:
            agg.avg_tokens_per_s += tr.avg_tokens_per_s
            agg.avg_samples_per_s += tr.avg_samples_per_s
            agg.avg_loss += tr.avg_loss
            agg.total_tokens += tr.total_tokens
            agg.total_time_s = max(agg.total_time_s, tr.total_time_s)
            agg.peak_gpu_mem_mb = max(agg.peak_gpu_mem_mb, tr.peak_gpu_mem_mb)
            agg.gpu_util_percent += tr.gpu_util_percent

        # Do the average after the summations
        agg.avg_loss /= len(training_results)
        agg.gpu_util_percent /= len(training_results)
        return agg

    def to_table(self):
        table = [
            ["Avg Tokens/sec", f"{self.avg_tokens_per_s:,.0f}"],
            ["Avg Samples/sec", f"{self.avg_samples_per_s:,.1f}"],
            ["Avg Loss", f"{self.avg_loss:.4f}"],
            ["Total Tokens", f"{self.total_tokens:,}"],
            ["Total Time", f"{self.total_time_s:.2f} sec"],
            ["Avg GPU Mem", f"{self.avg_gpu_mem_mb:.1f} MB"],
            ["Peak GPU Mem", f"{self.peak_gpu_mem_mb:.1f} MB"],
            ["Avg GPU Utilization", f"{self.avg_gpu_util_percent:.2f}%"],
        ]
        return tabulate(table, headers=["Metric", "Value"], tablefmt="github")


"""
The torch profiler metrics are a little trickier: we have a list of labels 
we're interested in and we have to go through multiple logs to pull out what 
we're looking for, unlike the `TrainingResults` where a single log line holds 
what we need.

Here we need two data classes: one for the operation summary and another to provide
the map of operation -> operation summary
"""


@dataclass
class ProfilerOperationSummary:
    # See experiments/* files for each experiment's specific list of operation labels used in torch's profiler
    operation: str
    calls: int = 0
    cpu_time_ms: float = 0.0
    gpu_time_ms: float = 0.0
    cpu_mem_mb: float = 0.0
    gpu_mem_mb: float = 0.0

    def add_cpu_metrics(self, record):
        self.cpu_time_ms += record["cpu_time_total"] / 1000
        self.cpu_mem_mb += record["cpu_memory_usage"] / (1024**2)
        # Super weird - but, it appears pytorch profiling puts the meaningful GPU info in the device type CPU records
        self.gpu_time_ms += record["device_time_total"] / 1000
        self.gpu_mem_mb += max(record["device_memory_usage"], 0) / (1024**2)

    def add_cuda_metrics(self, record):
        # Use only "self" time; total time was already counted in CPU block (see above comment)
        self.gpu_time_ms += record["self_device_time_total"] / 1000

    def merge(self, other):
        """
        Take another ProfilerOperationSummary and merge the metrics

        For any operation, we take the sum of each device's operation to get an experiment-level
        view of the operation
        """
        self.calls += other.calls
        self.cpu_time_ms += other.cpu_time_ms
        self.gpu_time_ms += other.gpu_time_ms
        self.cpu_mem_mb += other.cpu_mem_mb
        self.gpu_mem_mb += other.gpu_mem_mb


@dataclass
class ProfilerSummary(TabularMetric):
    # operations[op_label] --> ProfilerOperationSummary
    operation_labels: list[str]
    operations: dict = field(default_factory=dict)

    def update_from_profiler_metrics(self, records):
        self.operations = {
            op_label: ProfilerOperationSummary(operation=op_label)
            for op_label in self.operation_labels
        }

        for r in records:
            op = r["operation"]
            if op not in self.operation_labels:
                continue

            d_type = r["device_type"]
            self.operations[op].calls = r["count"]
            if d_type == "DeviceType.CPU":
                self.operations[op].add_cpu_metrics(r)
            elif d_type == "DeviceType.CUDA":
                self.operations[op].add_cuda_metrics(r)

    @classmethod
    def aggregate(cls, other_profiler_summaries):
        agg = cls(operation_labels=other_profiler_summaries[0].operation_labels)
        for other_profiler_summary in other_profiler_summaries:
            for op, op_summary in other_profiler_summary.operations.items():
                if op not in agg.operations:
                    # Init the op level summary if we don't have one yet
                    agg.operations[op] = ProfilerOperationSummary(operation=op)
                agg.operations[op].merge(op_summary)
        return agg

    def to_table(self):
        table = [
            [
                op_label,
                profiler_op_summary.calls,
                f"{profiler_op_summary.cpu_time_ms:.2f}",
                f"{profiler_op_summary.gpu_time_ms:.2f}",
                f"{profiler_op_summary.cpu_mem_mb:.2f}",
                f"{profiler_op_summary.gpu_mem_mb:.2f}",
            ]
            for op_label, profiler_op_summary in self.operations.items()
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
