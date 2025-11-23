"""
These dataclasses are meant to act as datamodel schemas for converting our
different result types into things we can use across aggregation scripts.


TODO: Dataclass for scaling efficiences
    * throughput_scaling_efficiency = (distributed_throughput / single_gpu_throughput) / num_gpus
    * memory_efficiency = single_gpu_peak_memory / (total_system_peak_memory / num_gpus)
    * communication_overhead = 1 - (distributed_throughput / (single_gpu_throughput * num_gpus))
"""

import abc
from dataclasses import dataclass

from tabulate import tabulate


class TabularMetric(abc.ABC):
    @abc.abstractmethod
    def to_table(self):
        return str()


@dataclass
class TrainingResults(TabularMetric):
    total_tokens: int = 0
    total_time_s: float = 0.0
    total_throughput: float = 0.0
    final_loss: float = 0.0
    avg_gpu_mem_mb: float = 0.0
    peak_gpu_mem_mb: float = 0.0
    avg_gpu_util_percent: float = 0.0

    def to_dict(self):
        return {
            "total_tokens": self.total_tokens,
            "total_time_s": self.total_time_s,
            "total_throughput": self.total_throughput,
            "final_loss": self.final_loss,
            "avg_gpu_mem_mb": self.avg_gpu_mem_mb,
            "peak_gpu_mem_mb": self.peak_gpu_mem_mb,
            "avg_gpu_util_percent": self.avg_gpu_util_percent,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            total_tokens=d["total_tokens"],
            total_time_s=d["total_time_s"],
            total_throughput=d["total_throughput"],
            final_loss=d["final_loss"],
            avg_gpu_mem_mb=d["avg_gpu_mem_mb"],
            peak_gpu_mem_mb=d["peak_gpu_mem_mb"],
            avg_gpu_util_percent=d["avg_gpu_util_percent"],
        )

    def to_table(self):
        table = [
            ["Total Tokens", f"{self.total_tokens:,}"],
            ["Total Time", f"{self.total_time_s:.2f} sec"],
            ["Total Throughput", f"{self.total_throughput:.2f} tokens/sec"],
            ["Final Loss", f"{self.final_loss:.4f}"],
            ["Avg GPU Mem", f"{self.avg_gpu_mem_mb:.1f} MB"],
            ["Peak GPU Mem", f"{self.peak_gpu_mem_mb:.1f} MB"],
            ["Avg GPU Utilization", f"{self.avg_gpu_util_percent:.2f}%"],
        ]
        return tabulate(table, headers=["Metric", "Value"], tablefmt="github")


class ExperimentSummary(TabularMetric):
    """
    Aggregate TrainingResults and ProfilerSummaries for a particular strategy

    Total tokens
        * In the case of DDP, all devices see DIFFERENT tokens, so we take the sum
        * In the case of pipeline parallelism, the last device contains the "total" (the others have 0, so we can still take the sum)
        * In the case of tensor parallelism, all devices see the SAME tokens, so we choose one
    Total time is the maximum across devices
    Total throughput is total tokens / total time
    Final loss is less important to us, but we'll take the maximum across devices
    Avergage GPU memory is the average of the averages
    Total GPU memory is the sum of the averages - gives a sense of general memory requirements
    Peak GPU memory is the maximum across all the devices
    Total peak GPU memory - gives a sense of the worst case memory requirements
    Average GPU utilization is the average of the averages
    Min average GPU utilization shows load balancing issues

    # TODO: HOW DO WE APPLY THESE OUTSIDE OF THE PROFILER?
    For communication time from ProfilerSummary - pick max value across devices. However, it's worth noting:
    * DDP
      * all devices do identical all-reduce work
      * communication happens in parallel, not sequentially
    * tensor parallel - same rules as DDP
      * all devices do identical all-reduce (and/or sum + scatter) work
      * communication happens in parallel, not sequentially
    * gpipe/pipeline parallelism
      * Different devices have different communication patterns (send-only, recv-only, send+recv)
      * System bottleneck = the device with highest communication overhead
    """

    def __init__(
        self, strategy, n_devices, training_results=None, profiler_results=None
    ):
        self.strategy = strategy
        self.training_results = training_results or []
        self.n_devices = n_devices

        # Aggregated training results
        self.total_tokens = 0
        self.total_time_s = 0.0
        self.total_throughput = 0.0
        self.final_loss = 0.0
        self.avg_gpu_mem_mb = 0.0
        self.total_avg_gpu_mem_mb = 0.0
        self.peak_gpu_mem_mb = 0.0
        self.total_peak_gpu_mem_mb = 0.0
        self.avg_gpu_util_percent = 0.0
        self.min_avg_gpu_util_percent = 0.0

    def build_summary(self):
        for tr in self.training_results:
            if self.strategy in [
                "torch_ddp",
                "torch_gpipe",  # See comment why we're summing here
                "megatron_pipeline_parallel",  # See comment why we're summing here
            ]:  # TODO: Move these constants, fragile
                self.total_tokens += tr.total_tokens
            else:
                self.total_tokens = max(tr.total_tokens, self.total_tokens)
            self.total_time_s = max(tr.total_time_s, self.total_time_s)
            self.final_loss = max(tr.final_loss, self.final_loss)
            self.total_avg_gpu_mem_mb += tr.avg_gpu_mem_mb
            self.peak_gpu_mem_mb = max(tr.peak_gpu_mem_mb, self.peak_gpu_mem_mb)
            self.total_peak_gpu_mem_mb += tr.peak_gpu_mem_mb
            self.avg_gpu_util_percent += tr.avg_gpu_util_percent
            self.min_avg_gpu_util_percent = min(
                tr.avg_gpu_util_percent, self.min_avg_gpu_util_percent or 100
            )
        self.total_throughput = self.total_tokens / self.total_time_s
        self.avg_gpu_mem_mb = self.total_avg_gpu_mem_mb / self.n_devices
        self.avg_gpu_util_percent /= self.n_devices

    def to_table(self):
        table = [
            ["Number of devices", f"{self.n_devices}"],
            ["Total Tokens", f"{self.total_tokens:,}"],
            ["Total Time", f"{self.total_time_s:.2f} sec"],
            ["Total Throughput", f"{self.total_throughput:.2f} tokens/sec"],
            ["Final Loss", f"{self.final_loss:.4f}"],
            ["Avg GPU Mem", f"{self.avg_gpu_mem_mb:.1f} MB"],
            ["Total avg GPU Mem", f"{self.total_avg_gpu_mem_mb:.1f} MB"],
            ["Peak GPU Mem", f"{self.peak_gpu_mem_mb:.1f} MB"],
            ["Total peak GPU Mem", f"{self.total_peak_gpu_mem_mb:.1f} MB"],
            ["Avg GPU Utilization", f"{self.avg_gpu_util_percent:.2f}%"],
            ["Min avg GPU Utilization", f"{self.min_avg_gpu_util_percent:.2f}%"],
        ]
        return tabulate(table, headers=["Metric", "Value"], tablefmt="github")


class ComparisonSummmary:
    def __init__(self, baseline_results, experiment_summary):
        self.baseline_results = baseline_results
        self.experiment_summary = experiment_summary

    def to_table(self):
        """
        Given a summary of multiple devices for a specific experiment type and a specific
        single GPU baseline, calculate

        * Distributed Training Overhead: everything the training strategy adds relative to single GPU
            (Multi-GPU total_time - Single GPU total_time) / Multi-GPU total_time
            - we want small %s

        * Throughput Efficiency: what's the speedup factor (percentage)
            Multi-GPU total throughput / single GPU total throughput
            - we want big %s (as close to 100% as possible)

        * Memory Efficiency: how much better/worse is the memory usage?
            Loading...
        """

        distributed_training_overhead = (
            (self.experiment_summary.total_time_s - self.baseline_results.total_time_s)
            / self.experiment_summary.total_time_s
            * 100
        )

        # Validation: this should just be (100% - comms overhead)
        throughput_efficiency_percent = (
            self.experiment_summary.total_throughput
            / self.baseline_results.total_throughput
        ) * 100

        total_mem_used = self.experiment_summary.avg_gpu_mem_mb
        if self.experiment_summary.strategy == "torch_ddp":
            total_mem_used *= self.experiment_summary.n_devices
        memory_scaling_factor = (
            total_mem_used / (self.baseline_results.avg_gpu_mem_mb)
        )
        table = [
            ["Strategy", self.experiment_summary.strategy],
            ["Number of Devices", self.experiment_summary.n_devices],
            ["Total Time", f"{self.experiment_summary.total_time_s:.2f} seconds"],
            [
                "Total Throughput",
                f"{self.experiment_summary.total_throughput:.2f} tokens/sec",
            ],
            ["Avg GPU Mem", f"{self.experiment_summary.avg_gpu_mem_mb:.2f} MB"],
            ["Avg GPU Util %", f"{self.experiment_summary.avg_gpu_util_percent:.2f}%"],
            ["Distributed Training Overhead", f"{distributed_training_overhead:.2f}%"],
            ["Throughput Efficiency", f"{throughput_efficiency_percent:.2f}%"],
            ["Memory Scaling Factor", f"{memory_scaling_factor:.2f}"],
        ]
        return tabulate(table, headers=["Metric", "Value"], tablefmt="github")
