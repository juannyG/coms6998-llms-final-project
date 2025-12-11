"""
These dataclasses are meant to act as datamodel schemas for converting our
different result types into things we can use across aggregation scripts.
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
    Aggregate TrainingResults for a particular strategy

    Total tokens
        * In the case of DDP + ZeRO, all devices see DIFFERENT tokens, so we take the sum
        * In the case of pipeline parallelism, the last device contains the "total" (the others have 0, so we can still take the sum)
        * In the case of tensor parallelism, all devices see the SAME tokens, so we choose one
    Total time is the maximum across devices
    Total throughput is total tokens / total time
    Final loss is less important to us, but we'll take the maximum across devices
    Avergage GPU memory is the average of the averages
    Total GPU memory is the sum of the averages - gives a sense of general memory requirements
    Peak GPU memory is the maximum across all the devices
    Average GPU utilization is the average of the averages
    Min average GPU utilization shows load balancing issues
    """

    def __init__(
        self,
        strategy,
        model_size,
        n_devices,
        training_results=None,
    ):
        self.strategy = strategy
        self.training_results = training_results or []
        self.n_devices = n_devices
        self.model_size = model_size

        # Aggregated training results
        self.total_tokens = 0
        self.total_time_s = 0.0
        self.total_throughput = 0.0
        self.final_loss = 0.0
        self.avg_gpu_mem_mb = 0.0
        self.total_avg_gpu_mem_mb = 0.0
        self.peak_gpu_mem_mb = 0.0
        self.avg_gpu_util_percent = 0.0
        self.min_avg_gpu_util_percent = 0.0

    def build_summary(self):
        for tr in self.training_results:
            if 'simple_zero' in self.strategy or self.strategy in [
                "torch_ddp",
                "torch_gpipe",  # See comment why we're summing here
                "megatron_pipeline_parallel",  # See comment why we're summing here
                "megatron_ddp",  # See comment why we're summing here
                "simple_zero",  # See comment why we're summing here
            ]:  # TODO: Move these constants, fragile
                self.total_tokens += tr.total_tokens
            else:
                self.total_tokens = max(tr.total_tokens, self.total_tokens)
            self.total_time_s = max(tr.total_time_s, self.total_time_s)
            self.final_loss = max(tr.final_loss, self.final_loss)
            self.total_avg_gpu_mem_mb += tr.avg_gpu_mem_mb
            self.peak_gpu_mem_mb = max(tr.peak_gpu_mem_mb, self.peak_gpu_mem_mb)
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
            ["Avg GPU Utilization", f"{self.avg_gpu_util_percent:.2f}%"],
            ["Min avg GPU Utilization", f"{self.min_avg_gpu_util_percent:.2f}%"],
        ]
        return tabulate(table, headers=["Metric", "Value"], tablefmt="github")


class ComparisonSummmary:
    """
    Given a summary of multiple devices for a specific experiment type and a specific
    single-GPU baseline, compute the following metrics:

    * Relative Runtime Overhead %
        - Definition: How much slower or faster the multi-GPU run is compared to single GPU.
        - Computation: (multi_gpu_total_time / single_gpu_total_time - 1) * 100
        - Interpretation:
            • > 0%  --> multi-GPU is slower (bad)
            • < 0%  --> multi-GPU is faster (good)
        - Notes: This captures effective *end-to-end* overhead, including overlapped
          communication/computation

    * Ideal Scaling Throughput (Shallue et al., 2018)
        - Computation: single_gpu_throughput * num_devices
        - Interpretation: The throughput we would get under perfect linear scaling.

    * Throughput scaling factor
        - Computation: multi_gpu_throughput / single_gpu_throughput
        - Interpretation:
            • 1.0  --> same speed as a single GPU
            • < 1  --> slower than single GPU
            • > 1  --> faster than single GPU
            • ~= num_devices --> ideal scaling
        - Purpose: Measures absolute performance gain (not normalized by device count).

    * Throughput Scaling Efficiency (Shallue et al., 2018)
        - Computation: (throughput scaling factor / num_devices) * 100
        - Interpretation:
            • 100% --> perfect linear scaling
            • 0–100% --> typical real-world scaling
            • Always <= 100% by definition (unlike relative throughput uplift)

    * Memory Scaling Factor
        - Computation: multi_gpu_avg_memory / single_gpu_avg_memory
        - Interpretation:
            • < 1 --> uses less memory per GPU than single GPU baseline
            • ~1 --> similar memory behavior
            • > 1 --> uses more memory per GPU (common for DDP due to full model replication)
    """

    def __init__(self, baseline_results, experiment_summary):
        self.baseline_results = baseline_results
        self.experiment_summary = experiment_summary

        self.relative_runtime_overhead_percent = (
            (self.experiment_summary.total_time_s / self.baseline_results.total_time_s)
            - 1
        ) * 100

        self.ideal_scaling_throughput = (
            self.baseline_results.total_throughput * self.experiment_summary.n_devices
        )

        self.throughput_scaling_factor = (
            self.experiment_summary.total_throughput
            / self.baseline_results.total_throughput
        )

        self.throughput_efficiency_percent = (
            self.throughput_scaling_factor / self.experiment_summary.n_devices
        ) * 100

        self.memory_scaling_factor = (
            self.experiment_summary.avg_gpu_mem_mb
            / self.baseline_results.avg_gpu_mem_mb
        )

        self.total_gpu_mem_mb = (
            self.experiment_summary.avg_gpu_mem_mb * self.experiment_summary.n_devices
        )
        self.total_memory_scaling_factor = self.total_gpu_mem_mb / (
            self.baseline_results.avg_gpu_mem_mb
        )

        self.peak_memory_scaling_factor = (
            self.experiment_summary.peak_gpu_mem_mb
            / self.baseline_results.peak_gpu_mem_mb
        )

    def to_table(self):
        table = [
            ["Strategy", self.experiment_summary.strategy],
            ["Number of Devices", self.experiment_summary.n_devices],
            ["Total Time", f"{self.experiment_summary.total_time_s:.2f} seconds"],
            [
                "Total Throughput",
                f"{self.experiment_summary.total_throughput:.2f} tokens/sec",
            ],
            ["Avg GPU Mem", f"{self.experiment_summary.avg_gpu_mem_mb:.2f} MB"],
            ["Total GPU Mem", f"{self.total_gpu_mem_mb:.2f} MB"],
            ["Peak GPU Mem", f"{self.experiment_summary.peak_gpu_mem_mb:.2f} MB"],
            ["Avg GPU Util %", f"{self.experiment_summary.avg_gpu_util_percent:.2f}%"],
            [
                "Relative Runtime Overhead",
                f"{self.relative_runtime_overhead_percent:.2f}%",
            ],
            ["Ideal Scaling Throughput", f"{self.ideal_scaling_throughput} tokens/sec"],
            ["Throughput Scaling Factor", f"{self.throughput_scaling_factor}"],
            ["Throughput Efficiency", f"{self.throughput_efficiency_percent:.2f}%"],
            ["Memory Scaling Factor", f"{self.memory_scaling_factor:.2f}"],
            ["Total Memory Scaling Factor", f"{self.total_memory_scaling_factor:.2f}"],
            ["Peak Memory Scaling Factor", f"{self.peak_memory_scaling_factor:.2f}"],
        ]
        return tabulate(table, headers=["Metric", "Value"], tablefmt="github")
