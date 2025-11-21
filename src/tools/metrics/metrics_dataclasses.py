"""
These dataclasses are meant to act as datamodel schemas for converting our
different result types into things we can use across aggregation scripts.


TODO: Dataclass for scaling efficiences
    * throughput_scaling_efficiency = (distributed_throughput / single_gpu_throughput) / num_gpus
    * memory_efficiency = single_gpu_peak_memory / (total_system_peak_memory / num_gpus)
    * communication_overhead = 1 - (distributed_throughput / (single_gpu_throughput * num_gpus))

Using the profiler data, we can extract comms measurements, extrapolate up to 200 (# of training steps)
and get
    * comm_time_percent = total_communication_time / total_step_time
    * pure_compute_overhead = comm_overhead - comm_time_percent
"""

import abc
from dataclasses import dataclass, field

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
        self.profiler_results = profiler_results or []
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

        # Aggregated comms time, based on profiler data
        self.communication_time_s = 0.0

    def build_summary(self):
        for tr in self.training_results:
            if self.strategy in [
                "torch_ddp",
                "torch_gpipe",
                "megatron_pipeline_parallel",
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
            ["Communication time", f"{self.communication_time_s:.2f} sec"],
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



"""
GPIPE
c10d::send
c10d::recv_
nccl:coalesced
ncclDevKernel_SendRecv
TODO: does bfloat16 have any extra things?

DDP
nccl:all_reduce
ncclDevKernel_AllReduce_Sum_f32_RING_LL <--- what about for bfloat16?

Megatorn Tensor Parallel
_ReduceFromModelParallelRegion
c10d::allreduce_
nccl:all_reduce
ncclDevKernel_AllReduce_Sum_bf16_RING_LL,  # BFloat16 comms
ncclDevKernel_AllReduce_Sum_f32_RING_LL,   # Float32 comms
sum_and_scatter<c10::BFloat16, long>
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
