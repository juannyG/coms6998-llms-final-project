"""
These dataclasses are meant to act as datamodel schemas for converting our
different result types into things we can use across aggregation scripts.
"""

from dataclasses import dataclass, field


@dataclass
class TrainingResults:
    avg_tokens_per_s: float = 0.0
    avg_samples_per_s: float = 0.0
    avg_loss: float = 0.0
    total_tokens: int = 0
    total_time_s: float = 0.0
    peak_gpu_mem_mb: float = 0.0
    gpu_util_percent: float = 0.0

    @classmethod
    def from_dict(cls, d):
        return cls(
            avg_tokens_per_s=d["avg_tokens_per_s"],
            avg_samples_per_s=d["avg_samples_per_s"],
            avg_loss=d["avg_loss"],
            total_tokens=d["total_tokens"],
            total_time_s=d["total_time_s"],
            peak_gpu_mem_mb=d["peak_gpu_mem_mb"],
            gpu_util_percent=d.get("gpu_util_percent"),
        )


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


@dataclass
class ProfilerSummary:
    # operations[op_label] = ProfilerOperationSummary
    operations: dict = field(default_factory=dict)

    def update_from_profiler_metrics(self, records, operation_labels):
        self.operations = {
            op_label: ProfilerOperationSummary(operation=op_label)
            for op_label in operation_labels
        }

        for r in records:
            op = r["operation"]
            d_type = r["device_type"]
            self.operations[op].calls = r["count"]
            if d_type == "DeviceType.CPU":
                self.operations[op].add_cpu_metrics(r)
            elif d_type == "DeviceType.CUDA":
                self.operations[op].add_cuda_metrics(r)
