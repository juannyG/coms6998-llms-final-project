# Post processing scripts

After one or more experiments have been run, here are some scripts to show the results in a 
more readable fashion than looking through the raw logs.

## summary.py

```sh 
usage: summary.py [-h] [--files [FILES ...]] [--dir DIR] [--baseline BASELINE] {device,experiment,compare}

Provide metrics summary at different levels in a more concise and readable manner

positional arguments:
  {device,experiment,compare}  The type of summary you want produced

options:
  -h, --help           show this help message and exit
  --files [FILES ...]  List of files
  --dir DIR            Directory containing rank JSONL files.
  --baseline BASELINE
```

### Examples
```sh 
$ python summary.py --files ../path/to/logs/single_gpu/10m/1760664637/cuda_0.log -- device

=== Results for experiment: single_gpu/10m/1760664637/cuda_0 ===
| Metric              | Value               |
|---------------------|---------------------|
| Total Tokens        | 406,400             |
| Total Time          | 6.92 sec            |
| Total Throughput    | 58697.50 tokens/sec |
| Final Loss          | 9.0131              |
| Avg GPU Mem         | 248.1 MB            |
| Peak GPU Mem        | 616.4 MB            |
| Avg GPU Utilization | 51.69%              |


$ python tools/metrics/summary.py experiment --dir ../logs/tensor_parallel/10m/1763908350/

=== Aggregated Results for tensor_parallel/10m/1763908350 ===
| Metric                  | Value               |
|-------------------------|---------------------|
| Number of devices       | 2                   |
| Total Tokens            | 406,400             |
| Total Time              | 5.05 sec            |
| Total Throughput        | 80424.62 tokens/sec |
| Final Loss              | 8.3598              |
| Avg GPU Mem             | 140.5 MB            |
| Total avg GPU Mem       | 280.9 MB            |
| Peak GPU Mem            | 313.0 MB            |
| Total peak GPU Mem      | 626.0 MB            |
| Avg GPU Utilization     | 72.27%              |
| Min avg GPU Utilization | 64.22%              |


$ python tools/metrics/summary.py compare --baseline ../logs/single_gpu/10m/1763905923/cuda_0.log --dir ../logs/tensor_parallel/10m

=== Baseline metrics of single_gpu/10m
| Metric              | Value               |
|---------------------|---------------------|
| Total Tokens        | 406,400             |
| Total Time          | 4.35 sec            |
| Total Throughput    | 93459.55 tokens/sec |
| Final Loss          | 0.0313              |
| Avg GPU Mem         | 225.3 MB            |
| Peak GPU Mem        | 503.8 MB            |
| Avg GPU Utilization | 56.78%              |

=== Comparison Results of single_gpu/10m against tensor_parallel/10m/1763908350 (2 devices) ===
| Metric                        | Value               |
|-------------------------------|---------------------|
| Strategy                      | tensor_parallel     |
| Number of Devices             | 2                   |
| Total Time                    | 5.05 seconds        |
| Total Throughput              | 80424.62 tokens/sec |
| Avg GPU Mem                   | 140.46 MB           |
| Avg GPU Util %                | 72.27%              |
| Distributed Training Overhead | 13.95%              |
| Throughput Efficiency         | 86.05%              |
| Memory Scaling Factor         | 0.62                |

=== Comparison Results of single_gpu/10m against tensor_parallel/10m/1763908903 (4 devices) ===
| Metric                        | Value               |
|-------------------------------|---------------------|
| Strategy                      | tensor_parallel     |
| Number of Devices             | 4                   |
| Total Time                    | 15.86 seconds       |
| Total Throughput              | 25623.16 tokens/sec |
| Avg GPU Mem                   | 100.43 MB           |
| Avg GPU Util %                | 93.38%              |
| Distributed Training Overhead | 72.58%              |
| Throughput Efficiency         | 27.42%              |
| Memory Scaling Factor         | 0.45                |
```

```
