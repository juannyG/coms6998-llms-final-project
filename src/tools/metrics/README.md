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


$ python summary.py compare \
 --baseline ../path/to/logs/single_gpu/10m/1763603317/cuda_0.log \
 --dir ../path/to/logs/torch_ddp/10m

=== Baseline metrics of single_gpu/10m
| Metric              | Value               |
|---------------------|---------------------|
| Total Tokens        | 406,400             |
| Total Time          | 6.92 sec            |
| Total Throughput    | 58697.50 tokens/sec |
| Final Loss          | 9.0131              |
| Avg GPU Mem         | 248.1 MB            |
| Peak GPU Mem        | 616.4 MB            |
| Avg GPU Utilization | 51.69%              |

=== Comparison Results of single_gpu/10m against torch_ddp/10m/1763605017 (2 devices) ===
| Metric                 | Value               |
|------------------------|---------------------|
| Strategy               | torch_ddp           |
| Number of Devices      | 2                   |
| Total Time             | 7.42 seconds        |
| Total Throughput       | 54795.42 tokens/sec |
| Avg GPU Mem            | 260.64 MB           |
| Avg GPU Util %         | 40.42%              |
| Communication Overhead | 6.65%               |
| Throughput Efficiency  | 93.35%              |
| Memory Scaling Factor  | 2.10                |

=== Comparison Results of single_gpu/10m against torch_ddp/10m/1763605149 (4 devices) ===
| Metric                 | Value               |
|------------------------|---------------------|
| Strategy               | torch_ddp           |
| Number of Devices      | 4                   |
| Total Time             | 7.59 seconds        |
| Total Throughput       | 53543.19 tokens/sec |
| Avg GPU Mem            | 243.75 MB           |
| Avg GPU Util %         | 41.78%              |
| Communication Overhead | 8.78%               |
| Throughput Efficiency  | 91.22%              |
| Memory Scaling Factor  | 3.93                |
```
```

