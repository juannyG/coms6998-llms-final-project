# Post processing scripts

After one or more experiments have been run, here are some scripts to show the results in a 
more readable fashion than looking through the raw logs.

## summarize_metrics.py

```sh 
usage: summarize_metrics.py [-h] [--files [FILES ...]] [--dir DIR] {training,profiler}

Provide per-device metrics summary in a more concise and readable manner.

positional arguments:
  {training,profiler}  The type of metric to extract and summarize.

options:
  -h, --help           show this help message and exit
  --files [FILES ...]  List of files
  --dir DIR            Directory containing rank JSONL files.
```

### Examples
```sh 
$ python summarize_metrics.py --files ../path/to/logs/run_single_gpu_10m_cuda_0_1760664637.log -- training

=== Results for experiment: run_single_gpu_10m_cuda_0_1760664637 ===
| Metric          | Value    |
|-----------------|----------|
| Avg Tokens/sec  | 145,976  |
| Avg Samples/sec | 1,149.4  |
| Avg Loss        | 9.0635   |
| Total Tokens    | 406,400  |
| Total Time      | 3.08 sec |
| Peak GPU Mem    | 617.6 MB |
| GPU Utilization | 83%      |


$ python summarize_metrics.py --dir ../../../logs -- profiler

=== Results for experiment: run_single_gpu_10m_cuda_0_1760664637 ===
| Operation            |   Calls |   CPU Time (ms) |   GPU Time (ms) |   CPU Memory (MB) |   GPU Memory (MB) |
|----------------------|---------|-----------------|-----------------|-------------------|-------------------|
| model_forward        |       8 |           43.3  |           68.96 |                 0 |           1817.03 |
| model_loss           |       8 |            1.12 |            3.36 |                 0 |            496.1  |
| model_backward       |       8 |           69.1  |            0.02 |                -0 |              0    |
| model_optimizer_step |       8 |            8.49 |           10.42 |                 0 |              0    |
```
