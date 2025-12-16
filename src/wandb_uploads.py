import wandb
import pandas as pd
import os

ENTITY = "jmg2048-columbia-university"
PROJECT = "fall25-sllm-final-project"


def upload_megatron():
    group_name = "tp-ddp-pp"
    csv_path = "../results/single-tp-dpp-pp-experiment-results.csv"

    df = pd.read_csv(csv_path)

    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        name="megatron_raw_table",
        job_type="raw_data",
        reinit=True,
    )
    wandb.log({"megatron_raw_table": wandb.Table(dataframe=df)})
    run.finish()

    strategy_labels = {
        "megatron_ddp": "Data Parallel",
        "tensor_parallel": "Tensor Parallel",
        "megatron_pipeline_parallel": "Pipeline Parallel",
    }

    plot_metrics = {
        "throughput_tokens_sec": "Throughput",
        "throughput_efficiency_percent": "Efficiency",
        "relative_runtime_overhead_percent": "Runtime Overhead",
        "peak_gpu_mem_mb": "Peak GPU Memory",
        "avg_gpu_mem_mb": "Average GPU Memory",
        "avg_gpu_util_percent": "Average GPU Utilization",
    }

    for model_size in df["model_size"].unique():
        model_df = df[df["model_size"] == model_size]

        run = wandb.init(
            project=PROJECT,
            entity=ENTITY,
            name=f"megatron_{model_size}",
            job_type="megatron_plots",
            group=group_name,
            config={"model_size": model_size},
            reinit=True,
        )

        for num_devices in sorted(model_df["num_devices"].unique()):
            step_metrics = {}

            for _, row in model_df[model_df["num_devices"] == num_devices].iterrows():
                strategy = strategy_labels.get(row["strategy"], row["strategy"])

                for csv_key, label in plot_metrics.items():
                    if pd.notna(row.get(csv_key)):
                        step_metrics[f"{strategy}/{label}"] = row[csv_key]

            if step_metrics:
                wandb.log(step_metrics, step=int(num_devices))
        run.finish()


def upload_zero():
    label_to_stage_mapping = {
        "simple_single_gpu": "Baseline",
        "simple_zero_stage1": "Stage 1",
        "simple_zero_stage2": "Stage 2",
        "simple_zero_stage3": "Stage 3",
        "simple_zero_stage3_offload": "Stage 3 Offload",
    }

    stage_order = {
        "Baseline": 0,
        "Stage 1": 1,
        "Stage 2": 2,
        "Stage 3": 3,
        "Stage 3 Offload": 4,
    }

    group_name = "simple-single-and-zero"
    csv_path = "../results/simple-single-and-zero-experiment-results.csv"

    df = pd.read_csv(csv_path)
    model_sizes = df["model_size"].unique()
    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        name="zero_raw_table",
        job_type="raw_data",
        reinit=True,
    )
    wandb.log({"zero_raw_table": wandb.Table(dataframe=df)})
    run.finish()

    for model_size in model_sizes:
        model_df = df[df["model_size"] == model_size]
        model_df["stage"] = model_df["strategy"].map(label_to_stage_mapping)
        model_df["stage_idx"] = model_df["stage"].map(stage_order)

        run = wandb.init(
            project=PROJECT,
            entity=ENTITY,
            name=f"zero_{model_size}",
            group=group_name,
            job_type="zero_metrics",
            config={"model_size": model_size, "x_axis": "zero_stage"},
            reinit=True,
        )

        # IMPORTANT: iterate over stage FIRST
        for stage_idx in sorted(model_df["stage_idx"].unique()):
            stage_df = model_df[model_df["stage_idx"] == stage_idx]
            step_metrics = {}

            for _, row in stage_df.iterrows():
                num_devices = row["num_devices"]
                for k in row.keys():
                    if k == "num_devices":
                        continue
                    step_metrics.update({f"zero_{num_devices}_gpu/{k}": row[k]})

            wandb.log(step_metrics, step=int(stage_idx))

        run.summary["zero_data_table"] = wandb.Table(dataframe=model_df)
        run.summary["zero_stage_mapping"] = stage_order
        run.finish()


def upload_pp_sidequest():
    df = pd.read_csv("../results/pp-microbatch-side-quest.csv")
    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        name="pp_microbatch_side_quest",
        job_type="analysis",
        reinit=True,
    )
    table = wandb.Table(dataframe=df)

    wandb.log({"pp_microbatch_table": table})
    run.finish()


if __name__ == "__main__":
    upload_megatron()
    upload_zero()
    upload_pp_sidequest()
