PYTHON=python
TRAIN_SCRIPT=run_experiment.py
SIMPLE_SINGLE_GPU_EXPERIMENT=simple_single_gpu
SINGLE_GPU_EXPERIMENT=tensor_parallel
DDP_EXPERIMENT=megatron_ddp
PP_EXPERIMENT=megatron_pipeline_parallel
TP_EXPERIMENT=tensor_parallel
SIMPLE_ZERO_EXPERIMENT=simple_zero
LOG_PATH=../logs

.NOTPARALLEL:
.PHONY: all single tp ddp pp simple_single simple_zero
all: single ddp tp pp simple_single simple_zero

clean:
	rm -rf $(LOG_PATH)/*

single: \
    single_10m \
    single_100m \
    single_300m \
    single_500m \
    single_1b

single_%:
	mkdir $(LOG_PATH)/single_gpu; \
		RUN_ID=$$(date +%s) torchrun --nproc-per-node=1 $(TRAIN_SCRIPT) $(SINGLE_GPU_EXPERIMENT) $*; \
		mv $(LOG_PATH)/tensor_parallel/$* $(LOG_PATH)/single_gpu/

ddp: ddp2 ddp4

ddp2: \
    ddp2_10m \
    ddp2_100m \
    ddp2_300m \
    ddp2_500m \
    ddp2_1b

ddp2_%:
	RUN_ID=$$(date +%s) torchrun --nproc-per-node=2 $(TRAIN_SCRIPT) $(DDP_EXPERIMENT) $*

ddp4: \
    ddp4_10m \
    ddp4_100m \
    ddp4_300m \
    ddp4_500m \
    ddp4_1b

ddp4_%:
	RUN_ID=$$(date +%s) torchrun --nproc-per-node=4 $(TRAIN_SCRIPT) $(DDP_EXPERIMENT) $*


tp: tp2 tp4

tp2: \
    tp2_10m \
    tp2_100m \
    tp2_300m \
    tp2_500m \
    tp2_1b

tp2_%:
	RUN_ID=$$(date +%s) torchrun --nproc-per-node=2 $(TRAIN_SCRIPT) $(TP_EXPERIMENT) $*

tp4: \
    tp4_10m \
    tp4_100m \
    tp4_300m \
    tp4_500m \
    tp4_1b

tp4_%:
	RUN_ID=$$(date +%s) torchrun --nproc-per-node=4 $(TRAIN_SCRIPT) $(TP_EXPERIMENT) $*

pp: pp2 pp4

pp2: \
    pp2_10m \
    pp2_100m \
    pp2_300m \
    pp2_500m \
    pp2_1b

pp2_%:
	RUN_ID=$$(date +%s) torchrun --nproc-per-node=2 $(TRAIN_SCRIPT) $(PP_EXPERIMENT) $*

pp4: \
    pp4_10m \
    pp4_100m \
    pp4_300m \
    pp4_500m \
    pp4_1b

pp4_%:
	RUN_ID=$$(date +%s) torchrun --nproc-per-node=4 $(TRAIN_SCRIPT) $(PP_EXPERIMENT) $*

###########################################################
# Experiments that use SimpleTransformerDecoder
# Single GPU baseline + ZeRO stage 1 / 2 / 3 / 3-offload
###########################################################
simple_single: \
    simple_single_10m \
    simple_single_100m \
    simple_single_300m \
    simple_single_500m \
    simple_single_1b

# TODO EXPAND SO 300m/500m/1b use ZeRO config!
simple_single_%:
	RUN_ID=$$(date +%s) python $(TRAIN_SCRIPT) $(SIMPLE_SINGLE_GPU_EXPERIMENT) zero-$*

simple_zero: simple_zero1 simple_zero2 simple_zero3 simple_zero3_offload

# ZeRO Stage 1
simple_zero1: simple_zero1_2gpu simple_zero1_4gpu

simple_zero1_2gpu: \
	simple_zero1_2gpu_10m \
	simple_zero1_2gpu_100m \
	simple_zero1_2gpu_300m \
	simple_zero1_2gpu_500m \
	simple_zero1_2gpu_1b

simple_zero1_2gpu_%:
	RUN_ID=$$(date +%s) ZERO_CONFIG=zero_configs/zero_2g_stage1.yaml torchrun --nproc-per-node=2 $(TRAIN_SCRIPT) $(SIMPLE_ZERO_EXPERIMENT) zero-$* --experiment-subtype stage1

simple_zero1_4gpu: \
	simple_zero1_4gpu_10m \
	simple_zero1_4gpu_100m \
	simple_zero1_4gpu_300m \
	simple_zero1_4gpu_500m \
	simple_zero1_4gpu_1b

simple_zero1_4gpu_%:
	RUN_ID=$$(date +%s) ZERO_CONFIG=zero_configs/zero_2g_stage1.yaml torchrun --nproc-per-node=4 $(TRAIN_SCRIPT) $(SIMPLE_ZERO_EXPERIMENT) zero-$* --experiment-subtype stage1

# ZeRO Stage 2
simple_zero2: simple_zero2_2gpu simple_zero2_4gpu

simple_zero2_2gpu: \
	simple_zero2_2gpu_10m \
	simple_zero2_2gpu_100m \
	simple_zero2_2gpu_300m \
	simple_zero2_2gpu_500m \
	simple_zero2_2gpu_1b

simple_zero2_2gpu_%:
	RUN_ID=$$(date +%s) ZERO_CONFIG=zero_configs/zero_2g_stage2.yaml torchrun --nproc-per-node=2 $(TRAIN_SCRIPT) $(SIMPLE_ZERO_EXPERIMENT) zero-$* --experiment-subtype stage2

simple_zero2_4gpu: \
	simple_zero2_4gpu_10m \
	simple_zero2_4gpu_100m \
	simple_zero2_4gpu_300m \
	simple_zero2_4gpu_500m \
	simple_zero2_4gpu_1b

simple_zero2_4gpu_%:
	RUN_ID=$$(date +%s) ZERO_CONFIG=zero_configs/zero_2g_stage2.yaml torchrun --nproc-per-node=4 $(TRAIN_SCRIPT) $(SIMPLE_ZERO_EXPERIMENT) zero-$* --experiment-subtype stage2

# ZeRO Stage 3
simple_zero3: simple_zero3_2gpu simple_zero3_4gpu

simple_zero3_2gpu: \
	simple_zero3_2gpu_10m \
	simple_zero3_2gpu_100m \
	simple_zero3_2gpu_300m \
	simple_zero3_2gpu_500m \
	simple_zero3_2gpu_1b

simple_zero3_2gpu_%:
	RUN_ID=$$(date +%s) ZERO_CONFIG=zero_configs/zero_2g_stage3.yaml torchrun --nproc-per-node=2 $(TRAIN_SCRIPT) $(SIMPLE_ZERO_EXPERIMENT) zero-$* --experiment-subtype stage3

simple_zero3_4gpu: \
	simple_zero3_4gpu_10m \
	simple_zero3_4gpu_100m \
	simple_zero3_4gpu_300m \
	simple_zero3_4gpu_500m \
	simple_zero3_4gpu_1b

simple_zero3_4gpu_%:
	RUN_ID=$$(date +%s) ZERO_CONFIG=zero_configs/zero_2g_stage3.yaml torchrun --nproc-per-node=4 $(TRAIN_SCRIPT) $(SIMPLE_ZERO_EXPERIMENT) zero-$* --experiment-subtype stage3

# ZeRO Stage 3 + Offload
simple_zero3_offload: simple_zero3_offload_2gpu simple_zero3_offload_4gpu

simple_zero3_offload_2gpu: \
	simple_zero3_offload_2gpu_10m \
	simple_zero3_offload_2gpu_100m \
	simple_zero3_offload_2gpu_300m \
	simple_zero3_offload_2gpu_500m \
	simple_zero3_offload_2gpu_1b

simple_zero3_offload_2gpu_%:
	RUN_ID=$$(date +%s) ZERO_CONFIG=zero_configs/zero_2g_stage3.yaml torchrun --nproc-per-node=2 $(TRAIN_SCRIPT) $(SIMPLE_ZERO_EXPERIMENT) zero-$* --experiment-subtype stage3_offload

simple_zero3_offload_4gpu: \
	simple_zero3_offload_4gpu_10m \
	simple_zero3_offload_4gpu_100m \
	simple_zero3_offload_4gpu_300m \
	simple_zero3_offload_4gpu_500m \
	simple_zero3_offload_4gpu_1b

simple_zero3_offload_4gpu_%:
	RUN_ID=$$(date +%s) ZERO_CONFIG=zero_configs/zero_2g_stage3.yaml torchrun --nproc-per-node=4 $(TRAIN_SCRIPT) $(SIMPLE_ZERO_EXPERIMENT) zero-$* --experiment-subtype stage3_offload

# Metrics runners
metrics_ddp: \
	metrics_ddp_10m \
	metrics_ddp_100m \
	metrics_ddp_300m \
	metrics_ddp_500m \
	metrics_ddp_1b

metrics_tp: \
	metrics_tp_10m \
	metrics_tp_100m \
	metrics_tp_300m \
	metrics_tp_500m \
	metrics_tp_1b

metrics_pp: \
	metrics_pp_10m \
	metrics_pp_100m \
	metrics_pp_300m \
	metrics_pp_500m \
	metrics_pp_1b

metrics_ddp_%:
	python tools/metrics/summary.py compare --baseline $(LOG_PATH)/single_gpu/$*/*/cuda_0.log --dir $(LOG_PATH)/megatron_ddp/$*

metrics_tp_%:
	python tools/metrics/summary.py compare --baseline $(LOG_PATH)/single_gpu/$*/*/cuda_0.log --dir $(LOG_PATH)/tensor_parallel/$*

metrics_pp_%:
	python tools/metrics/summary.py compare --baseline $(LOG_PATH)/single_gpu/$*/*/cuda_0.log --dir $(LOG_PATH)/megatron_pipeline_parallel/$*

#metrics_zero_%:
#	python tools/metrics/summary.py compare --baseline $(LOG_PATH)/single_gpu/$*/*/cuda_0.log --dir $(LOG_PATH)/TODO/$*



