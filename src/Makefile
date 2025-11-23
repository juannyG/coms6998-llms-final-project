PYTHON=python
TRAIN_SCRIPT=run_experiment.py
SINGLE_GPU_EXPERIMENT=tensor_parallel
DDP_EXPERIMENT=megatron_ddp
PP_EXPERIMENT=megatron_pipeline_parallel # STILL NEEDS PROFILER ADDITION
TP_EXPERIMENT=tensor_parallel
LOG_PATH=../logs
#ZERO_EXPERIMENT=zero

.NOTPARALLEL:
.PHONY: all single tp ddp pp zero
all: single ddp tp pp zero

clean:
	rm -rf $(LOG_PATH)/*

single: \
    single_10m \
    single_100m \
    single_300m \
    single_500m \
    single_1b

single_%:
	RUN_ID=$$(date +%s); \
				 mkdir $(LOG_PATH)/single_gpu; \
				 torchrun --nproc-per-node=1 $(TRAIN_SCRIPT) $(SINGLE_GPU_EXPERIMENT) $*; \
				 mv $(LOG_PATH)/tensor_parallel/$* $(LOG_PATH)/single_gpu/

ddp: ddp2 ddp4

ddp2: \
    ddp2_10m \
    ddp2_100m \
    ddp2_300m \
    ddp2_500m \
    ddp2_1b

ddp2_%:
	RUN_ID=$$(date +%s); \
				 torchrun --nproc-per-node=2 $(TRAIN_SCRIPT) $(DDP_EXPERIMENT) $*

ddp4: \
    ddp4_10m \
    ddp4_100m \
    ddp4_300m \
    ddp4_500m \
    ddp4_1b

ddp4_%:
	RUN_ID=$$(date +%s); \
				torchrun --nproc-per-node=4 $(TRAIN_SCRIPT) $(DDP_EXPERIMENT) $*


tp: tp2 tp4

tp2: \
    tp2_10m \
    tp2_100m \
    tp2_300m \
    tp2_500m \
    tp2_1b

tp2_%:
	RUN_ID=$$(date +%s); \
				 torchrun --nproc-per-node=2 $(TRAIN_SCRIPT) $(TP_EXPERIMENT) $*

tp4: \
    tp4_10m \
    tp4_100m \
    tp4_300m \
    tp4_500m \
    tp4_1b

tp4_%:
	RUN_ID=$$(date +%s); \
				 torchrun --nproc-per-node=4 $(TRAIN_SCRIPT) $(TP_EXPERIMENT) $*

pp: pp2 pp4

pp2: \
    pp2_10m \
    pp2_100m \
    pp2_300m \
    pp2_500m \
    pp2_1b

pp2_%:
	RUN_ID=$$(date +%s); \
				 torchrun --nproc-per-node=2 $(TRAIN_SCRIPT) $(PP_EXPERIMENT) $*

pp4: \
    pp4_10m \
    pp4_100m \
    pp4_300m \
    pp4_500m \
    pp4_1b

pp4_%:
	RUN_ID=$$(date +%s); \
				 torchrun --nproc-per-node=4 $(TRAIN_SCRIPT) $(PP_EXPERIMENT) $*


# Need the other branch
#
#zero: zero2 zero3
#
#zero2: \
#    zero2_4gpu_10m \
#    zero2_4gpu_100m \
#    zero2_4gpu_300m \
#    zero2_4gpu_500m \
#    zero2_4gpu_1b
#
#zero2_4gpu_%:
# RUN_ID=$$(date +%s); \
# 		ZERO_CONFIG=zero_configs/zero_2g_stage1.yaml; \
# 		torchrun --nproc-per-node=4 $(TRAIN_SCRIPT) $(ZERO_EXPERIMENT) $*
#
#zero3: \
#    zero3_4gpu_10m \
#    zero3_4gpu_100m \
#    zero3_4gpu_300m \
#    zero3_4gpu_500m \
#    zero3_4gpu_1b
#
#zero3_4gpu_%:
# RUN_ID=$$(date +%s); \
# 		ZERO_CONFIG=zero_configs/zero_3g_stage1.yaml; \
# 		torchrun --nproc-per-node=4 $(TRAIN_SCRIPT) $(ZERO_EXPERIMENT) $*

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

metrics_tp_%:
	python tools/metrics/summary.py compare --baseline $(LOG_PATH)/single_gpu/$*/*/cuda_0.log --dir $(LOG_PATH)/tensor_parallel/$*

metrics_ddp_%:
	python tools/metrics/summary.py compare --baseline $(LOG_PATH)/single_gpu/$*/*/cuda_0.log --dir $(LOG_PATH)/megatron_ddp/$*

metrics_pp_%:
	python tools/metrics/summary.py compare --baseline $(LOG_PATH)/single_gpu/$*/*/cuda_0.log --dir $(LOG_PATH)/megatron_pipeline_parallel/$*

#metrics_zero_%:
#	python tools/metrics/summary.py compare --baseline $(LOG_PATH)/single_gpu/$*/*/cuda_0.log --dir $(LOG_PATH)/TODO/$*



