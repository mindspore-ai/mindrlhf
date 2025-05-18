#!/bin/bash

export MS_ALLOC_CONF="memory_tracker:False,enable_vmm:True"
export GLOG_v=2

export vLLM_MODEL_BACKEND=MindFormers
export vLLM_MODEL_MEMORY_USE_GB=40
export MINDFORMERS_MODEL_CONFIG=/path/to/configs/predict_qwen2_5_7b_instruct.yaml
export HCCL_EXEC_TIMEOUT=7200
export MS_JIT_MODULES=vllm_mindspore,research


msrun --worker_num=8 --local_worker_num=8 \
--master_addr=127.0.0.1 --master_port=9887 \
--join=True --log_dir=./prof_vllm_log \
examples/grpo/qwen_grpo_tutorial/main.py \
--config examples/grpo/qwen_grpo_tutorial/grpo_config.yaml \
--tokenizer_dir /path/to/configs/ \
--dataset_file /path/to/limr_template_qwenr1.mindrecord \
--save_checkpoint_dir /path/to/train \
--actor_checkpoint_path "/path/to/qwen25_7b" \
--ref_checkpoint_path "/path/to/qwen25_7b" \
--generate_checkpoint_path "/path/to/qwen25_7b" > vllm.log 2>&1 &
