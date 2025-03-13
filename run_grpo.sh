#!/bin/bash

export MS_ALLOC_CONF="memory_tracker:False,enable_vmm:false"
export MS_DEV_DUMP_IR_PASSES="hwopt_d_after_stream_assign,valid,graph_build"
export MS_DEV_RUNTIME_CONF="memory_statistics:False"
export MS_ENABLE_LCCL=off
export GLOG_v=3

root_path="$(realpath "$(dirname "$0")")"
cd $root_path

export PYTHONPATH=/path/to/mindformers:$PYTHONPATH

msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 \
--master_port=9190 --join=False --log_dir=./qwen2_one_log \
examples/grpo/qwen_grpo_tutorial/grpo_one_stage.py \
--sft_path_infer ./model_configs/qwen_grpo/predict_qwen2_5_7b_instruct.yaml \
--sft_path_train ./model_configs/qwen_grpo/finetune_qwen2_5_7b.yaml \
--vocab_path /path/to/vocab.json \
--merges_file_path /path/to/merges.txt \
--mind_dataset_dir /path/to/grpo_data/cvalues_one.mindrecord \
--save_data_file /path/to/grpo_data/grpo_1024.mindrecord \
--save_ckpt_dir /path/to/ckpt/train \
--use_parallel True \
--load_sft_checkpoint_infer "/path/to/ckpt/infer" \
--load_sft_checkpoint_train "/path/to/ckpt/train" \
--load_ref_checkpoint "/path/to/ckpt/infer" \
--enable_compile_cache False