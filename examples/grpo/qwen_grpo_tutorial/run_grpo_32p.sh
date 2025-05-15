#!/bin/bash

export MS_ALLOC_CONF="memory_tracker:False,enable_vmm:True"
export GLOG_v=2
export MS_DEV_DUMP_IR_PASSES="hwopt_d_after_stream_assign,valid,graph_build"
export MS_DEV_RUNTIME_CONF="memory_statistics:False"
export MS_ENABLE_LCCL=off

root_path="$(realpath "$(dirname "$0")")"
root_path=$root_path/../../../
cd $root_path
WORKDIR=$PWD/../
export PYTHONPATH=$root_path:$PYTHONPATH  # 指定mindrlhf路径
export MINDFORMERS_PATH=$WORKDIR/mindformers/

noderank=$1
master_addr=$2
echo $noderank
msrun --worker_num=32 --local_worker_num=8 --node_rank=$noderank --master_addr=$master_addr \
--master_port=9190 --join=False --log_dir=./qwen2_one_log_$noderank \
examples/grpo/qwen_grpo_tutorial/main.py \
--config examples/grpo/qwen_grpo_tutorial/grpo_config.yaml \
--sft_path_infer model_configs/qwen_grpo/predict_qwen2_5_32b_instruct_481.yaml \
--sft_path_ref model_configs/qwen_grpo/predict_qwen2_5_32b_instruct_481.yaml \
--sft_path_train model_configs/qwen_grpo/finetune_qwen2_5_32b_8k_184.yaml \
--vocab_path /path/to/vocab.json \
--merges_file_path /path/to/merges.txt \
--mind_dataset_dir /path/to/grpo_data/cvalues_one.mindrecord \
--save_data_file ./grpo_1024_$noderank.mindrecord \
--save_ckpt_dir /path/to/ckpt/train \
--use_parallel True \
--load_sft_checkpoint_infer "/path/to/ckpt/infer" \
--load_sft_checkpoint_train "/path/to/ckpt/train/" \
--load_ref_checkpoint "/path/to/ckpt/infer" \
--enable_compile_cache False \
