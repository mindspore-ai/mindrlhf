#!/bin/bash

export MS_ALLOC_CONF="memory_tracker:False,enable_vmm:false"
# export MS_DEV_DUMP_IR_PASSES="step_parallel"
# export ASCEND_GLOBAL_LOG_LEVEL=3
# export ASCEND_SLOG_PRINT_TO_STDOUT=1
export GLOG_v=3
export MS_DEV_DUMP_IR_PASSES="hwopt_d_after_stream_assign,valid,graph_build"
export MS_DEV_RUNTIME_CONF="memory_statistics:False"
export MS_ENABLE_LCCL=off  # 910C上必须得设成off

# root_path=/grpo/hx/zhouhongye/mindrlhf_2
root_path="$(realpath "$(dirname "$0")")"
cd $root_path

export PYTHONPATH=/grpo/hx/zhouhongye/mindformers:$PYTHONPATH  # 指定mindformers路径
export PYTHONPATH=$root_path:$PYTHONPATH  # 指定mindrlhf路径

ps -ef |grep -i python |grep -i [name] |grep -v grep |awk '{print $2}' |xargs -t -I {} kill -9 {}
ps -ef |grep -i msrun |grep -i [name] |grep -v grep |awk '{print $2}' |xargs -t -I {} kill -9 {}
rm -rf graph
rm -rf graph_infer
rm -rf graph_train
rm -rf qwen2_one_log
rm -rf stragegy merge_strategy

msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 \
--master_port=9190 --join=False --log_dir=./qwen2_one_log \
examples/qwen_grpo_tutorial/grpo_one_stage.py \
--sft_path_infer ./model_configs/qwen_grpo/predict_qwen2_7b_instruct.yaml \
--sft_path_train ./model_configs/qwen_grpo/finetune_qwen2_7b.yaml \
--vocab_path /path/to/vocab.json \
--merges_file_path /path/to/merges.txt \
--mind_dataset_dir /path/to/grpo_data/cvalues_one.mindrecord \
--save_data_file /path/to/grpo_data/grpo_1024.mindrecord \
--save_ckpt_dir /path/to/ckpt/train \
--use_parallel True \
--load_sft_checkpoint_infer "" \
--load_sft_checkpoint_train "" \
--enable_compile_cache False \

# msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 \
# --master_port=9190 --join=False --log_dir=./qwen2_one_log \
# examples/qwen_grpo_tutorial/grpo_one_stage.py \
# --sft_path_infer ./model_configs/qwen_grpo/predict_qwen2_7b_instruct.yaml \
# --sft_path_train ./model_configs/qwen_grpo/finetune_qwen2_7b.yaml \
# --vocab_path /grpo/hx/qwen25-32b/vocab.json \
# --merges_file_path /grpo/hx/qwen25-32b/merges.txt \
# --mind_dataset_dir /grpo/hx/merge0222/grpo_data/cvalues_one.mindrecord \
# --save_data_file /grpo/hx/merge0222/grpo_data/grpo_1024.mindrecord \
# --save_ckpt_dir /grpo/hx/merge0222/ckpt/train \
# --use_parallel True \
# --load_sft_checkpoint_infer "" \
# --load_sft_checkpoint_train "" \
# --enable_compile_cache False \