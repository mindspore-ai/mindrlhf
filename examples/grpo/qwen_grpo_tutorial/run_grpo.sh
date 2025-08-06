#!/bin/bash

export MS_ALLOC_CONF="memory_tracker:False,enable_vmm:True"
export GLOG_v=2

export vLLM_MODEL_BACKEND=MindFormers
export HCCL_EXEC_TIMEOUT=7200
export MS_JIT_MODULES=vllm_mindspore,research
export MS_ENABLE_LCCL=off

root_path="$(realpath "$(dirname "$0")")"
root_path=$root_path/../../../
cd $root_path
export PYTHONPATH=$root_path:$PYTHONPATH  # define mindrlhf path

TRAIN_OUTPUT="train_output"
mkdir $TRAIN_OUTPUT

export MINDFORMERS_PATH=/path/to/mindformers # need modify
export MSADAPTER_PATH=/path/to/msadapter # need modify (msadapter lib path)
export QWEN_MODEL_PATH=/path/to/qwen25_7b # need modify (only support huggingface model path)
export DATASET_FILE=/path/to/limr_template_qwenr1.mindrecord # need modify

export PYTHONPATH=$MSADAPTER_PATH:$MINDFORMERS_PATH:$PYTHONPATH

ACTOR_MODEL="./model_configs/qwen_grpo/qwen2_5_7b/finetune_qwen2_5_7b.yaml"
REF_MODEL="./model_configs/qwen_grpo/qwen2_5_7b/finetune_qwen2_5_7b.yaml"
GEN_MODEL="./model_configs/qwen_grpo/qwen2_5_7b/predict_qwen2_5_7b_instruct.yaml"

msrun --worker_num=8 --local_worker_num=8 \
--master_addr=127.0.0.1 --master_port=9887 \
--join=True --log_dir=./prof_vllm_log \
examples/grpo/qwen_grpo_tutorial/main.py \
  rl_config.tokenizer_dir=$QWEN_MODEL_PATH \
  rl_config.dataset_file=$DATASET_FILE \
  rl_config.beta=0.01 \
  rl_config.num_rollouts=4 \
  rl_config.save_prompt_completions_data=True \
  rl_config.save_prompt_completions_dir="./$TRAIN_OUTPUT/saved_prompt_completions" \
  rl_config.save_data_file="./$TRAIN_OUTPUT/saved_data_file" \
  actor_config.model_config=$ACTOR_MODEL \
  actor_config.load=$QWEN_MODEL_PATH \
  actor_config.save="./$TRAIN_OUTPUT/saved_model" \
  actor_config.parallel_config.micro_batch_num=2 \
  actor_config.parallel_config.vocab_emb_dp=False \
  actor_config.recompute_config.recompute=True \
  actor_config.enable_parallel_optimizer=True \
  actor_config.use_eod_attn_mask_compression=True \
  ref_config.model_config=$REF_MODEL \
  ref_config.load=$QWEN_MODEL_PATH \
  ref_config.ref_model_batch_size=2 \
  ref_config.use_eod_attn_mask_compression=True \
  reward_config.verifier_function=["qwen_accuracy_reward","format_reward"] \
  reward_config.verifier_weight=[1.0,1.0] \
  generate_config.model_config=$GEN_MODEL \
  generate_config.load=$QWEN_MODEL_PATH \
  generate_config.infer_model_batch_size=2 \
  generate_config.max_model_len=32768 \
  generate_config.max_num_batched_tokens=32768 \
  generate_config.gpu_memory_utilization=0.5 \
  generate_config.sampling_config.temperature=0.8 \
  generate_config.sampling_config.repetition_penalty=1.05 \
  generate_config.sampling_config.top_p=0.8 \
  generate_config.sampling_config.top_k=20 > vllm.log 2>&1 &
