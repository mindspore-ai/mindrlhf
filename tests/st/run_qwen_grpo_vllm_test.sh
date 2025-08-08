#!/bin/bash
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

export MS_ENABLE_LCCL=off
export GLOG_v=2
WORKDIR="$(realpath "$(dirname "$0")")"
echo "WORKDIR is $WORKDIR"
cd $WORKDIR
export MINDRLHF_PATH=$WORKDIR/../../
export MINDFORMERS_PATH=$WORKDIR/mindformers/
export PYTHONPATH=$MINDRLHF_PATH:$MINDFORMERS_PATH:$PYTHONPATH
echo "PYTHONPATH is $PYTHONPATH"

# vllm config
export vLLM_MODEL_BACKEND=MindFormers
export HCCL_EXEC_TIMEOUT=7200
export MS_JIT_MODULES=vllm_mindspore,research
export MS_MEMORY_STATISTIC=1

mkdir -p $WORKDIR/dataset/
data_path="$WORKDIR/dataset/mini_gsm8k.mindrecord"

python ./qwen2_5/rlhf_data.py \
--vocab_path "$WORKDIR/qwen2_5_vllm/vocab.json" \
--merges_file_path "$WORKDIR/qwen2_5_vllm/merges.txt" \
--file_path "$WORKDIR/qwen2_5/mini_gsm8k.jsonl" \
--output_path $data_path > $WORKDIR/data_preprocess.log

TRAIN_OUTPUT_DIR=$WORKDIR/grpo_train_output
mkdir -p $TRAIN_OUTPUT_DIR

export MINDRLHF_TEST=1
if [ ! $DUMP_RECONSTRUCT_CONFIG_PATH ];then
  export DUMP_RECONSTRUCT_CONFIG_PATH=$TRAIN_OUTPUT_DIR/dump_configs
else
  echo "DUMP_RECONSTRUCT_CONFIG_PATH has been set to $DUMP_RECONSTRUCT_CONFIG_PATH"
fi

ACTOR_MODEL="./qwen2_5_vllm/finetune_qwen2_5_7b_st.yaml"
REF_MODEL="./qwen2_5_vllm/finetune_qwen2_5_7b_st.yaml"
GEN_MODEL="./qwen2_5_vllm/predict_qwen2_5_7b_instruct_st.yaml"

msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 \
--master_port=7475 --join=True --log_dir=$TRAIN_OUTPUT_DIR \
../../examples/grpo/qwen_grpo_tutorial/main.py \
  rl_config.dataset_file=$data_path \
  rl_config.tokenizer_dir=$WORKDIR/qwen2_5_vllm \
  rl_config.save_data_file="$TRAIN_OUTPUT_DIR/saved_data_file.mindrecord" \
  rl_config.num_generations=4 \
  rl_config.beta=0.01 \
  rl_config.num_rollouts=1 \
  actor_config.model_config=$ACTOR_MODEL \
  actor_config.load="" \
  actor_config.save="$TRAIN_OUTPUT_DIR/ckpt/train" \
  actor_config.parallel_config.micro_batch_num=2 \
  actor_config.parallel_config.vocab_emb_dp=False \
  actor_config.recompute_config.recompute=True \
  actor_config.enable_parallel_optimizer=True \
  actor_config.use_eod_attn_mask_compression=True \
  ref_config.model_config=$REF_MODEL \
  ref_config.load="" \
  ref_config.ref_model_batch_size=2 \
  ref_config.use_eod_attn_mask_compression=True \
  ref_config.parallel_config.data_parallel=4 \
  ref_config.parallel_config.model_parallel=2 \
  ref_config.parallel_config.pipeline_stage=1 \
  reward_config.verifier_function=["accuracy_reward","format_reward"] \
  generate_config.model_config=$GEN_MODEL \
  generate_config.load="$WORKDIR/qwen2_5_vllm" \
  generate_config.parallel_config.data_parallel=4 \
  generate_config.parallel_config.model_parallel=2 \
  generate_config.parallel_config.pipeline_stage=1 \
  generate_config.max_model_len=32768 \
  generate_config.max_num_batched_tokens=32768 \
  generate_config.gpu_memory_utilization=0.5 \
  generate_config.sampling_config.max_tokens=16 \
  generate_config.sampling_config.min_tokens=2 \
  generate_config.sampling_config.temperature=0.8 \
  generate_config.sampling_config.repetition_penalty=1.05 \
  generate_config.sampling_config.top_p=0.8 \
  generate_config.sampling_config.top_k=20 \
  context.max_device_memory="58GB"
