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
"""
    run grpo one stage
"""

import time
import os
import argparse

import mindspore as ms
from mindspore import context
from mindspore.communication.management import get_rank, get_group_size

from mindformers import MindFormerConfig
from mindformers import LlamaConfig
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config
from mindrlhf.utils import TransformParametersD2D

from mindrlhf.trainer.grpo_trainer import GRPOTrainer
from mindrlhf.configs.grpo_configs import GRPOConfig
from mindrlhf.utils.configs import (
    combine_grpo_config,
    init_grpo_configs,
    init_grpo_dataset,
    init_grpo_network_and_optimizer,
    format_time_delta,
    init_reshard
)
from mindrlhf.utils import transfer_from_str_to_bool
from mindrlhf.models.qwen2.qwen2_tokenizer import Qwen2Tokenizer
from mindrlhf.reward.reward_fn import reward_func_from_jiaoda, accuracy_reward, format_reward


# pylint: disable=W0212
def main(sft_path_infer, sft_path_train, use_parallel, args):
    """
    grpo one stage
    """
    # init context
    enable_compile_cache = transfer_from_str_to_bool(args.enable_compile_cache)
    grpo_config, sft_config_infer, sft_model_config_infer, sft_model_config_train, ref_model_config = \
        init_grpo_configs(args)
    build_context(sft_config_infer)
    build_parallel_config(sft_config_infer)
    context.set_context(
        enable_compile_cache=enable_compile_cache,
        compile_cache_path="./generate_cache",
    )
    rank_id = get_rank() if use_parallel else 0

    tokenizer = Qwen2Tokenizer(args.vocab_path, args.merges_file_path, add_bos_token=False, add_eos_token=False)
    trainer = GRPOTrainer(
        grpo_config=grpo_config,
        sft_model_config_infer=sft_model_config_infer,
        sft_model_config_train=sft_model_config_train,
        ref_model_config=ref_model_config,
        reward_funcs=[accuracy_reward, format_reward],
        tokenizer=tokenizer,
    )
    trainer.grpo_model_infer.grpo_model.policy_model.model.add_flags_recursive(is_first_iteration=True)
    trainer.make_experience(num_generations=args.pre_num_generations, rank_id=rank_id, pre_run_flag=True)
    sample = trainer.store[0]
    trainer.store = [sample for _ in range(args.pre_store_data)]
    dataset = init_grpo_dataset(trainer)
    grpo_with_grad = init_grpo_network_and_optimizer(trainer, dataset)
    trainer.load_checkpoint()
    # reward 
    reshard_param, reshard_param_policy2ref = init_reshard(trainer)
    for n in range(grpo_config.epochs):
        # do generate
        steps = trainer.prompt_dataset.get_dataset_size() // trainer.prompt_dataset.get_batch_size()

        for i in range(steps):
            print(f"--------- epoch:{n} step:{i} ---------")
            trainer.make_experience(num_generations=grpo_config.num_generations, rank_id=rank_id)
            # TODO：swap 接口需要重新设计
            if n != 0 or i != 0:
                for param in grpo_with_grad.network.get_parameters(expand=True):
                    param._load()
                for param in grpo_with_grad.optimizer.moments1:
                    param._load()
                for param in grpo_with_grad.optimizer.moments2:
                    param._load()
                if trainer.train_pp_stage > 1:
                    for param in grpo_with_grad.accu_grads:
                        param._load()
            print("model_train and optimizer load")

            # do train
            dataset = init_grpo_dataset(trainer)
            trainer.train(grpo_with_grad, dataset)

            # after train，swap optimizer
            for param in grpo_with_grad.optimizer.moments1:
                param._offload()
            for param in grpo_with_grad.optimizer.moments2:
                param._offload()
            if trainer.train_pp_stage > 1:
                for param in grpo_with_grad.accu_grads:
                    param._offload()
            print("optimizer offload")

            # load infer model
            for param in trainer.grpo_model_infer.grpo_model.get_parameters(expand=True):
                param._load()
            for param in trainer.ref_model.get_parameters(expand=True):
                param._load()
            print("model_infer and ref_model load")

            start_time = time.time()
            reshard_param.transform()
            print(f"model_train to model_infer ckpt_transform: {format_time_delta(time.time() - start_time)}")

            if grpo_config.sync_ref_model:
                if (i+1) % grpo_config.ref_model_sync_steps == 0:
                    start_time = time.time()
                    reshard_param_policy2ref.transform()
                    print(f"model_train to ref ckpt_transform: {format_time_delta(time.time() - start_time)}")
            for param in grpo_with_grad.network.get_parameters(expand=True):
                param._offload()
            print("model_train offload")
    for param in grpo_with_grad.network.get_parameters(expand=True):
        param._load()
    trainer.save_checkpoint(rank_id=get_rank(), steps=grpo_config.epochs)
    print("save checkpoint done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="qwen make experience")
    parser.add_argument("--sft_path_infer", type=str, default=None, help="sft model path", required=True)
    parser.add_argument("--sft_path_train", type=str, default=None, help="sft model path", required=True)
    parser.add_argument("--vocab_path", required=True, help="path to vocab.json")
    parser.add_argument("--merges_file_path", required=True, help="path to merges.txt")
    parser.add_argument("--save_data_file", type=str, default=None, help="save_data_file")
    parser.add_argument("--mind_dataset_dir", type=str, default=None, help="mind_dataset_dir", required=True)
    parser.add_argument("--save_ckpt_dir", type=str, default="./", help="save_ckpt_dir")
    parser.add_argument("--use_parallel", type=str, default=True, help="use_parallel")
    parser.add_argument("--load_sft_checkpoint_infer", type=str, default=None, help="load checkpoint path")
    parser.add_argument("--load_sft_checkpoint_train", type=str, default=None, help="load checkpoint path")
    parser.add_argument("--load_ref_checkpoint", type=str, default=None, help="load checkpoint path")
    parser.add_argument("--enable_compile_cache", type=str, default=False, help="enable compile cache")
    parser.add_argument("--pre_num_generations", type=int, default=1, help="pre generate times")
    parser.add_argument("--pre_store_data", type=int, default=16, help="pre generate times")

    args = parser.parse_args()

    main(
        args.sft_path_infer,
        args.sft_path_train,
        args.use_parallel,
        args
    )
