# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore import context, ops
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
    init_grpo_dataset,
    init_grpo_network_and_optimizer,
)
from mindrlhf.utils import transfer_from_str_to_bool
from mindrlhf.models.qwen2.qwen2_tokenizer import Qwen2Tokenizer
from mindrlhf.reward.reward_fn import reward_func_from_jiaoda, accuracy_reward, format_reward


def format_time_delta(seconds):
    "计算时间差"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{seconds:.4f}"

def main(sft_path_infer, sft_path_train, use_parallel, args):
    """
    grpo one stage
    """
    use_parallel = transfer_from_str_to_bool(use_parallel)
    enable_compile_cache = transfer_from_str_to_bool(args.enable_compile_cache)

    # init config with yaml
    sft_config_infer = MindFormerConfig(sft_path_infer)
    sft_config_infer.use_parallel = use_parallel
    os.environ["RUN_MODE"] = sft_config_infer.run_mode

    # init sft infer model
    sft_config_infer.model.model_config.parallel_config = (
        sft_config_infer.parallel_config
    )

    sft_model_config_infer = LlamaConfig(**sft_config_infer.model.model_config)
    sft_model_config_infer.checkpoint_name_or_path = args.load_sft_checkpoint_infer
    sft_model_config_infer.model_name = "llama"

    # init sft train config
    sft_config_train = MindFormerConfig(sft_path_train)
    sft_config_train.use_parallel = use_parallel
    sft_config_train.model.model_config.parallel_config = (
        sft_config_train.parallel_config
    )
    sft_config_train.model.model_config.parallel_config.recompute = sft_config_train.recompute_config
    sft_model_config_train = LlamaConfig(**sft_config_train.model.model_config)
    sft_model_config_train.checkpoint_name_or_path = args.load_sft_checkpoint_train
    sft_model_config_train.model_name = "llama"

    # init grpo config
    grpo_config = GRPOConfig()
    grpo_config.mind_dataset_dir = args.mind_dataset_dir
    grpo_config.save_data_file = args.save_data_file
    grpo_config.save_ckpt_dir = args.save_ckpt_dir
    grpo_config.align_type = "rlhf_stages"
    grpo_config.use_parallel = use_parallel
    grpo_config = combine_grpo_config(grpo_config, sft_model_config_infer)  # grpo_config infer 和 train 共用

    # init ref model
    ref_config = MindFormerConfig(sft_path_infer)
    ref_config.use_parallel = use_parallel
    ref_config.model.model_config.parallel_config = ref_config.parallel_config
    ref_config.model.model_config.use_past = False
    ref_model_config = LlamaConfig(**ref_config.model.model_config)
    ref_model_config.checkpoint_name_or_path = args.load_ref_checkpoint
    ref_model_config.model_name = "llama"

    # init context
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
    print("trainer.sft_model_config_infer:", trainer.sft_model_config_infer)

    trainer.grpo_model_infer.grpo_model.policy_model.model.add_flags_recursive(is_first_iteration=True)
    trainer.make_experience(num_generations=args.pre_num_generations, rank_id=rank_id, pre_run_flag=True)
    sample = trainer.store[0]
    trainer.store = [sample for _ in range(args.pre_store_data)]

    dataset = init_grpo_dataset(trainer)
    data = next(dataset.create_dict_iterator())
    print("data:\n", data)

    # ================= pre compile model for loading dist ckpt ========================
    grpo_with_grad = init_grpo_network_and_optimizer(trainer)
    grpo_with_grad.set_train(True)
    trainer.grpo_model_train.grpo_model_train.policy_model.model.set_train(True)
    start_time = time.time()
    stage_name = 'train'
    context.set_auto_parallel_context(
        strategy_ckpt_config={
            "save_file":
                f"../strategy/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt"},
        pipeline_stages=trainer.train_pp_stage         
    )
    grpo_with_grad.compile(**data)

    stage_name = 'other'
    context.set_auto_parallel_context(
        strategy_ckpt_config={
            "save_file":
                f"../strategy/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt"})
    print(f"grpo_with_grad time: {format_time_delta(time.time() - start_time)}")

    trainer.grpo_model_infer.grpo_model.policy_model.model.set_train(False)
    trainer.ref_model.model.set_train(False)
    print(f"ref_model.class.name: {trainer.ref_model.model.__class__.__name__}")
    # ====================================================================================
    trainer.load_checkpoint()

    # 权重倒换
    start_time = time.time()
    def match_func(s1, s2):
        s1 = s1[s1.find('.')+1: ]
        s2 = s2[s2.find('.')+1: ]
        return s1 == s2

    def match_func_policy2ref(s1, s2):
        s1 = s1[s1.find('.')+1: ]
        s1 = s1[s1.find('.')+1: ]
        return s1 == s2

    src_merged_stra = "../merge_strategy/train_policy_merged_strategy.ckpt"
    dst_merged_stra = "../merge_strategy/infer_policy_merged_strategy.ckpt"
    ref_merged_stra = "../merge_strategy/infer_ref_merged_strategy.ckpt"

    if get_rank() in list(range(0, get_group_size(), get_group_size() // context.get_auto_parallel_context("pipeline_stages"))):
        ms.merge_pipeline_strategys("../strategy/train_policy_strategy/", src_merged_stra)
        ms.merge_pipeline_strategys("../strategy/infer_policy_strategy/", dst_merged_stra)
        ms.merge_pipeline_strategys("../strategy/infer_ref_strategy/", ref_merged_stra)
    ms.mint.distributed.barrier()
    reshard_param = TransformParametersD2D(trainer.grpo_model_train, trainer.grpo_model_infer,
                                          src_merged_stra, dst_merged_stra, match_func)
    ms.communication.comm_func.barrier()
    reshard_param_policy2ref = TransformParametersD2D(trainer.grpo_model_train, trainer.ref_model,
                                           src_merged_stra, ref_merged_stra, match_func=match_func_policy2ref)
    ms.communication.comm_func.barrier()

    for n in range(grpo_config.epochs):
        # do generate
        steps = trainer.prompt_dataset.get_dataset_size() // trainer.prompt_dataset.get_batch_size()

        for i in range(steps):
            print(f"--------- epoch:{n} step:{i} ---------")
            trainer.make_experience(num_generations=grpo_config.num_generations, rank_id=rank_id)

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

            for param in grpo_with_grad.optimizer.moments1:
                param._offload()
            for param in grpo_with_grad.optimizer.moments2:
                param._offload()
            if trainer.train_pp_stage > 1:
                for param in grpo_with_grad.accu_grads:
                    param._offload()
            print("optimizer offload")

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
                print("grpo with grad offload debug: ", param.name, flush=True)
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
    parser.add_argument("--use_parallel", type=str, default=False, help="use_parallel")
    parser.add_argument("--load_sft_checkpoint_infer", type=str, default=None, help="load checkpoint path")
    parser.add_argument("--load_sft_checkpoint_train", type=str, default=None, help="load checkpoint path")
    parser.add_argument("--load_ref_checkpoint", type=str, default=None, help="load checkpoint path")
    parser.add_argument("--enable_compile_cache", type=str, default=False, help="enable compile cache")
    parser.add_argument("--pre_num_generations", type=int, default=1, help="pre generate times")
    parser.add_argument("--pre_store_data", type=int, default=16, help="pre generate times")

    my_args = parser.parse_args()

    main(
        my_args.sft_path_infer,
        my_args.sft_path_train,
        my_args.use_parallel,
        my_args
    )
