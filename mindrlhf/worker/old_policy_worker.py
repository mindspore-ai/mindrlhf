# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Reference Worker """

# python
import time
import os
import numpy as np

import mindspore as ms
from mindspore.communication import get_rank
from mindspore import context, ops

from mindformers import MindFormerConfig
from mindformers.trainer.utils import load_distributed_checkpoint
from mindformers import LlamaConfig
from mindformers import logger
from research.deepseek3.deepseek3_config import DeepseekV3Config

from mindrlhf.models.grpo_models import CausalLMHybrid
from mindrlhf.worker.worker import Worker
from mindrlhf.utils import print_perf_stat
from mindrlhf.configs.grpo_configs import GRPOConfig
from mindrlhf.utils.utils import load_safetensors

class OldPolicyWorker(Worker):
    """
    This class generates responses.
    """

    def __init__(self, grpo_config: GRPOConfig, sft_path_train, args):
        super().__init__()
        self.grpo_config = grpo_config
        self.load_ckpt_format = self.grpo_config.rl_config.load_ckpt_format
        if grpo_config.rl_config.num_iterations <= 1:
            self.on_device = False
            logger.info(f"num_iterations {grpo_config.rl_config.num_iterations} <= 1, OldPolicyWorker is not enalbled")
        else:
            logger.info("init OldPolicyWorker")
            self.args = args
            old_policy_config = MindFormerConfig(sft_path_train)
            old_policy_config.use_parallel = grpo_config.rl_config.use_parallel
            old_policy_config.parallel_config = MindFormerConfig(**grpo_config.actor_config.parallel_config.param_dict)
            logger.info(f"old_policy parallel_config:{old_policy_config.parallel_config}")
            old_policy_config.recompute_config = grpo_config.actor_config.recompute_config.param_dict
            logger.info(f"old_policy_config recompute_config:{old_policy_config.recompute_config}")
            old_policy_config.model.model_config.parallel_config = old_policy_config.parallel_config
            old_policy_config.model.model_config.parallel_config.recompute = old_policy_config.recompute_config

            if args.custom_model_name in ["qwen", "llama"]:
                old_policy_config.model.model_config.use_eod_attn_mask_compression = (
                    grpo_config.actor_config.use_eod_attn_mask_compression
                )
                old_policy_model_config = LlamaConfig(**old_policy_config.model.model_config)
                old_policy_model_config.model_name = "llama"
            elif args.custom_model_name == "deepseek":
                old_policy_config.model.model_config.moe_config = old_policy_config.moe_config
                old_policy_model_config = DeepseekV3Config(**old_policy_config.model.model_config)
                old_policy_model_config.model_name = "deepseek_training"
            else:
                raise ValueError(f"model_name should in ['qwen', 'llama','deepseek'], but get {args.custom_model_name}")

            old_policy_model_config.checkpoint_name_or_path = grpo_config.actor_config.load
            self.old_policy_ckpt_path = old_policy_model_config.checkpoint_name_or_path
            old_policy_model_config.checkpoint_name_or_path = None

            self.old_policy_pp_stage = old_policy_model_config.parallel_config.pipeline_stage or 1
            self.old_policy_dp = old_policy_model_config.parallel_config.data_parallel
            self.enable_parallel_optimizer = (
                grpo_config.actor_config.enable_parallel_optimizer
                and self.old_policy_dp > 1
            )
            context.set_auto_parallel_context(
                pipeline_stages=self.old_policy_pp_stage, enable_parallel_optimizer=self.enable_parallel_optimizer
            )
            self.old_policy_model_config = old_policy_model_config
            self.old_policy_model = CausalLMHybrid(old_policy_model_config, grpo_config)
            self.old_policy_model.model.set_train(False)
            for name, param in self.old_policy_model.parameters_and_names():
                param.name = name
            self.on_device = True
            self.save_strategy_dir = grpo_config.rl_config.save_strategy_dir

    def get_old_policy_dp(self):
        return self.old_policy_dp

    def model(self):
        if self.grpo_config.rl_config.num_iterations <= 1:
            return None
        return self.old_policy_model

    def compile(self):
        """compile and save strategy"""
        if self.grpo_config.rl_config.num_iterations <= 1:
            return
        self.old_policy_model.model.set_train(False)
        context.set_auto_parallel_context(
            pipeline_stages=self.old_policy_pp_stage, enable_parallel_optimizer=self.enable_parallel_optimizer
        )
        old_policy_bs = self.grpo_config.rl_config.batch_size * self.old_policy_dp
        fake_data = ms.Tensor(shape=(old_policy_bs, self.grpo_config.rl_config.seq_length), dtype=ms.int32)
        actual_seq_data = ms.Tensor(shape=(old_policy_bs, self.grpo_config.rl_config.pack_num), dtype=ms.int32)
        start_time = time.time()
        stage_name = "old_policy"
        context.set_auto_parallel_context(
            strategy_ckpt_config={
                "save_file": f"{self.save_strategy_dir}/{stage_name}_strategy/strategy_{get_rank()}.ckpt"
            },
            pipeline_stages=self.old_policy_pp_stage,
        )
        self.old_policy_model.compile(
            fake_data, None, None, None, False, False, fake_data, actual_seq_data, False, False
        )
        stage_name = "other"
        context.set_auto_parallel_context(
            strategy_ckpt_config={
                "save_file": f"{self.save_strategy_dir}/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt"
            }
        )
        end_time = time.time()
        print_perf_stat(start_time, end_time, "old policy model compile")

    # pylint: disable=W0613
    def offset_actual_seq_length(self, data, offset):
        bs = data.shape[0] // self.old_policy_model.dp
        n = data.shape[1]
        data_type = data.dtype
        data = data.reshape((self.old_policy_model.dp, bs, n))
        offsets = ops.cast(ops.range(0, bs * offset, offset).reshape((1, bs, 1)), data_type)
        data = data + offsets
        actual_seq_lenth = ops.cast(ops.reshape(data, (-1,)), data_type)
        return actual_seq_lenth

    def compute_old_log_prob(self, prompt_completion_ids_tensor, samples, actual_sequence_length):
        """compute old log probs"""
        np.set_printoptions(threshold=1024)
        context.set_auto_parallel_context(
            pipeline_stages=self.old_policy_pp_stage, enable_parallel_optimizer=self.enable_parallel_optimizer
        )
        actual_seq_length = self.offset_actual_seq_length(actual_sequence_length, prompt_completion_ids_tensor.shape[1])
        logger.info(
            f"precision old policy model inputs are {prompt_completion_ids_tensor}, "
            f"{samples}, {actual_sequence_length}"
        )

        old_per_token_logps = self.old_policy_model(
            prompt_completion_ids_tensor, None, None, None, False, False, samples, actual_seq_length, False, False
        )

        logger.info(f"old_logprobs precision is {old_per_token_logps}")
        return old_per_token_logps

    def offload(self):
        """offload old policy model"""
        if self.grpo_config.rl_config.num_iterations <= 1:
            return
        if self.on_device is False:
            return
        logger.info(f"before offload old policy model {ms.hal.memory_stats()}")
        start_time = time.time()
        for param in self.old_policy_model.get_parameters(expand=True):
            # pylint: disable=W0212
            param._offload()
        end_time = time.time()
        print_perf_stat(start_time, end_time, "offload old policy model")
        logger.info(f"after offload old policy model {ms.hal.memory_stats()}")
        self.on_device = False

    def load(self):
        """load old policy model"""
        if self.grpo_config.rl_config.num_iterations <= 1:
            return
        if self.on_device:
            return
        logger.info(f"before load old policy model {ms.hal.memory_stats()}")
        start_time = time.time()
        for param in self.old_policy_model.get_parameters(expand=True):
            # pylint: disable=W0212
            param._load()
        end_time = time.time()
        print_perf_stat(start_time, end_time, "load old policy model")
        logger.info(f"after load old policy model {ms.hal.memory_stats()}")
        self.on_device = True

    def load_checkpoint(self):
        """load checkpoint"""
        if self.grpo_config.rl_config.num_iterations <= 1:
            return
        if not self.old_policy_ckpt_path:
            return

        if not os.path.exists(self.old_policy_ckpt_path):
            raise ValueError(f"old policy model checkpoint path: {self.old_policy_ckpt_path} not exists")

        if self.old_policy_ckpt_path and self.load_ckpt_format in  ["ms_safetensors", "hf_safetensors"]:
            self.on_device = True
            strategy_path = os.path.join(self.save_strategy_dir, "merge_strategy", "old_policy_merged_strategy.ckpt")
            network = self.old_policy_model.model
            prefix = "model."
            load_safetensors(self.old_policy_ckpt_path, self.load_ckpt_format, network,
                             self.old_policy_model, prefix, strategy_path)
            return
        load_ckpt_func = load_distributed_checkpoint if self.grpo_config.rl_config.use_parallel else ms.load_checkpoint
        logger.info(f"use_parallel is {self.grpo_config.rl_config.use_parallel} {load_ckpt_func}")
        if self.old_policy_ckpt_path:
            self.on_device = True
            param_dict = load_ckpt_func(self.old_policy_ckpt_path)
            new_param_dict = {"model." + k: v for k, v in param_dict.items()}
            logger.info(f"begin to load old policy model from: {self.old_policy_ckpt_path}")
            for _, param in self.old_policy_model.parameters_and_names():
                logger.info(f"old policy model para names:   {param.name}")
            param_not_load, ckpt_not_load = ms.load_param_into_net(self.old_policy_model, new_param_dict)
            logger.info(f"param not load: {param_not_load}")
            logger.info(f"ckpt not load: {ckpt_not_load}")

    def convert_map_dict(self, source_dict, **kwargs):
        """convert_map_dict"""
        network = self.old_policy_model.model
        prefix = "model."
        weight_dict = network.convert_map_dict(source_dict, **kwargs)
        new_weight_dict = {f"{prefix}{key}": value for key, value in weight_dict.items()}
        return new_weight_dict

    def check_not_on_device(self):
        if self.grpo_config.rl_config.num_iterations <= 1:
            return
        assert not self.on_device, (
            "when reshard_mem_opt_level is equal to 0, " "old policy model must not on device before transform param"
        )

    def check_on_device(self):
        if self.grpo_config.rl_config.num_iterations <= 1:
            return
        assert self.on_device, (
            "when reshard_mem_opt_level is equal to 0, " "old policy model must on device before transform param"
        )
