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
"""Reference Worker."""
import os

import numpy as np
from omegaconf import DictConfig, OmegaConf

from mindformers import LlamaConfig
from mindformers import MindFormerConfig
from mindformers import logger
from mindformers.trainer.utils import load_distributed_checkpoint

import mindspore as ms
from mindspore import context, ops
from mindspore.communication import get_rank

from mindrlhf.models.grpo_models import CausalLMHybrid
from mindrlhf.utils import TimeConsumingCollector
from mindrlhf.utils.utils import load_safetensors
from mindrlhf.worker.worker import Worker
from research.deepseek3.deepseek3_config import DeepseekV3Config


class OldPolicyWorker(Worker):
    """
    This class generates responses.
    """

    SAVED_MODEL_CONFIG_YAML = "saved_old_policy_model_config.yaml"

    def __init__(self, grpo_config: DictConfig, **kwargs):
        super().__init__(config=grpo_config, worker_type=Worker.WorkerType.OLD_POLICY, **kwargs)
        logger.info("init OldPolicyWorker")
        self.grpo_config = grpo_config
        self.old_policy_config = MindFormerConfig(**OmegaConf.to_container(self.reconstructed_model_config))

        if self.model_name in ["qwen2.5", "llama"]:
            self.old_policy_model_config = LlamaConfig(**self.old_policy_config.model.model_config)
            self.old_policy_model_config.model_name = "llama"
        elif self.model_name == "deepseek":
            self.old_policy_model_config = DeepseekV3Config(**self.old_policy_config.model.model_config)
            self.old_policy_model_config.model_name = "deepseek_training"
        else:
            raise ValueError(f"model_name should in ['qwen2.5', 'llama','deepseek'], but get {self.model_name}")
        self.old_policy_model_config.checkpoint_name_or_path = None
        self.dump_mf_conf_to_yaml(self.old_policy_model_config, self.SAVED_MODEL_CONFIG_YAML)

        assert self.old_policy_model_config.parallel_config.pipeline_stage == self.pipeline_stage
        self.old_policy_pp_stage = self.pipeline_stage or 1

        assert self.old_policy_model_config.parallel_config.data_parallel == self.data_parallel
        self.old_policy_dp = self.data_parallel
        self.enable_parallel_optimizer = (
            self.grpo_config.actor_config.enable_parallel_optimizer and self.old_policy_dp > 1
        )
        context.set_auto_parallel_context(
            pipeline_stages=self.old_policy_pp_stage, enable_parallel_optimizer=self.enable_parallel_optimizer
        )
        self.old_policy_model = CausalLMHybrid(self.old_policy_model_config, grpo_config)
        self.old_policy_model.model.set_train(False)
        for name, param in self.old_policy_model.parameters_and_names():
            param.name = name
        self.on_device = True

    def get_old_policy_dp(self):
        """Get old policy model data parallel size"""
        return self.old_policy_dp

    def model(self):
        """Get old policy model"""
        return self.old_policy_model

    def compile(self):
        """compile and save strategy"""
        self.old_policy_model.model.set_train(False)
        context.set_auto_parallel_context(
            pipeline_stages=self.old_policy_pp_stage, enable_parallel_optimizer=self.enable_parallel_optimizer
        )
        old_policy_bs = self.grpo_config.rl_config.batch_size * self.old_policy_dp
        fake_data = ms.Tensor(shape=(old_policy_bs, self.grpo_config.rl_config.seq_length), dtype=ms.int32)
        actual_seq_data = ms.Tensor(shape=(old_policy_bs, self.grpo_config.rl_config.pack_num), dtype=ms.int32)
        with TimeConsumingCollector("old policy model compile"):
            stage_name = "old_policy"
            context.set_auto_parallel_context(
                strategy_ckpt_config={
                    "save_file": f"{self.save_strategy_dir}/{stage_name}_strategy/strategy_{get_rank()}.ckpt",
                    "only_trainable_params": False,
                },
                pipeline_stages=self.old_policy_pp_stage,
            )
            self.old_policy_model.compile(
                fake_data,
                None,
                None,
                None,
                False,
                False,
                fake_data,
                actual_seq_data,
                False,
                False,
                calculate_entropy=self.grpo_config.rl_config.calculate_entropy,
            )
            stage_name = "other"
            context.set_auto_parallel_context(
                strategy_ckpt_config={
                    "save_file": f"{self.save_strategy_dir}/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt",
                    "only_trainable_params": False,
                }
            )

    def offset_actual_sequence_length(self, data, offset):
        """Offset actual sequence length"""
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
        actual_sequence_length = self.offset_actual_sequence_length(
            actual_sequence_length, prompt_completion_ids_tensor.shape[1]
        )
        logger.info(
            f"precision old policy model inputs are {prompt_completion_ids_tensor}, "
            f"{samples}, {actual_sequence_length}"
        )

        results = self.old_policy_model(
            prompt_completion_ids_tensor,
            None,
            None,
            None,
            False,
            False,
            samples,
            actual_sequence_length,
            False,
            False,
            calculate_entropy=self.grpo_config.rl_config.calculate_entropy,
        )

        if self.grpo_config.rl_config.calculate_entropy:
            entropy, old_per_token_logps = results
        else:
            entropy = ms.Tensor([0.0])
            old_per_token_logps = results

        logger.info(f"old_logprobs precision is {old_per_token_logps}")
        return entropy, old_per_token_logps

    def offload(self):
        """offload old policy model"""
        if self.on_device is False:
            return
        logger.info(f"before offload old policy model {ms.hal.memory_stats()}")
        with TimeConsumingCollector("offload old policy model"):
            for param in self.old_policy_model.get_parameters(expand=True):
                # pylint: disable=W0212
                param._offload()
        logger.info(f"after offload old policy model {ms.hal.memory_stats()}")
        self.on_device = False

    def load(self):
        """load old policy model"""
        if self.on_device:
            return
        logger.info(f"before load old policy model {ms.hal.memory_stats()}")
        with TimeConsumingCollector("load old policy model"):
            for param in self.old_policy_model.get_parameters(expand=True):
                # pylint: disable=W0212
                param._load()
        logger.info(f"after load old policy model {ms.hal.memory_stats()}")
        self.on_device = True

    def load_checkpoint(self):
        """load checkpoint"""
        if not self.model_path:
            return

        if not os.path.exists(self.model_path):
            raise ValueError(f"old policy model checkpoint path: {self.model_path} not exists")

        if self.model_path and self.load_ckpt_format in ["ms_safetensors", "hf_safetensors"]:
            self.on_device = True
            strategy_path = os.path.join(self.save_strategy_dir, "merge_strategy", "old_policy_merged_strategy.ckpt")
            network = self.old_policy_model.model
            prefix = "model."
            load_safetensors(
                self.model_path, self.load_ckpt_format, network, self.old_policy_model, prefix, strategy_path
            )
            return
        load_ckpt_func = load_distributed_checkpoint if self.grpo_config.rl_config.use_parallel else ms.load_checkpoint
        logger.info(f"use_parallel is {self.grpo_config.rl_config.use_parallel} {load_ckpt_func}")
        if self.model_path:
            self.on_device = True
            param_dict = load_ckpt_func(self.model_path)
            new_param_dict = {"model." + k: v for k, v in param_dict.items()}
            logger.info(f"begin to load old policy model from: {self.model_path}")
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
        """Check params not on device"""
        if self.on_device:
            raise RuntimeError(
                "when reshard_mem_opt_level is equal to 0, old policy model must not on device before transform param"
            )

    def check_on_device(self):
        """Check params on device"""
        if not self.on_device:
            raise RuntimeError(
                "when reshard_mem_opt_level is equal to 0, old policy model must on device before transform param"
            )


class DummyOldPolicyWorker(Worker):
    # pylint: disable=W0231,W0613
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        def method(*args, **kwargs):
            pass

        return method


def get_old_policy_worker(*args, **kwargs):
    grpo_config = kwargs.get("grpo_config", None)
    if grpo_config and grpo_config.rl_config.enable_oldpolicy:
        return OldPolicyWorker(*args, **kwargs)
    logger.info(f"enable_oldpolicy is {grpo_config.rl_config.enable_oldpolicy}, OldPolicyWorker is not enabled")
    return DummyOldPolicyWorker(*args, **kwargs)


def set_enable_old_policy(grpo_config):
    """set enable_oldpolicy"""
    # calculate train batch size
    micro_batch_num = 1
    if grpo_config.actor_config.parallel_config.pipeline_stage > 1:
        micro_batch_num = grpo_config.actor_config.parallel_config.micro_batch_num
    train_bs = (
        grpo_config.rl_config.batch_size * grpo_config.actor_config.parallel_config.data_parallel * micro_batch_num
    )
    # calculate global batch size
    global_bs = (
        grpo_config.rl_config.num_rollouts
        * grpo_config.rl_config.num_generations
        * grpo_config.rl_config.chunk_size
        * grpo_config.generate_config.parallel_config.data_parallel
    )
    # set enable old policy
    if grpo_config.rl_config.calculate_entropy:
        logger.warning(
            f"enable_oldpolicy is set to True, because "
            f"calculate_entropy is "
            f"{grpo_config.rl_config.calculate_entropy}."
        )
        grpo_config.rl_config.enable_oldpolicy = True
        return

    if train_bs == global_bs and grpo_config.rl_config.num_iterations == 1:
        if grpo_config.rl_config.enable_oldpolicy:
            logger.warning(
                f"enable_oldpolicy is set to False, because train batch size({train_bs}) is equal "
                f"to global batch size({global_bs}) and "
                f"num_iterations is {grpo_config.rl_config.num_iterations}."
            )
        grpo_config.rl_config.enable_oldpolicy = False
