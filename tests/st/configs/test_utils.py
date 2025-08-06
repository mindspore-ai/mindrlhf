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
"""Test case of GRPO config."""
import os
import sys

WORKDIR = os.path.dirname(os.path.abspath(__file__))
mindrlhf_path = os.path.join(WORKDIR, "../../../")
mindformers_path = os.path.join(WORKDIR, "../mindformers")
sys.path = [mindrlhf_path, mindformers_path] + sys.path

from omegaconf import OmegaConf

from mindrlhf.configs import GRPOConfigGenerator


class TestGRPOUtils:
    """Test GRPO utils."""

    def test_grpo_config_generator(self):
        """
        Feature: Uniform config.
        Description: Test GRPOConfigGenerator.
        Expectation: Assert pass.
        """
        conf_from_yaml = OmegaConf.load("../../../examples/grpo/qwen_grpo_tutorial/grpo_config.yaml")
        conf_from_yaml.actor_config.model_config = (
            "../../../model_configs/qwen_grpo/qwen2_5_7b/finetune_qwen2_5_7b.yaml"
        )
        conf_from_yaml.actor_config.parallel_config.micro_batch_num = 2
        conf_from_yaml.actor_config.parallel_config.vocab_emb_dp = False
        conf_from_yaml.actor_config.recompute_config.recompute = True
        conf_from_yaml.actor_config.enable_parallel_optimizer = True
        conf_from_yaml.actor_config.use_eod_attn_mask_compression = True

        conf_from_yaml.ref_config.model_config = "../../../model_configs/qwen_grpo/qwen2_5_7b/finetune_qwen2_5_7b.yaml"
        conf_from_yaml.ref_config.ref_model_batch_size = 2
        conf_from_yaml.ref_config.use_eod_attn_mask_compression = True

        conf_from_yaml.generate_config.model_config = (
            "../../../model_configs/qwen_grpo/qwen2_5_7b/predict_qwen2_5_7b_instruct.yaml"
        )
        conf_from_yaml.generate_config.load = "Qwen2.5-7B"
        conf_from_yaml.generate_config.infer_model_batch_size = 2
        conf_from_yaml.generate_config.max_model_len = 32768
        conf_from_yaml.generate_config.max_num_batched_tokens = 32768
        conf_from_yaml.generate_config.gpu_memory_utilization = 0.5
        conf_from_yaml.generate_config.sampling_config.temperature = 0.8
        conf_from_yaml.generate_config.sampling_config.repetition_penalty = 1.05
        conf_from_yaml.generate_config.sampling_config.top_p = 0.8
        conf_from_yaml.generate_config.sampling_config.top_k = 20

        conf = GRPOConfigGenerator.create_config(conf_from_yaml)
        whole_config = OmegaConf.load("../../data/qwen_grpo_base_config/complete_qwen_grpo_config.yaml")
        assert conf.actor_config.reconstructed_model_config == whole_config.actor_config.reconstructed_model_config
        assert conf.ref_config.reconstructed_model_config == whole_config.ref_config.reconstructed_model_config
        assert (
            conf.generate_config.reconstructed_model_config == whole_config.generate_config.reconstructed_model_config
        )
