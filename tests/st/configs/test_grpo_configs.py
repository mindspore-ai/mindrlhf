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

from mindrlhf.configs import ActorConfig, RLConfig, RefConfig, RewardConfig, GenerateConfig
from mindrlhf.configs.grpo_configs import VllmMode, ContextConfig, ParallelConfig
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="dryrun_only", essential_mark="essential")
def test_consistency_of_yaml_and_dataclass_of_parallel_config():
    """
    Feature: Uniform config.
    Description: Check the consistency of ParallelConfig structures.
    Expectation: Assert pass.
    """
    conf_from_yaml = OmegaConf.load("../../../examples/grpo/qwen_grpo_tutorial/grpo_config.yaml")
    actor_parallel_conf = OmegaConf.structured(ParallelConfig)
    ref_parallel_conf = OmegaConf.structured(ParallelConfig)
    ref_parallel_conf.data_parallel = 2
    ref_parallel_conf.pipeline_stage = 1
    ref_parallel_conf.micro_batch_num = 1
    gen_parallel_conf = OmegaConf.structured(ParallelConfig)
    gen_parallel_conf.data_parallel = 2
    gen_parallel_conf.pipeline_stage = 1
    gen_parallel_conf.micro_batch_num = 1
    assert conf_from_yaml.actor_config.parallel_config == actor_parallel_conf, (
        f"conf_from_yaml.actor_config.parallel_config={conf_from_yaml.actor_config.parallel_config}\n"
        f"actor_parallel_conf={actor_parallel_conf}"
    )
    assert conf_from_yaml.ref_config.parallel_config == ref_parallel_conf, (
        f"conf_from_yaml.ref_config.parallel_config={conf_from_yaml.ref_config.parallel_config}\n"
        f"ref_parallel_conf={ref_parallel_conf}"
    )
    assert conf_from_yaml.generate_config.parallel_config == gen_parallel_conf, (
        f"conf_from_yaml.generate_config.parallel_config={conf_from_yaml.generate_config.parallel_config}\n"
        f"gen_parallel_conf={gen_parallel_conf}"
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="dryrun_only", essential_mark="essential")
def test_consistency_of_yaml_and_dataclass_of_rl_config():
    """
    Feature: Uniform config.
    Description: Check the consistency of RLConfig structures.
    Expectation: Assert pass.
    """
    conf_from_yaml = OmegaConf.load("../../../examples/grpo/qwen_grpo_tutorial/grpo_config.yaml")
    base_conf = OmegaConf.structured(RLConfig)
    assert conf_from_yaml.rl_config == base_conf, (
        f"conf_from_yaml.rl_config={conf_from_yaml.rl_config},\n" f"base_conf.rl_config={base_conf}"
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="dryrun_only", essential_mark="essential")
def test_consistency_of_yaml_and_dataclass_of_actor_config():
    """
    Feature: Uniform config.
    Description: Check the consistency of ActorConfig structures.
    Expectation: Assert pass.
    """
    conf_from_yaml = OmegaConf.load("../../../examples/grpo/qwen_grpo_tutorial/grpo_config.yaml")
    conf_from_yaml.actor_config.reconstructed_model_config = OmegaConf.create()
    actor_config = OmegaConf.structured(ActorConfig)
    assert conf_from_yaml.actor_config.parallel_config == actor_config.parallel_config
    assert conf_from_yaml.actor_config.recompute_config == actor_config.recompute_config
    assert conf_from_yaml.actor_config.optimizer == actor_config.optimizer
    assert conf_from_yaml.actor_config.lr_schedule == actor_config.lr_schedule
    assert conf_from_yaml.actor_config == actor_config


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="dryrun_only", essential_mark="essential")
def test_consistency_of_yaml_and_dataclass_of_ref_config():
    """
    Feature: Uniform config.
    Description: Check the consistency of RefConfig structures.
    Expectation: Assert pass.
    """
    conf_from_yaml = OmegaConf.load("../../../examples/grpo/qwen_grpo_tutorial/grpo_config.yaml")
    conf_from_yaml.ref_config.reconstructed_model_config = OmegaConf.create()
    ref_conf = OmegaConf.structured(RefConfig)
    ref_conf.parallel_config.data_parallel = 2
    ref_conf.parallel_config.pipeline_stage = 1
    ref_conf.parallel_config.micro_batch_num = 1
    assert conf_from_yaml.ref_config.parallel_config == ref_conf.parallel_config
    assert conf_from_yaml.ref_config.recompute_config == ref_conf.recompute_config
    assert conf_from_yaml.ref_config == ref_conf


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="dryrun_only", essential_mark="essential")
def test_consistency_of_yaml_and_dataclass_of_reward_config():
    """
    Feature: Uniform config.
    Description: Check the consistency of RewardConfig structures.
    Expectation: Assert pass.
    """
    conf_from_yaml = OmegaConf.load("../../../examples/grpo/qwen_grpo_tutorial/grpo_config.yaml")
    reward_conf = OmegaConf.structured(RewardConfig)
    assert conf_from_yaml.reward_config == reward_conf


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="dryrun_only", essential_mark="essential")
def test_consistency_of_yaml_and_dataclass_of_gen_config():
    """
    Feature: Uniform config.
    Description: Check the consistency of GenerateConfig structures.
    Expectation: Assert pass.
    """
    conf_from_yaml = OmegaConf.load("../../../examples/grpo/qwen_grpo_tutorial/grpo_config.yaml")
    conf_from_yaml.generate_config.use_vllm = VllmMode(conf_from_yaml.generate_config.use_vllm)
    conf_from_yaml.generate_config.reconstructed_model_config = OmegaConf.create()
    gen_conf = OmegaConf.structured(GenerateConfig)
    gen_conf.parallel_config.data_parallel = 2
    gen_conf.parallel_config.pipeline_stage = 1
    gen_conf.parallel_config.micro_batch_num = 1
    assert conf_from_yaml.generate_config.sampling_config == gen_conf.sampling_config
    assert conf_from_yaml.generate_config.parallel_config == gen_conf.parallel_config
    assert conf_from_yaml.generate_config == gen_conf


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="dryrun_only", essential_mark="essential")
def test_consistency_of_yaml_and_dataclass_of_context_config():
    """
    Feature: Uniform config.
    Description: Check the consistency of ContextConfig structures.
    Expectation: Assert pass.
    """
    conf_from_yaml = OmegaConf.load("../../../examples/grpo/qwen_grpo_tutorial/grpo_config.yaml")
    context_conf = OmegaConf.structured(ContextConfig)
    assert conf_from_yaml.context == context_conf
