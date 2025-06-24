#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
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
"""Qwen2 Sample Patch."""

import os
from typing import Optional
import numpy as np

from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size

import mindspore as ms
from mindspore import Tensor, runtime, JitConfig
from mindspore.common.api import _pynative_executor

from mindformers.tools.register.config import MindFormerConfig
from mindformers.core.context import build_mf_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.models.llama import LlamaConfig as LlamaConfig_MF
from mindformers import logger
from research.qwen2_5.infer.qwen2_5 import (
    ParallelQwenForCausalLM as ParallelQwenForCausalLM_MF,
)

# from vllm_mindspore.model_executor.layers.sampler import get_sampler
# from vllm_mindspore.model_executor.models.mf_models.qwen2 import Qwen2ForCausalLM
# from vllm_mindspore.utils import calc_block_num


def sample(
    cls,
    logits: Tensor,
    sampling_metadata: SamplingMetadata,
) -> Optional[SamplerOutput]:
    _pynative_executor.sync()
    runtime.synchronize()
    next_tokens = cls.sampler(logits, sampling_metadata)
    _pynative_executor.sync()
    runtime.synchronize()
    return next_tokens

# FIXME
# def qwen2_init(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
#     """Qwen2 init"""
#     logger.info("### Apply monkey patch for Qwen2ForCausalLM init")
#     super(Qwen2ForCausalLM, self).__init__(vllm_config=vllm_config, prefix=prefix)

#     self.mf_config = MindFormerConfig(os.getenv("MINDFORMERS_MODEL_CONFIG"))
#     build_mf_context(self.mf_config)
#     build_parallel_config(self.mf_config)
#     self.mf_config.model.model_config.parallel_config = self.mf_config.parallel_config
#     self.mf_config.model.model_config.parallel_config.model_parallel = get_tensor_model_parallel_world_size()
#     self.mf_config.model.model_config.parallel_config.pipeline_stage = 1

#     self.mf_model_config = LlamaConfig_MF(**self.mf_config.model.model_config)
#     # Cannot get num_gpu_blocks from cache config now, calculate one first.
#     self.mf_model_config.num_blocks = calc_block_num(self.cache_config, self.model_config, self.parallel_config)
#     self.mf_model_config.block_size = self.cache_config.block_size
#     if self.mf_config.moe_config:
#         self.mf_model_config.moe_config = self.mf_config.moe_config
#     self.mf_model_config.return_hidden_states = True
#     self.mf_model_config.npu_mem_size = 0

#     # qwen qkv concat will support in next version
#     self.mf_model_config.qkv_concat = False
#     self.mf_config.model.model_config.qkv_concat = False

#     # Initial network
#     from mindrlhf.utils.reshard_optimizer import apply_opt_communication_groups
#     apply_opt_communication_groups()
#     self.network = ParallelQwenForCausalLM_MF(self.mf_model_config)

#     # pylint: disable=W0212
#     self.network._jit_config_dict = JitConfig(jit_level="O0", infer_boost="on").jit_config_dict

#     self.mf_config.load_checkpoint = self.get_model_path()

#     self.mf_kvcaches_init = False

#     self.sampler = get_sampler()
#     self.set_modules({"model": self.network})

#     self.prefill_mask = Tensor(np.triu(np.ones(shape=(128, 128), dtype=np.float16), k=1), dtype=ms.bfloat16)

#     self.decode_mask = Tensor(
#         np.triu(np.ones(shape=(self.mf_model_config.seq_length, self.mf_model_config.seq_length), dtype=np.int8), k=1),
#         dtype=ms.bfloat16,
#     )

#     self.hard_mask = Tensor([0], dtype=ms.bfloat16).reshape(1, 1)

#     self.gather = ms.ops.Gather()
#     self.set_flags = False