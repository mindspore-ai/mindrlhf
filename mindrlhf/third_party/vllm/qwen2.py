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

from typing import Optional

from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata

from mindspore import Tensor, runtime
from mindspore.common.api import _pynative_executor

from vllm_mindspore.model_executor.models.mf_models.qwen2 import Qwen2ForCausalLM


def sample(
        cls,
        logits: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
    """ sample """
    _pynative_executor.sync()
    runtime.synchronize()
    _pynative_executor.set_async_for_graph(True)
    next_tokens = cls.sampler(logits, sampling_metadata)
    _pynative_executor.sync()
    runtime.synchronize()
    _pynative_executor.set_async_for_graph(False)
    return next_tokens


Qwen2ForCausalLM.sample = sample
