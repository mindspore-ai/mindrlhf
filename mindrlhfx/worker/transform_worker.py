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

# python
import os
import time

# mindspore
import mindspore
import mindspore as ms
from mindspore.communication import get_rank
from mindspore.communication.management import get_group_size
from mindspore import context

# mindformers
from mindformers import logger

# mindrlhf
from mindrlhf.utils import TransformParametersD2D
from mindrlhfx.worker.worker import Worker, format_time_delta


def match_func(s1, s2):
    s1 = s1[s1.find('.')+1:]
    s2 = s2[s2.find('.')+1:]
    return s1 == s2


def match_func_policy2ref(s1, s2):
    s1 = s1[s1.find('.')+1:]
    s1 = s1[s1.find('.')+1:]
    return s1 == s2


class TransformWorker(Worker):
    def __init__(self, grpo_config, sft_train_model, sft_infer_model, ref_model):
        logger.info("Start prepare for parameter resharding in sft training.")
        self.sync_ref_model = grpo_config.sync_ref_model
        self.ref_model_sync_steps = grpo_config.ref_model_sync_steps
        # TODO: save strategy
        src_merged_stra = "../../merge_strategy/train_policy_merged_strategy.ckpt"
        dst_merged_stra = "../../merge_strategy/infer_policy_merged_strategy.ckpt"
        ref_merged_stra = "../../merge_strategy/infer_ref_merged_strategy.ckpt"
        if get_rank() in list(range(0, get_group_size(), get_group_size() // context.get_auto_parallel_context("pipeline_stages"))):
            ms.merge_pipeline_strategys("../../strategy/train_policy_strategy/", src_merged_stra)
            ms.merge_pipeline_strategys("../../strategy/infer_policy_strategy/", dst_merged_stra)
            ms.merge_pipeline_strategys("../../strategy/infer_ref_strategy/", ref_merged_stra)
        ms.mint.distributed.barrier()
        self.reshard_param_policy2infer = TransformParametersD2D(sft_train_model, sft_infer_model,
                                                                 src_merged_stra, dst_merged_stra, match_func)
        ms.communication.comm_func.barrier()
        self.reshard_param_policy2ref = TransformParametersD2D(sft_train_model, ref_model,
                                                               src_merged_stra, ref_merged_stra, match_func=match_func_policy2ref)
        ms.communication.comm_func.barrier()

    def reshard_params(self, step_num):
        start_time = time.time()
        self.reshard_param_policy2infer.transform()
        if self.sync_ref_model and ((step_num + 1) % self.ref_model_sync_steps == 0):
            self.reshard_param_policy2ref.transform()
        logger.info(f"权重倒换执行：{format_time_delta(time.time() - start_time)}")
