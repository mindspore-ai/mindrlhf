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
from mindrlhf.configs.grpo_configs import VllmMode
from mindrlhf.worker.worker import Worker, format_time_delta


def match_func(s1, s2):
    s1 = s1[s1.find('.')+1:]
    s2 = s2[s2.find('.')+1:]
    return s1 == s2


def match_func_policy2ref(s1, s2):
    s1 = s1[s1.find('.')+1:]
    s1 = s1[s1.find('.')+1:]
    return s1 == s2


def match_func_vllm(s1, s2):
    s1 = s1[s1.find('.')+1: ]
    # get rid of the first 'model'
    # eg. policy_model.model.model.layer -> policy_model.model.layer
    tmp1 = s1[:s1.find('.')]
    tmp2 = s1[s1.find('.model')+6:]
    s1 = tmp1 + tmp2
    s2 = s2[s2.find('.')+1: ]
    return s1 == s2


class TransformWorker(Worker):
    def __init__(self, grpo_config, sft_train_model, sft_infer_model, ref_model):
        super(TransformWorker, self).__init__()
        logger.info("Start prepare for parameter resharding in sft training.")
        self.sync_ref_model = grpo_config.sync_ref_model
        self.ref_model_sync_steps = grpo_config.ref_model_sync_steps
        self.save_strategy_dir = grpo_config.save_strategy_dir
        # TODO: save strategy
        ms.mint.distributed.barrier()
        src_merged_stra = os.path.join(self.save_strategy_dir, "merge_strategy/train_policy_merged_strategy.ckpt")
        dst_merged_stra = os.path.join(self.save_strategy_dir, "merge_strategy/infer_policy_merged_strategy.ckpt")
        ref_merged_stra = os.path.join(self.save_strategy_dir, "merge_strategy/infer_ref_merged_strategy.ckpt")
        if get_rank() == 0:
            ms.merge_pipeline_strategys(os.path.join(self.save_strategy_dir, "strategy_file/train_policy_strategy/"), src_merged_stra)
            ms.merge_pipeline_strategys(os.path.join(self.save_strategy_dir, "strategy_file/infer_policy_strategy/"), dst_merged_stra)
            ms.merge_pipeline_strategys(os.path.join(self.save_strategy_dir, "strategy_file/infer_ref_strategy/"), ref_merged_stra)
        else:
            print("Waiting for other workers to merge strategies.")
            time.sleep(10)
        ms.mint.distributed.barrier()
        if grpo_config.use_vllm == VllmMode.ORIGIN:
            self.reshard_param_policy2infer = TransformParametersD2D(sft_train_model, sft_infer_model,
                                                                 src_merged_stra, dst_merged_stra, match_func, offload_src=True, load_dst=True)
        else:
            self.reshard_param_policy2infer = TransformParametersD2D(sft_train_model, sft_infer_model,
                                                                    src_merged_stra, dst_merged_stra, match_func_vllm, offload_src=True, load_dst=True)
        ms.communication.comm_func.barrier()
        self.reshard_param_policy2ref = TransformParametersD2D(sft_train_model, ref_model,
                                                               src_merged_stra, ref_merged_stra, match_func=match_func_policy2ref)
        ms.communication.comm_func.barrier()

    def reshard_params(self, updata_ref=False):
        start_time = time.time()
        self.reshard_param_policy2infer.transform()
        logger.info(f"Total time for transfering policy to infer：{format_time_delta(time.time() - start_time)}")
        if updata_ref:
            start_time = time.time()
            self.reshard_param_policy2ref.transform()
            logger.info(f"Total time for transfering policy to ref{format_time_delta(time.time() - start_time)}")

