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
""" Transform Worker """

import os
import time

# mindspore
import mindspore as ms
from mindspore.communication import get_rank

# mindformers
from mindformers import logger

# mindrlhf
from mindrlhf.worker.worker import Worker
from mindrlhf.configs.grpo_configs import VllmMode
from mindrlhf.utils import TransformParametersD2D, TransformParametersD2DForDSv3, TimeConsumingCollector
from mindrlhf.configs.grpo_configs import GRPOConfig


def match_func(s1, s2):
    """Match func."""
    s1 = s1[s1.find(".") + 1 :]
    s2 = s2[s2.find(".") + 1 :]
    return s1 == s2


def match_func_dkv3(s1, s2):
    """match_func_dkv3."""
    s1 = s1[s1.find(".") + 1 :]
    s2 = s2[s2.find(".") + 1 :]

    def match_layout_num(s1, s2):
        split_s1 = s1.split(".")
        split_s2 = s2.split(".")
        s1_layer_num = -1
        s2_layer_num = -1
        for i, value in enumerate(split_s1):
            if value == "layers":
                s1_layer_num = split_s1[i + 1]
        for i, value in enumerate(split_s2):
            if value == "layers":
                s2_layer_num = split_s2[i + 1]
        return int(s1_layer_num) == int(s2_layer_num)

    if "attention.l2q_nope_proj.weight" in s1 and "attention.l2q_proj.weight" in s2:
        return match_layout_num(s1, s2)
    if "attention.l2q_pe_proj.weight" in s1 and "attention.l2q_proj.weight" in s2:
        return match_layout_num(s1, s2)
    if "attention.kv2l_k_pe.weight" in s1 and "attention.kv2l.weight" in s2:
        return match_layout_num(s1, s2)
    if "attention.kv2l_latent_kv.weight" in s1 and "attention.kv2l.weight" in s2:
        return match_layout_num(s1, s2)
    if "routed_experts.topk_bias" in s1 and "router.e_score_correction_bias" in s2:
        return match_layout_num(s1, s2)
    if "ffn.w1" in s1 and "ffn.w1.weight" in s2:
        return match_layout_num(s1, s2)
    if "ffn.w2" in s1 and "ffn.w2.weight" in s2:
        return match_layout_num(s1, s2)
    if "ffn.w3" in s1 and "ffn.w3.weight" in s2:
        return match_layout_num(s1, s2)
    if "router_dense.weight" in s1 and "router.dense.weight" in s2:
        return match_layout_num(s1, s2)
    return s1 == s2


def match_func_policy2ref(s1, s2):
    s1 = s1[s1.find(".") + 1:]
    s1 = s1[s1.find(".") + 1:]
    return s1 == s2


def match_func_vllm(s1, s2):
    """Weight match_func_vllm for qwen 2.5."""
    s1 = s1[s1.find(".") + 1:]
    # get rid of the first 'model'
    # eg. policy_model.model.model.layer -> policy_model.model.layer
    tmp1 = s1[: s1.find(".")]
    tmp2 = s1[s1.find(".model") + 6:]
    s1 = tmp1 + tmp2
    s2 = s2[s2.find(".") + 1:]
    return s1 == s2


def match_func_dkv3_vllm(s1, s2):
    """
    match_func_dkv3_vllm
    """
    s1 = s1[s1.find(".") + 1:]
    # get rid of the first 'model'
    # eg. policy_model.model.model.layer -> policy_model.model.layer
    tmp1 = s1[: s1.find(".")]
    tmp2 = s1[s1.find(".model") + 6:]
    s1 = tmp1 + tmp2
    s2 = s2[s2.find(".") + 1:]

    def match_layout_num(s1, s2):
        split_s1 = s1.split(".")
        split_s2 = s2.split(".")
        s1_layer_num = -1
        s2_layer_num = -1
        for i, value in enumerate(split_s1):
            if value == "layers":
                s1_layer_num = split_s1[i + 1]
        for i, value in enumerate(split_s2):
            if value == "layers":
                s2_layer_num = split_s2[i + 1]
        return int(s1_layer_num) == int(s2_layer_num)

    if "attention.l2q_nope_proj.weight" in s1 and "attention.l2q_proj.weight" in s2:
        return match_layout_num(s1, s2)
    if "attention.l2q_pe_proj.weight" in s1 and "attention.l2q_proj.weight" in s2:
        return match_layout_num(s1, s2)
    if "attention.kv2l_k_pe.weight" in s1 and "attention.kv2l.weight" in s2:
        return match_layout_num(s1, s2)
    if "attention.kv2l_latent_kv.weight" in s1 and "attention.kv2l.weight" in s2:
        return match_layout_num(s1, s2)
    if "routed_experts.topk_bias" in s1 and "router.e_score_correction_bias" in s2:
        return match_layout_num(s1, s2)
    if "ffn.w1" in s1 and "ffn.w1.weight" in s2:
        return match_layout_num(s1, s2)
    if "ffn.w2" in s1 and "ffn.w2.weight" in s2:
        return match_layout_num(s1, s2)
    if "ffn.w3" in s1 and "ffn.w3.weight" in s2:
        return match_layout_num(s1, s2)
    if "router_dense.weight" in s1 and "router.dense.weight" in s2:
        return match_layout_num(s1, s2)
    return s1 == s2


class TransformWorker(Worker):
    """TransformWorker"""

    def __init__(
        self,
        grpo_config: GRPOConfig,
        sft_model_config_train,
        sft_train_model,
        sft_infer_model,
        ref_model,
        old_policy_model,
    ):
        super(TransformWorker, self).__init__()
        logger.info("Start prepare for parameter resharding in sft training.")
        self.grpo_config = grpo_config
        self.sync_ref_model = grpo_config.ref_config.sync_ref_model
        self.ref_model_sync_steps = grpo_config.ref_config.ref_model_sync_steps
        self.save_strategy_dir = grpo_config.rl_config.save_strategy_dir
        # TODO: save strategy
        ms.mint.distributed.barrier()
        src_merged_stra = os.path.join(self.save_strategy_dir, "merge_strategy", "train_policy_merged_strategy.ckpt")
        dst_merged_stra = os.path.join(self.save_strategy_dir, "merge_strategy", "infer_policy_merged_strategy.ckpt")
        ref_merged_stra = os.path.join(self.save_strategy_dir, "merge_strategy", "infer_ref_merged_strategy.ckpt")
        old_merged_stra = os.path.join(self.save_strategy_dir, "merge_strategy", "old_policy_merged_strategy.ckpt")

        with TimeConsumingCollector("TransformWorker init"):
            if get_rank() == 0:
                ms.merge_pipeline_strategys(
                    os.path.join(self.save_strategy_dir, "train_policy_strategy"), src_merged_stra
                )
                ms.merge_pipeline_strategys(
                    os.path.join(self.save_strategy_dir, "infer_policy_strategy"), dst_merged_stra
                )
                ms.merge_pipeline_strategys(os.path.join(self.save_strategy_dir, "infer_ref_strategy"), ref_merged_stra)
                if grpo_config.rl_config.enable_oldpolicy:
                    ms.merge_pipeline_strategys(
                        os.path.join(self.save_strategy_dir, "old_policy_strategy"), old_merged_stra
                    )
            else:
                logger.info("Waiting for main worker to merge strategies.")
                time.sleep(10)
            ms.mint.distributed.barrier()
            model_type = "deepseekv3" if "deepseek" in grpo_config.rl_config.model_name else ""
            transform_args = {}
            reshard_mode = grpo_config.rl_config.reshard_mode
            if model_type == "deepseekv3":
                transform_args = {
                    "n_head": sft_model_config_train.num_heads,
                    "qk_nope_head_dim": sft_model_config_train.qk_nope_head_dim,
                    "qk_rope_head_dim": sft_model_config_train.qk_rope_head_dim,
                    "tok_embedding_shape": (129280, 7168),
                }
            if grpo_config.generate_config.use_vllm == VllmMode.ORIGIN:
                if model_type == "deepseekv3":
                    self.reshard_param_policy2infer = TransformParametersD2DForDSv3(
                        sft_train_model,
                        sft_infer_model,
                        transform_args,
                        src_merged_stra,
                        dst_merged_stra,
                        match_func_dkv3,
                        reshard_mode,
                    )
                else:
                    self.reshard_param_policy2infer = TransformParametersD2D(
                        sft_train_model, sft_infer_model, src_merged_stra, dst_merged_stra, match_func, reshard_mode
                    )
            else:
                if model_type == "deepseekv3":
                    self.reshard_param_policy2infer = TransformParametersD2DForDSv3(
                        sft_train_model,
                        sft_infer_model,
                        transform_args,
                        src_merged_stra,
                        dst_merged_stra,
                        match_func_dkv3_vllm,
                        reshard_mode,
                    )
                else:
                    self.reshard_param_policy2infer = TransformParametersD2D(
                        sft_train_model,
                        sft_infer_model,
                        src_merged_stra,
                        dst_merged_stra,
                        match_func_vllm,
                        reshard_mode,
                    )
            ms.communication.comm_func.barrier()
            self.reshard_param_policy2ref = TransformParametersD2D(
                sft_train_model,
                ref_model,
                src_merged_stra,
                ref_merged_stra,
                match_func=match_func_policy2ref,
                reshard_mode=reshard_mode,
            )
            if grpo_config.rl_config.enable_oldpolicy:
                self.old_policy_param_policy2old = TransformParametersD2D(
                    sft_train_model,
                    old_policy_model,
                    src_merged_stra,
                    old_merged_stra,
                    match_func=match_func_policy2ref,
                    reshard_mode=reshard_mode,
                )
            ms.communication.comm_func.barrier()

    def reshard_params(self, step_num, input_on_device_flag_dict=None):
        """
        reshard parameter from src to dst
        """
        if input_on_device_flag_dict is None:
            input_on_device_flag_dict = {
                "policy2infer": (True, True),
                "policy2ref": (True, True),
                "policy2old": (True, True),
            }
        policy2infer_flag = input_on_device_flag_dict.get("policy2infer")
        policy2ref_flag = input_on_device_flag_dict.get("policy2ref")
        policy2old_flag = input_on_device_flag_dict.get("policy2old")
        if policy2infer_flag is None or policy2ref_flag is None or policy2old_flag is None:
            raise ValueError("Key in input_on_device_flag_dict must be policy2infer, policy2ref or policy2old")
        with TimeConsumingCollector("reshard params"):
            self.reshard_param_policy2infer.transform(policy2infer_flag)
            if self.grpo_config.rl_config.enable_oldpolicy:
                self.old_policy_param_policy2old.transform(policy2old_flag)
            if self.sync_ref_model and ((step_num + 1) % self.ref_model_sync_steps == 0):
                self.reshard_param_policy2ref.transform(policy2ref_flag)
