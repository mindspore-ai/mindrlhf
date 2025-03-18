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
import numpy as np
import time

# mindspore
import mindspore
import mindspore as ms
from mindspore import Tensor
from mindspore.communication import GlobalComm, get_rank
from mindspore import context
from mindspore import communication as D

# mindformers
from mindformers import MindFormerConfig
from mindformers.trainer.utils import load_distributed_checkpoint
from mindformers import LlamaConfig
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.experimental.infer.core.utils import generate_state_dict
from mindformers.experimental.parallel_core.pynative.utils import save_strategy_file
from mindformers import logger

# mindrlhf
from mindrlhf.utils import transfer_from_str_to_bool
from mindrlhf.models.qwen2.qwen2_tokenizer import Qwen2Tokenizer
from mindrlhf.models.grpo_models import CausalLMHybrid, GRPOModelInfer
from mindrlhf.utils.configs import (
    combine_grpo_config,
)
from mindrlhf.utils.utils import get_valid_length_each_example
from mindrlhfx.worker.worker import Worker


class InferWorker(Worker):
    '''
    This class generates responses.
    '''

    def __init__(self, grpo_config, sft_path_infer, args):
        super().__init__()
        logger.info("init InferWorker")
        self.args = args
        sft_config_infer = MindFormerConfig(sft_path_infer)
        sft_config_infer.use_parallel = args.use_parallel
        enable_compile_cache = transfer_from_str_to_bool(args.enable_compile_cache)
        os.environ["RUN_MODE"] = sft_config_infer.run_mode

        # Reentrancy protection for distributed init.
        if not GlobalComm.INITED:
            logger.info(f"launch actor roll out sft_config_infer.use_parallel {sft_config_infer.use_parallel}")
            build_context(sft_config_infer)
        build_parallel_config(sft_config_infer)
        context.set_context(
            enable_compile_cache=enable_compile_cache,
            compile_cache_path="./generate_cache"
        )

        # init sft infer model
        sft_config_infer.model.model_config.parallel_config = (
            sft_config_infer.parallel_config
        )
        sft_model_config_infer = LlamaConfig(**sft_config_infer.model.model_config)
        sft_model_config_infer.checkpoint_name_or_path = args.load_sft_checkpoint_infer
        sft_model_config_infer.model_name = "llama"

        self.grpo_config = combine_grpo_config(grpo_config, sft_model_config_infer)
        self.sft_ckpt_path_infer = sft_model_config_infer.checkpoint_name_or_path
        # Must set this to None before building policy model.
        sft_model_config_infer.checkpoint_name_or_path = None

        context.set_auto_parallel_context(parallel_mode="stand_alone", full_batch=False)
        policy_model = CausalLMHybrid(sft_model_config_infer, self.grpo_config)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True)
        self.grpo_model_infer = GRPOModelInfer(self.grpo_config, policy_model)
        self.grpo_model_infer.set_train(False)
        self.sft_model_config_infer = sft_model_config_infer
        self.infer_pp_stage = sft_model_config_infer.parallel_config.pipeline_stage or 1
        self.grpo_model_infer.grpo_model.policy_model.model.add_flags_recursive(is_first_iteration=True)
        self.grpo_model_infer.grpo_model.policy_model.model.set_train(False)
        self.on_device = True

    def model(self):
        return self.grpo_model_infer

    def get_updated_grpo_config(self):
        return self.grpo_config

    def get_infer_dp(self):
        return self.sft_model_config_infer.parallel_config.data_parallel

    def _allgather_data(self, batch_input, data_parallel_size, padding_length=128):
        """
        allgather_data
        """
        lengths = []
        padded_arrays = []
        local_bs = len(batch_input)
        for array in batch_input:
            lengths.append(len(array))
            padded_array = [0] * padding_length
            padded_array[:len(array)] = array
            padded_arrays.append(padded_array)
        padded_arrays = Tensor(padded_arrays).astype(mindspore.int32)
        lengths = Tensor(lengths).astype(mindspore.int32)
        all_padded_arrays, _ = D.comm_func.all_gather_into_tensor(padded_arrays)
        all_lengths, _ = D.comm_func.all_gather_into_tensor(lengths)

        all_lengths = all_lengths.asnumpy()
        all_padded_arrays = all_padded_arrays.asnumpy()

        world_size = D.get_group_size()
        all_other_group_size = world_size // data_parallel_size
        output_batch = []
        for i in range(0, world_size * local_bs, all_other_group_size * local_bs):
            for k in range(local_bs):
                global_idx = i + k
                output_batch.append(list(all_padded_arrays[global_idx][:all_lengths[global_idx]]))

        return output_batch

    def post_process_infer_outputs(self, results):
        right_padding_responses, responses_mask, left_padding_prompts, prompts_mask = results
        # allgather data
        right_padding_responses_batch = self._allgather_data(right_padding_responses, self.sft_model_config_infer.parallel_config.data_parallel,
                                                             padding_length=self.grpo_config.max_decode_length)
        responses_mask_batch = self._allgather_data(responses_mask, self.sft_model_config_infer.parallel_config.data_parallel,
                                                    padding_length=self.grpo_config.max_decode_length)
        left_padding_prompts_batch = self._allgather_data(left_padding_prompts, self.sft_model_config_infer.parallel_config.data_parallel,
                                                          padding_length=self.grpo_config.seq_length - self.grpo_config.max_decode_length)
        prompts_mask_batch = self._allgather_data(prompts_mask, self.sft_model_config_infer.parallel_config.data_parallel,
                                                  padding_length=self.grpo_config.seq_length - self.grpo_config.max_decode_length)
        right_padding_responses = np.array(right_padding_responses_batch).astype(np.int32)
        responses_mask = np.array(responses_mask_batch).astype(np.int32)
        left_padding_prompts = np.array(left_padding_prompts_batch).astype(np.int32)
        prompts_mask = np.array(prompts_mask_batch).astype(np.int32)
        return right_padding_responses, responses_mask, left_padding_prompts, prompts_mask

    # For SPMD, developer could call 'post_process_infer_outputs' to process data.
    # For MPMD, data should be collected to driver process and dispatch to other ray actors.
    def generate(self, input_ids):
        context.set_auto_parallel_context(pipeline_stages=self.infer_pp_stage,
                                          parallel_mode="stand_alone", full_batch=False)
        np.set_printoptions(threshold=1024)
        tokenizer = Qwen2Tokenizer(self.args.vocab_path, self.args.merges_file_path,
                                   add_bos_token=False, add_eos_token=False)

        def print_data(data, tokenizer, name):
            decoded_str2 = tokenizer.decode(data)
            logger.info(f"{name} strs are {decoded_str2}")

        logger.info("input ids are {}".format(print_data(input_ids.asnumpy().tolist(), tokenizer, "input_ids")))
        logger.info(f"input ids for precision is {input_ids.asnumpy()}")
        """ Policy model generates responses for a batch of prompts. """
        input_ids_numpy = input_ids.asnumpy()

        _, max_valid_length = get_valid_length_each_example(
            input_ids_numpy, self.grpo_model_infer.grpo_model.pad_token_id)  # get valid length and max length in a batch

        generate_begin_time = time.time()
        logger.info(f"input_ids shape {input_ids.shape}")
        outputs = self.grpo_model_infer.grpo_model.policy_model.model.generate(input_ids_numpy[:, :max_valid_length],
                                                                               max_new_tokens=self.grpo_config.max_decode_length,
                                                                               do_sample=True)

        logger.info(f"outputs for precision is {outputs}")
        logger.info(f"Generating elapsed time: {time.time() - generate_begin_time}")

        input_ids_list = input_ids_numpy.tolist()
        num_sample = len(input_ids_list)
        left_padding_prompts = np.ones((num_sample, self.grpo_config.max_prompt_length)) * \
            self.grpo_config.pad_token_id  # 初始化存储prompt的数组，序列长度最大为max_prompt_length
        right_padding_responses = np.ones((num_sample, self.grpo_config.max_decode_length)) * \
            self.grpo_config.pad_token_id  # 初始化存储response的数组，序列长度最大为max_decode_length
        prompt_len = (np.array(input_ids_list) != self.grpo_config.pad_token_id).astype(
            int).sum(1)  # 计算每个样本的prompt长度（不包含padding token)

        for i in range(num_sample):
            # 只包含response, 范围是从 "prompt结束位置" 到 "prompt结束位置+最大生成长度"
            response = outputs[i][prompt_len[i]: prompt_len[i]+self.grpo_config.max_decode_length]
            right_padding_responses[i, :len(response)] = response

            left_padding_prompts[i, self.grpo_config.max_prompt_length-prompt_len[i]:] = input_ids_list[i][:prompt_len[i]]  # 整个batch的样本右对齐（左侧进行padding）

        responses_mask = (right_padding_responses != self.grpo_config.pad_token_id).astype(np.int32)
        prompts_mask = (left_padding_prompts != self.grpo_config.pad_token_id).astype(np.int32)

        print_data(right_padding_responses.astype(np.int32).tolist(), tokenizer, "right_padding_responses")
        print_data(responses_mask.tolist(), tokenizer, "responses_mask")
        print_data(left_padding_prompts.astype(np.int32).tolist(), tokenizer, "left_padding_prompts")
        print_data(prompts_mask.tolist(), tokenizer, "prompts_mask")

        logger.info(
            f"precision return value is {right_padding_responses.astype(np.int32)}, {responses_mask}, {left_padding_prompts.astype(np.int32)}, {prompts_mask}")
        return right_padding_responses.astype(np.int32), responses_mask, left_padding_prompts.astype(np.int32), prompts_mask

    def generate_strategy(self):
        stage_name = 'infer'
        static_dict = generate_state_dict(self.grpo_model_infer.grpo_model.policy_model.model)
        save_strategy_file(static_dict, f"../../strategy/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt")
        stage_name = 'other'
        context.set_auto_parallel_context(
            strategy_ckpt_config={
                "save_file":
                    f"../../strategy/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt"})
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True)

    def offload(self):
        if self.on_device is False:
            return
        logger.info(f'before offload stf infer {ms.hal.memory_stats()}')
        for param in self.grpo_model_infer.grpo_model.get_parameters(expand=True):
            param._offload()
        logger.info(f'after offload stf infer {ms.hal.memory_stats()}')
        self.on_device = False

    def load(self):
        if self.on_device:
            return
        logger.info(f'before load stf infer {ms.hal.memory_stats()}')
        for param in self.grpo_model_infer.grpo_model.get_parameters(expand=True):
            param._load()
        logger.info(f'after load stf infer {ms.hal.memory_stats()}')
        self.on_device = True

    def load_checkpoint(self):
        load_ckpt_func = load_distributed_checkpoint if self.grpo_config.use_parallel else ms.load_checkpoint
        logger.info(f"self.grpo_config.use_parallel is {self.grpo_config.use_parallel} {load_ckpt_func}")
        if self.sft_ckpt_path_infer:
            param_dict = load_ckpt_func(self.sft_ckpt_path_infer)
            new_param_dict = {'grpo_model.policy_model.model.' + k: v for k, v in param_dict.items()}

            logger.info(f"begin to load infer policy model from: {self.sft_ckpt_path_infer}")
            logger.info("###############")
            logger.info(f"self.grpo_config.use_parallel: {self.grpo_config.use_parallel}")
            logger.info(new_param_dict.keys())
            for _, param in self.grpo_model_infer.grpo_model.policy_model.parameters_and_names():
                logger.info(f"infer model para names:   {param.name}")
            param_not_load, ckpt_not_load = ms.load_param_into_net(self.grpo_model_infer.grpo_model.policy_model,
                                                                   new_param_dict)
            logger.info(f"param not load: {param_not_load}")
            logger.info(f"ckpt not load: {ckpt_not_load}")
