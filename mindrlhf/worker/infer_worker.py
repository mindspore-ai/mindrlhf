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
"""Infererence Worker."""

# python
import os
import time
from glob import glob
import numpy as np

# mindspore
import mindspore as ms
from mindspore import Tensor
from mindspore.communication import GlobalComm, get_rank
from mindspore import context
from mindspore import communication as D
from mindspore.common.api import _pynative_executor

# mindformers
from mindformers import MindFormerConfig
from mindformers.trainer.utils import load_distributed_checkpoint
from mindformers import LlamaConfig
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.experimental.infer.core.utils import generate_state_dict
from mindformers.experimental.parallel_core.pynative.utils import save_strategy_file
from mindformers import logger
from mindformers.models.build_tokenizer import build_tokenizer

# mindrlhf
from mindrlhf.utils import transfer_from_str_to_bool
from mindrlhf.models.qwen2.qwen2_tokenizer import Qwen2Tokenizer
from mindrlhf.models.grpo_models import CausalLMHybrid, GRPOModelInfer
from mindrlhf.utils.configs import (
    combine_grpo_config,
)
from mindrlhf.configs.grpo_configs import VllmMode
from mindrlhf.utils.utils import get_valid_length_each_example
from mindrlhf.worker.worker import Worker


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
        self.sft_model_config_infer = sft_model_config_infer

        if args.custom_model_name == "llama":
            sft_config_infer.processor.tokenizer.tokenizer_file = args.vocab_path
            self.tokenizer = build_tokenizer(sft_config_infer.processor.tokenizer)
        else:
            self.tokenizer = Qwen2Tokenizer(
                args.vocab_path, args.merges_file_path, add_bos_token=False, add_eos_token=False)

        context.set_auto_parallel_context(parallel_mode="stand_alone", full_batch=False)
        self.use_vllm = grpo_config.use_vllm
        policy_model = None
        if self.use_vllm != VllmMode.ORIGIN:
            # vllm
			# pylint: disable=W0611
            import vllm_mindspore
            _pynative_executor.set_async_for_graph(False)
            import mindrlhf.third_party.vllm.qwen2
            import mindrlhf.third_party.vllm.ascend
            from mindrlhf.third_party.vllm.llm import LLM
            from vllm import SamplingParams
            hf_config = self.build_qwen_hf_config()
            self.tokenizer.max_token_id = max(self.tokenizer.get_vocab().values())
            # 初始化vllm
            logger.info(f"init LLM, block_size: {sft_model_config_infer.block_size}, "
                        f"max_model_len = {self.grpo_config.max_model_len}, "
                        f"max_num_batched_tokens: {self.grpo_config.max_num_batched_tokens}, "
                        f"max_num_seqs: {self.grpo_config.max_num_seqs}, "
                        f"num_scheduler_steps: {self.grpo_config.num_scheduler_steps}, "
                        f"gpu_memory_utilization: {self.grpo_config.gpu_memory_utilization}")
            vllm_start_time = time.time()
            self.inference_engine = LLM(tokenizer=self.tokenizer,
                                        model_hf_config=hf_config,
                                        tensor_parallel_size=sft_model_config_infer.parallel_config.model_parallel,
                                        dtype="bfloat16",
                                        block_size=sft_model_config_infer.block_size,
                                        skip_tokenizer_init=False,
                                        max_model_len=self.grpo_config.max_model_len,             # 上下文总长，影响prompt长度和生成长度，小于max_num_batched_tokens
                                        max_num_batched_tokens=self.grpo_config.max_num_batched_tokens,
                                        max_num_seqs=self.grpo_config.max_num_seqs,
                                        num_scheduler_steps=self.grpo_config.num_scheduler_steps,
                                        gpu_memory_utilization=self.grpo_config.gpu_memory_utilization
                                        )
            logger.info(f"init LLM end, cost time: {time.time() - vllm_start_time}")
            logger.info(f"temperature: {self.grpo_config.temperature}, "
                        f"repetition_penalty: {self.grpo_config.repetition_penalty}, "
                        f"top_p: {self.grpo_config.top_p}, top_k: {self.grpo_config.top_k}, "
                        f"stop_token_ids: {self.grpo_config.eos_token_id}, "
                        f"max_tokens: {self.grpo_config.max_decode_length}, "
                        f"detokenize: {self.grpo_config.detokenize}")
            vllm_start_time = time.time()
            self.sampling_params = SamplingParams(
                repetition_penalty=self.grpo_config.repetition_penalty,
                temperature=self.grpo_config.temperature,
                top_p=self.grpo_config.top_p,
                top_k=self.grpo_config.top_k,
                stop_token_ids=self.grpo_config.eos_token_id,
                max_tokens=self.grpo_config.max_decode_length,
                min_tokens=4,
                detokenize=self.grpo_config.detokenize
            )
            logger.info(f"init SamplingParams end, cost time: {time.time() - vllm_start_time}")
            policy_model = self.inference_engine.get_model()
            policy_model.dp = sft_model_config_infer.parallel_config.data_parallel
        else:
            # no vllm
            sft_model_config_infer.name = "grpo_infer"
            policy_model = CausalLMHybrid(sft_model_config_infer, self.grpo_config)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True)
        self.grpo_model_infer = GRPOModelInfer(self.grpo_config, policy_model)
        self.grpo_model_infer.set_train(False)
        self.infer_pp_stage = sft_model_config_infer.parallel_config.pipeline_stage or 1
        if self.use_vllm == VllmMode.ORIGIN:
            self.grpo_model_infer.grpo_model.policy_model.model.add_flags_recursive(is_first_iteration=True)
            self.grpo_model_infer.grpo_model.policy_model.model.set_train(False)
        else:
            self.grpo_model_infer.grpo_model.policy_model.add_flags_recursive(is_first_iteration=True)
            self.grpo_model_infer.grpo_model.policy_model.set_train(False)
        self.on_device = True
        self.save_strategy_dir = grpo_config.save_strategy_dir

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
        padded_arrays = Tensor(padded_arrays).astype(ms.int32)
        lengths = Tensor(lengths).astype(ms.int32)
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
        """ post_process_infer_outputs """
        right_padding_responses, responses_mask, left_padding_prompts, prompts_mask = results
        # allgather data
        right_padding_responses_batch = self._allgather_data(right_padding_responses,
                                                             self.sft_model_config_infer.parallel_config.data_parallel,
                                                             padding_length=self.grpo_config.max_decode_length)
        responses_mask_batch = self._allgather_data(responses_mask,
                                                    self.sft_model_config_infer.parallel_config.data_parallel,
                                                    padding_length=self.grpo_config.max_decode_length)
        left_padding_prompts_batch = self._allgather_data(
            left_padding_prompts,
            self.sft_model_config_infer.parallel_config.data_parallel,
            padding_length=self.grpo_config.seq_length - self.grpo_config.max_decode_length
        )
        prompts_mask_batch = self._allgather_data(
            prompts_mask, self.sft_model_config_infer.parallel_config.data_parallel,
            padding_length=self.grpo_config.seq_length - self.grpo_config.max_decode_length
        )
        right_padding_responses = np.array(right_padding_responses_batch).astype(np.int32)
        responses_mask = np.array(responses_mask_batch).astype(np.int32)
        left_padding_prompts = np.array(left_padding_prompts_batch).astype(np.int32)
        prompts_mask = np.array(prompts_mask_batch).astype(np.int32)

        return right_padding_responses, responses_mask, left_padding_prompts, prompts_mask

    # For SPMD, developer could call 'post_process_infer_outputs' to process data.
    # For MPMD, data should be collected to driver process and dispatch to other ray actors.
    def generate(self, input_ids_numpy, max_decode_length=0):
        """ Policy model generates responses for a batch of prompts. """
        context.set_auto_parallel_context(pipeline_stages=self.infer_pp_stage,
                                          parallel_mode="stand_alone", full_batch=False)
        np.set_printoptions(threshold=1024)

        def print_data(data, tokenizer, name):
            decoded_str2 = tokenizer.decode(data)
            logger.info(f"{name} strs are {decoded_str2}")

        logger.info(f"input_ids shape {input_ids_numpy.shape}")

        valid_length_each_example, max_valid_length = get_valid_length_each_example(
            input_ids_numpy, self.grpo_model_infer.grpo_model.pad_token_id)  # get valid length and max length in a batch

        generate_begin_time = time.time()
        if max_decode_length == 0:
            max_decode_length = self.grpo_config.max_decode_length
        min_decode_length = 4
        logger.info(f"max_decode_length is {max_decode_length}")
        logger.info(f"min_decode_length is {min_decode_length}")
        if self.use_vllm == VllmMode.DEBUG:
            # use vllm model
            logger.info("infer without vllm, use vllm model")
            outputs = self.grpo_model_infer.grpo_model.policy_model.generate(input_ids_numpy[:, :max_valid_length],
                                                                             max_new_tokens=max_decode_length,
                                                                             min_new_tokens=min_decode_length,
                                                                             do_sample=True)
            logger.info("infer without vllm end, use vllm model")
        elif self.use_vllm == VllmMode.ORIGIN:
            logger.info("infer without vllm, not use vllm model")
            outputs = self.grpo_model_infer.grpo_model.policy_model.model.generate(input_ids_numpy[:, :max_valid_length],
                                                                                   max_new_tokens=max_decode_length,
                                                                                   min_new_tokens=min_decode_length,
                                                                                   do_sample=True)
            logger.info("infer without vllm end, not use vllm model")
        else:
            logger.info("start vllm")
            prompt = input_ids_numpy[:, :max_valid_length]
            vllm_prompt = prompt.tolist()
            token_ids = self.inference_engine.pre_process_inputs(vllm_prompt, valid_length_each_example)
            outputs = self.inference_engine.generate(
                prompts=None,
                sampling_params=self.sampling_params,
                prompt_token_ids=token_ids,
                use_tqdm=False)
            logger.info("end vllm")
            outputs = outputs[0]

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
            if self.use_vllm == VllmMode.DEBUG or self.use_vllm == VllmMode.ORIGIN:
                response = outputs[i][prompt_len[i]: prompt_len[i] + self.grpo_config.max_decode_length]
            else:
                # vllm output中不包含prompt
                response = outputs[i]
            right_padding_responses[i, :len(response)] = response

            left_padding_prompts[i, self.grpo_config.max_prompt_length-prompt_len[i]:] = input_ids_list[i][:prompt_len[i]]  # 整个batch的样本右对齐（左侧进行padding）

        responses_mask = (right_padding_responses != self.grpo_config.pad_token_id).astype(np.int32)
        prompts_mask = (left_padding_prompts != self.grpo_config.pad_token_id).astype(np.int32)

        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True)
        return (
            right_padding_responses.astype(np.int32), responses_mask,
            left_padding_prompts.astype(np.int32), prompts_mask
        )

    def generate_strategy(self):
        """ generate_strategy """
        context.set_auto_parallel_context(pipeline_stages=self.infer_pp_stage,
                                          parallel_mode="stand_alone", full_batch=False)
        stage_name = 'infer'
        strategy_path = self.grpo_config.save_strategy_dir
        ms.mint.distributed.barrier()
        if self.use_vllm == VllmMode.ORIGIN:
            static_dict = generate_state_dict(self.grpo_model_infer.grpo_model.policy_model.model)
        else:
            static_dict = generate_state_dict(self.grpo_model_infer.grpo_model.policy_model)
        save_strategy_file(static_dict,
                           f"{strategy_path}/strategy_file/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt")
        stage_name = 'other'
        context.set_auto_parallel_context(
            strategy_ckpt_config={
                "save_file":
                    f"{strategy_path}/strategy_file/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt"})
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True)

    def offload(self, free_kv_cache=False):
        """offload infer checkpoint"""
        if self.on_device is False:
            return
        logger.info(f'before offload stf infer')
        if self.use_vllm == VllmMode.ORIGIN:
            for param in self.grpo_model_infer.grpo_model.get_parameters(expand=True):
                # pylint: disable=W0212
                param._offload()
        else:
            if free_kv_cache:
                self.inference_engine.free_cache_engine()
            for param in self.grpo_model_infer.grpo_model.get_parameters(expand=True):
                # pylint: disable=W0212
                if free_kv_cache and "paged_attention_mgr" in param.name:
                    continue
                param._offload()
        logger.info(f'after offload stf infer')
        self.on_device = False

    def load(self, init_kv_cache=False):
        """ load infer checkpoint """
        if self.on_device:
            return
        logger.info(f'before load stf infer {ms.hal.memory_stats()}')
        if self.use_vllm == VllmMode.ORIGIN:
            for param in self.grpo_model_infer.grpo_model.get_parameters(expand=True):
                # pylint: disable=W0212
                param._load()
        else:
            if init_kv_cache:
                self.inference_engine.init_cache_engine()
            for param in self.grpo_model_infer.grpo_model.get_parameters(expand=True):
                if not init_kv_cache and "paged_attention_mgr" in param.name:
                    continue
                # pylint: disable=W0212
                param._load()
            logger.info(f'after load stf infer {ms.hal.memory_stats()}')
        self.on_device = True

    # pylint: disable=E0402
    # pylint: disable=R1710
    def load_checkpoint(self):
        """ load infer checkpoint """
        if self.args.load_ckpt_format == "safetensors":
            self.on_device = True
            return self._load_checkpoint_safetensors()
        load_ckpt_func = load_distributed_checkpoint if self.grpo_config.use_parallel else ms.load_checkpoint
        logger.info(f"self.grpo_config.use_parallel is {self.grpo_config.use_parallel} {load_ckpt_func}")
        if self.sft_ckpt_path_infer:
            self.on_device = True
            param_dict = load_ckpt_func(self.sft_ckpt_path_infer)
            if self.use_vllm == VllmMode.ORIGIN:
                new_param_dict = {'grpo_model.policy_model.model.' + k: v for k, v in param_dict.items()}
            else:
                new_param_dict = {'grpo_model.policy_model.' + k: v for k, v in param_dict.items()}

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

    def build_qwen_hf_config(self):
        """ build_qwen_hf_config """
        import json
        from transformers import AutoConfig
        # build qwen hf config 硬编码

        print("hf_config_path: ", self.grpo_config.hf_config_path, flush=True)
        qwen_hf_config = {}
        qwen_hf_config["architectures"] = ["Qwen2ForCausalLM"]
        qwen_hf_config["attention_dropout"] = 0.0
        qwen_hf_config["bos_token_id"] = 151643 # 硬编码
        qwen_hf_config["eos_token_id"] = 151643 # 硬编码
        qwen_hf_config["hidden_act"] = "silu"
        qwen_hf_config["hidden_size"] = self.sft_model_config_infer.hidden_size
        qwen_hf_config["initializer_range"] = 0.02 # 硬编码，不知道对应哪一项
        qwen_hf_config["intermediate_size"] = self.sft_model_config_infer.intermediate_size
        qwen_hf_config["max_position_embeddings"] = self.sft_model_config_infer.max_position_embeddings
        qwen_hf_config["max_window_layers"] = self.sft_model_config_infer.num_layers
        qwen_hf_config["model_type"] = "qwen2"
        qwen_hf_config["num_attention_heads"] = self.sft_model_config_infer.num_heads
        qwen_hf_config["num_hidden_layers"] = self.sft_model_config_infer.num_layers
        qwen_hf_config["num_key_value_heads"] = self.sft_model_config_infer.n_kv_heads
        qwen_hf_config["rms_norm_eps"] = 1e-06
        qwen_hf_config["rope_theta"] = 1000000.0
        qwen_hf_config["sliding_window"] = self.sft_model_config_infer.max_position_embeddings # 不明白数值是131072
        qwen_hf_config["tie_word_embeddings"] = False
        qwen_hf_config["torch_dtype"] = "bfloat16" # 先硬编码
        qwen_hf_config["transformers_version"] = "4.46.1"
        qwen_hf_config["use_cache"] = True
        qwen_hf_config["use_sliding_window"] = False
        qwen_hf_config["vocab_size"] = self.sft_model_config_infer.vocab_size

        json_str = json.dumps(qwen_hf_config, indent=4)
        if get_rank() == 0:
            with open(self.grpo_config.hf_config_path, "w", encoding="utf-8") as f:
                f.write(json_str)
        ms.mint.distributed.barrier()

        hf_config = AutoConfig.from_pretrained(os.path.dirname(self.grpo_config.hf_config_path))
        print("hf config for vllm: ", hf_config, flush=True)

        return hf_config

    def _load_checkpoint_safetensors(self):
        """ load infer checkpoint safetensors """
        network = self.grpo_model_infer.grpo_model.policy_model
        name_map = None
        try:
            load_checkpoint_files = glob(
                os.path.join(self.sft_ckpt_path_infer, f"*.safetensors"))
            load_checkpoint_files.sort()
            name_map = network.obtain_name_map(load_checkpoint_files)
            if self.use_vllm != VllmMode.ORIGIN:
                prefix = "grpo_model.policy_model."
                name_map = {f"{prefix}{key}": value for key, value in name_map.items()}
        except Exception as e:
            raise TypeError(f"Please complete abstract function obtain_name_map. Details: {e}") from e

        strategy_path = os.path.join(self.grpo_config.save_strategy_dir,
                                     'merge_strategy/infer_policy_merged_strategy.ckpt')
        ms.load_distributed_checkpoint(
            network=network,
            predict_strategy=strategy_path,
            unified_safetensors_dir=self.sft_ckpt_path_infer,
            format='safetensors',
            name_map=name_map
        )
