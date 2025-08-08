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
"""Inference Worker."""
import os
import time
from omegaconf import DictConfig, OmegaConf

import numpy as np

from mindformers import LlamaConfig
from mindformers import MindFormerConfig
from mindformers import logger
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.parallel_core.inference.utils import generate_state_dict
from mindformers.trainer.utils import load_distributed_checkpoint

import mindspore as ms
from mindspore import Tensor
from mindspore import context
from mindspore import communication as D
from mindspore.communication import GlobalComm, get_rank

import mindrlhf.utils.reshard_optimizer as reshard_optimizer
from mindrlhf.configs.grpo_configs import VllmMode
from mindrlhf.models.grpo_models import CausalLMHybrid, GRPOModelInfer
from mindrlhf.utils import TimeConsumingCollector
from mindrlhf.utils.strategy_utils import save_strategy_file
from mindrlhf.utils.utils import get_valid_length_each_example, get_dp_rank, load_safetensors, enable_pynative_async
from mindrlhf.worker.worker import Worker
from research.deepseek3.deepseek3_config import DeepseekV3Config

DTYPE_STR = {ms.float16: "float16", ms.bfloat16: "bfloat16", ms.float32: "float32", ms.int8: "int8", ms.uint8: "uint8"}


class InferWorker(Worker):
    """
    This class generates responses.

    Args:
        grpo_config (DictConfig): GRPO config instance.
        args: Args will be removed in next release version.
        **kwargs: Key word args.
    """

    SAVED_MODEL_CONFIG_YAML = "saved_infer_model_config.yaml"

    def __init__(self, grpo_config: DictConfig, **kwargs):
        """Init infer worker."""
        super().__init__(config=grpo_config, worker_type=Worker.WorkerType.INFER, **kwargs)
        logger.info("Start init InferWorker")
        self.tokenizer = kwargs.get("tokenizer")
        self.grpo_config = grpo_config
        sft_config_infer = MindFormerConfig(**OmegaConf.to_container(self.reconstructed_model_config))

        os.environ["RUN_MODE"] = sft_config_infer.run_mode
        self.dump_mf_conf_to_yaml(sft_config_infer, "mf_infer_conf.yaml")
        if not GlobalComm.INITED:
            build_context(sft_config_infer)
        build_parallel_config(sft_config_infer)
        context.set_context(
            enable_compile_cache=self.grpo_config.rl_config.enable_compile_cache,
            compile_cache_path=self.grpo_config.rl_config.compile_cache_path,
        )

        if self.model_name in ["qwen2.5", "llama"]:
            self.sft_model_config_infer = LlamaConfig(**sft_config_infer.model.model_config)
            self.sft_model_config_infer.model_name = "llama"
        elif self.model_name == "deepseek":
            self.sft_model_config_infer = DeepseekV3Config(**sft_config_infer.model.model_config)
            self.sft_model_config_infer.model_name = "deepseek_infer"
        else:
            raise ValueError(f"model_name should in ['qwen2.5', 'llama','deepseek'], " f"but get {self.model_name}")

        self.sft_ckpt_path_infer = self.grpo_config.generate_config.load
        # Must set this to None before building policy model.
        self.sft_model_config_infer.checkpoint_name_or_path = None
        self.dp_rank_id = get_dp_rank(self.sft_model_config_infer.parallel_config.data_parallel)

        self.dump_mf_conf_to_yaml(self.sft_model_config_infer, self.SAVED_MODEL_CONFIG_YAML)

        context.set_auto_parallel_context(parallel_mode="stand_alone", full_batch=False)
        sim_level = os.getenv("MS_SIMULATION_LEVEL")
        if sim_level:
            logger.warning(f"MS_SIMULATION_LEVEL is set to {sim_level}, will not use vllm")
            self.use_vllm = VllmMode.ORIGIN
        else:
            self.use_vllm = grpo_config.generate_config.use_vllm
        if self.use_vllm != VllmMode.ORIGIN:
            self.policy_model = self.__init_use_vllm()
            self.old_phase = self.policy_model.phase
            self.policy_model.dp = self.sft_model_config_infer.parallel_config.data_parallel
        else:
            # no vllm
            self.policy_model = CausalLMHybrid(self.sft_model_config_infer, self.grpo_config)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True)
        self.grpo_model_infer = GRPOModelInfer(self.grpo_config, self.policy_model)
        self.grpo_model_infer.set_train(False)
        self.infer_pp_stage = self.sft_model_config_infer.parallel_config.pipeline_stage or 1
        if self.use_vllm == VllmMode.ORIGIN:
            self.grpo_model_infer.grpo_model.policy_model.model.add_flags_recursive(is_first_iteration=True)
            self.grpo_model_infer.grpo_model.policy_model.model.set_train(False)
        else:
            self.grpo_model_infer.grpo_model.policy_model.add_flags_recursive(is_first_iteration=True)
            self.grpo_model_infer.grpo_model.policy_model.set_train(False)
        self.on_device = True

    @property
    def load_ckpt_format(self):
        return self.grpo_config.rl_config.load_ckpt_format

    def refresh_policy_model_phase(self):
        """
        refresh policy model phase
        """
        if self.use_vllm != VllmMode.ORIGIN:
            self.policy_model.phase = self.old_phase

    @enable_pynative_async
    def __init_use_vllm(self):
        """
        init_vllm
        """
        # pylint: disable=W0611
        from mindrlhf.third_party.vllm import package_version, LLM
        from vllm import SamplingParams

        if self.grpo_config.rl_config.resume_training:
            logger.warning("Enable resume_training, skip loading infer model weights")
            from vllm_mindspore.model_executor.models.mf_models.qwen2 import Qwen2ForCausalLM
            from mindrlhf.third_party.vllm.qwen2 import load_weights

            Qwen2ForCausalLM.load_weights = load_weights
            self.sft_ckpt_path_infer = self.grpo_config.rl_config.tokenizer_dir

        self.tokenizer.max_token_id = max(self.tokenizer.get_vocab().values())
        # 初始化vllm
        logger.info(
            f"init LLM, block_size: {self.grpo_config.generate_config.block_size}, "
            f"max_model_len = {self.grpo_config.generate_config.max_model_len}, "
            f"max_num_batched_tokens: {self.grpo_config.generate_config.max_num_batched_tokens}, "
            f"max_num_seqs: {self.grpo_config.generate_config.max_num_seqs}, "
            f"num_scheduler_steps: {self.grpo_config.generate_config.num_scheduler_steps}, "
            f"gpu_memory_utilization: {self.grpo_config.generate_config.gpu_memory_utilization}, "
            f"seed: {self.dp_rank_id}"
        )
        if package_version.startswith("0.8"):
            from mindrlhf.third_party.vllm.vllm_v_general import initialize_parallel_state

            initialize_parallel_state(
                tensor_model_parallel_size=self.sft_model_config_infer.parallel_config.model_parallel
            )
            vllm_start_time = time.time()
            if self.load_ckpt_format not in ["hf_safetensors", "ms_safetensors"]:
                raise ValueError(
                    f"For vllm {package_version}, "
                    f"the infer model load_ckpt_format must be hf_safetensors or ms_safetensors."
                )
            if self.grpo_config.generate_config.num_scheduler_steps > 1:
                logger.warning(f"For VLLM V1, num_scheduler_steps > 1 is not supported, set it to 1.")
                self.grpo_config.generate_config.num_scheduler_steps = 1
            self.inference_engine = LLM(
                model=self.sft_ckpt_path_infer,  # path to hf model
                tensor_parallel_size=self.sft_model_config_infer.parallel_config.model_parallel,
                distributed_executor_backend="external_launcher",
                dtype="bfloat16",
                block_size=self.grpo_config.generate_config.block_size,
                skip_tokenizer_init=False,
                max_model_len=self.grpo_config.generate_config.max_model_len,
                max_num_batched_tokens=self.grpo_config.generate_config.max_num_batched_tokens,
                max_num_seqs=self.grpo_config.generate_config.max_num_seqs,
                num_scheduler_steps=self.grpo_config.generate_config.num_scheduler_steps,
                gpu_memory_utilization=self.grpo_config.generate_config.gpu_memory_utilization,
                trust_remote_code=self.grpo_config.generate_config.trust_remote_code,
                enable_prefix_caching=False,  # FIXME: not support prefix caching now.
                seed=self.dp_rank_id,
            )
            logger.info(f"init LLM end, cost time: {time.time() - vllm_start_time}")
            model_runner = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner
            policy_model = model_runner.model.network
        else:
            raise ValueError(f"Not support vllm version: {package_version}")
        logger.info(f"init LLM end, cost time: {time.time() - vllm_start_time}")
        logger.info(
            f"temperature: {self.grpo_config.generate_config.sampling_config.temperature}, "
            f"repetition_penalty: {self.grpo_config.generate_config.sampling_config.repetition_penalty}, "
            f"top_p: {self.grpo_config.generate_config.sampling_config.top_p}, "
            f"top_k: {self.grpo_config.generate_config.sampling_config.top_k}, "
            f"stop_token_ids: {self.grpo_config.generate_config.sampling_config.eos_token_id}, "
            f"max_tokens: {self.grpo_config.generate_config.sampling_config.max_tokens}, "
            f"detokenize: {self.grpo_config.generate_config.sampling_config.detokenize}"
        )
        vllm_start_time = time.time()
        self.sampling_params = SamplingParams(
            repetition_penalty=self.grpo_config.generate_config.sampling_config.repetition_penalty,
            temperature=self.grpo_config.generate_config.sampling_config.temperature,
            top_p=self.grpo_config.generate_config.sampling_config.top_p,
            top_k=self.grpo_config.generate_config.sampling_config.top_k,
            stop_token_ids=self.grpo_config.generate_config.sampling_config.eos_token_id,
            max_tokens=self.grpo_config.generate_config.sampling_config.max_tokens,
            min_tokens=self.grpo_config.generate_config.sampling_config.min_tokens,
            detokenize=self.grpo_config.generate_config.sampling_config.detokenize,
        )
        logger.info(f"init SamplingParams end, cost time: {time.time() - vllm_start_time}")
        return policy_model

    def model(self):
        """Return model"""
        return self.grpo_model_infer

    def get_updated_grpo_config(self):
        """Get updated grpo config."""
        return self.grpo_config

    def get_infer_dp(self):
        """Get infer dp."""
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
            padded_array[: len(array)] = array
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
        if reshard_optimizer.OPT_COMMUNICATION_GROUPS:
            collect_range = [_ * local_bs for _ in reshard_optimizer.OPT_COMMUNICATION_GROUPS["dp"][0]]
        else:
            collect_range = range(0, world_size * local_bs, all_other_group_size * local_bs)

        for i in collect_range:
            for k in range(local_bs):
                global_idx = i + k
                output_batch.append(list(all_padded_arrays[global_idx][: all_lengths[global_idx]]))

        return output_batch

    def post_process_infer_outputs(self, results):
        """post_process_infer_outputs"""
        with TimeConsumingCollector("post process infer outputs"):
            right_padding_responses, responses_mask, left_padding_prompts, prompts_mask = results
            max_tokens = self.grpo_config.generate_config.sampling_config.max_tokens
            # allgather data
            right_padding_responses_batch = self._allgather_data(
                right_padding_responses,
                self.sft_model_config_infer.parallel_config.data_parallel,
                padding_length=max_tokens,
            )
            responses_mask_batch = self._allgather_data(
                responses_mask, self.sft_model_config_infer.parallel_config.data_parallel, padding_length=max_tokens
            )
            left_padding_prompts_batch = self._allgather_data(
                left_padding_prompts,
                self.sft_model_config_infer.parallel_config.data_parallel,
                padding_length=self.grpo_config.rl_config.seq_length - max_tokens,
            )
            prompts_mask_batch = self._allgather_data(
                prompts_mask,
                self.sft_model_config_infer.parallel_config.data_parallel,
                padding_length=self.grpo_config.rl_config.seq_length - max_tokens,
            )
            right_padding_responses = np.array(right_padding_responses_batch).astype(np.int32)
            responses_mask = np.array(responses_mask_batch).astype(np.int32)
            left_padding_prompts = np.array(left_padding_prompts_batch).astype(np.int32)
            prompts_mask = np.array(prompts_mask_batch).astype(np.int32)
            return right_padding_responses, responses_mask, left_padding_prompts, prompts_mask

    # For SPMD, developer could call 'post_process_infer_outputs' to process data.
    # For MPMD, data should be collected to driver process and dispatch to other ray actors.
    def generate(self, input_ids_numpy, max_tokens=0):
        """Policy model generates responses for a batch of prompts."""
        ms.set_context(jit_config={"infer_boost": "on"})
        context.set_auto_parallel_context(
            pipeline_stages=self.infer_pp_stage, parallel_mode="stand_alone", full_batch=False
        )

        logger.info(f"input_ids shape {input_ids_numpy.shape}")

        valid_length_each_example, max_valid_length = get_valid_length_each_example(
            input_ids_numpy, self.grpo_model_infer.grpo_model.pad_token_id
        )  # get valid length and max length in a batch

        generate_begin_time = time.time()
        if max_tokens == 0:
            max_tokens = self.grpo_config.generate_config.sampling_config.max_tokens
        min_tokens = self.grpo_config.generate_config.sampling_config.min_tokens
        logger.info(f"max_tokens {max_tokens}")
        logger.info(f"min_tokens {min_tokens}")
        if self.use_vllm == VllmMode.DEBUG:
            # use vllm model
            logger.info("infer without vllm, use vllm model")
            outputs = self.grpo_model_infer.grpo_model.policy_model.model.generate(
                input_ids_numpy[:, :max_valid_length],
                max_new_tokens=max_tokens,
                min_new_tokens=min_tokens,
                do_sample=True,
                seed=self.dp_rank_id,
            )
            logger.info("infer without vllm end, use vllm model")
        elif self.use_vllm == VllmMode.ORIGIN:
            logger.info("infer without vllm, not use vllm model")
            outputs = self.grpo_model_infer.grpo_model.policy_model.model.generate(
                input_ids_numpy[:, :max_valid_length],
                max_new_tokens=max_tokens,
                min_new_tokens=min_tokens,
                do_sample=True,
                seed=self.dp_rank_id,
            )
            logger.info("infer without vllm end, not use vllm model")
        else:
            logger.info("start vllm")
            prompt = input_ids_numpy[:, :max_valid_length]
            vllm_prompt = prompt.tolist()
            outputs = self._vllm_generate(vllm_prompt, valid_length_each_example)
            logger.info("end vllm")

        logger.info(f"Generating elapsed time: {time.time() - generate_begin_time}")

        input_ids_list = input_ids_numpy.tolist()
        num_sample = len(input_ids_list)
        max_prompt_length = self.grpo_config.generate_config.max_prompt_length
        max_tokens = self.grpo_config.generate_config.sampling_config.max_tokens
        pad_token_id = self.grpo_config.generate_config.sampling_config.pad_token_id
        # init prompt, max length is max_prompt_length
        left_padding_prompts = np.ones((num_sample, max_prompt_length)) * pad_token_id
        # init response, max length is max_decode_length
        right_padding_responses = np.ones((num_sample, max_tokens)) * pad_token_id
        # prompt length without padding token
        prompt_len = (np.array(input_ids_list) != pad_token_id).astype(int).sum(1)

        for i in range(num_sample):
            # only response
            if self.use_vllm == VllmMode.ORIGIN:
                response = outputs[i][prompt_len[i] : prompt_len[i] + max_tokens]
            else:
                # vllm output without prompt
                response = outputs[i].outputs[0].token_ids
            right_padding_responses[i, : len(response)] = response

            # right padding
            left_padding_prompts[i, max_prompt_length - prompt_len[i] :] = input_ids_list[i][: prompt_len[i]]

        responses_mask = (right_padding_responses != pad_token_id).astype(np.int32)
        prompts_mask = (left_padding_prompts != pad_token_id).astype(np.int32)

        ms.set_context(jit_config={"infer_boost": "off"})
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True)
        return (
            right_padding_responses.astype(np.int32),
            responses_mask,
            left_padding_prompts.astype(np.int32),
            prompts_mask,
        )

    def generate_strategy(self, reshard_optimizer_):
        """generate strategy"""
        ms.set_context(jit_config={"infer_boost": "on"})
        context.set_auto_parallel_context(
            pipeline_stages=self.infer_pp_stage, parallel_mode="stand_alone", full_batch=False
        )
        stage_name = "infer"
        ms.mint.distributed.barrier()
        if self.use_vllm == VllmMode.ORIGIN:
            static_dict = generate_state_dict(self.grpo_model_infer.grpo_model.policy_model.model)
        else:
            static_dict = generate_state_dict(self.grpo_model_infer.grpo_model.policy_model)
        save_strategy_file(
            static_dict,
            reshard_optimizer_,
            f"{self.save_strategy_dir}/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt",
        )
        stage_name = "other"
        context.set_auto_parallel_context(
            strategy_ckpt_config={
                "save_file": f"{self.save_strategy_dir}/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt"
            }
        )
        ms.set_context(jit_config={"infer_boost": "off"})
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True)

    def offload(self):
        """offload stf infer"""
        if not self.on_device:
            return
        logger.info(f"before offload stf infer {ms.hal.memory_stats()}")
        with TimeConsumingCollector("offload stf infer"):
            skip_kv_cache = False
            if self.use_vllm == VllmMode.VLLM:
                self.inference_engine.free_cache_engine()
                skip_kv_cache = True
            for param in self.grpo_model_infer.grpo_model.get_parameters(expand=True):
                if skip_kv_cache and "paged_attention_mgr" in param.name:
                    continue
                # pylint: disable=W0212
                param._offload()
        logger.info(f"after offload stf infer {ms.hal.memory_stats()}")
        self.on_device = False

    def load(self, skip_kv_cache=False):
        """load stf infer"""
        if self.on_device:
            return
        logger.info(f"before load stf infer {ms.hal.memory_stats()}")
        with TimeConsumingCollector("load stf infer"):
            if self.use_vllm == VllmMode.VLLM and not skip_kv_cache:
                self._init_cache_engine()
                skip_kv_cache = True
            for param in self.grpo_model_infer.grpo_model.get_parameters(expand=True):
                if skip_kv_cache and "paged_attention_mgr" in param.name:
                    continue
                # pylint: disable=W0212
                param._load()
        logger.info(f"after load stf infer {ms.hal.memory_stats()}")
        self.on_device = True

    def load_kvcache(self):
        """load stf infer kvcache"""
        logger.info(f"before load stf infer kv cache {ms.hal.memory_stats()}")
        with TimeConsumingCollector("load stf infer kvcache"):
            if self.use_vllm == VllmMode.VLLM:
                self._init_cache_engine()
        logger.info(f"after load stf infer kv cache {ms.hal.memory_stats()}")
        self.on_device = True

    def load_checkpoint(self):
        """load_checkpoint"""
        logger.info(f"sft_ckpt_path_infer:{self.sft_ckpt_path_infer}")
        if not self.sft_ckpt_path_infer:
            return
        if not os.path.exists(self.sft_ckpt_path_infer):
            raise ValueError(f"infer model checkpoint path: {self.sft_ckpt_path_infer} not exists")

        if self.sft_ckpt_path_infer and self.load_ckpt_format in ["ms_safetensors", "hf_safetensors"]:
            self.on_device = True
            strategy_path = os.path.join(self.save_strategy_dir, "merge_strategy", "infer_policy_merged_strategy.ckpt")
            if self.use_vllm != VllmMode.ORIGIN:
                return
            network = self.grpo_model_infer.grpo_model.policy_model.model
            prefix = "grpo_model.policy_model.model."
            load_safetensors(
                self.sft_ckpt_path_infer,
                self.load_ckpt_format,
                network,
                self.grpo_model_infer.grpo_model.policy_model,
                prefix,
                strategy_path,
            )
            return
        load_ckpt_func = load_distributed_checkpoint if self.grpo_config.rl_config.use_parallel else ms.load_checkpoint
        logger.info(f"use_parallel is {self.grpo_config.rl_config.use_parallel} {load_ckpt_func}")
        if self.sft_ckpt_path_infer:
            self.on_device = True
            param_dict = load_ckpt_func(self.sft_ckpt_path_infer)
            if self.use_vllm == VllmMode.ORIGIN:
                new_param_dict = {"grpo_model.policy_model.model." + k: v for k, v in param_dict.items()}
            else:
                new_param_dict = {"grpo_model.policy_model." + k: v for k, v in param_dict.items()}

            logger.info(f"begin to load infer policy model from: {self.sft_ckpt_path_infer}")
            logger.info(new_param_dict.keys())
            for _, param in self.grpo_model_infer.grpo_model.policy_model.parameters_and_names():
                logger.info(f"infer model para names:   {param.name}")
            param_not_load, ckpt_not_load = ms.load_param_into_net(
                self.grpo_model_infer.grpo_model.policy_model, new_param_dict
            )
            logger.info(f"param not load: {param_not_load}")
            logger.info(f"ckpt not load: {ckpt_not_load}")

    def convert_map_dict(self, source_dict, **kwargs):
        """convert_map_dict"""
        if self.use_vllm != VllmMode.ORIGIN:
            network = self.grpo_model_infer.grpo_model.policy_model
            prefix = "grpo_model.policy_model."
        else:
            network = self.grpo_model_infer.grpo_model.policy_model.model
            prefix = "grpo_model.policy_model.model."
        weight_dict = network.convert_map_dict(source_dict, **kwargs)
        new_weight_dict = {f"{prefix}{key}": value for key, value in weight_dict.items()}
        return new_weight_dict

    @enable_pynative_async
    def _vllm_generate(self, vllm_prompt, valid_length_each_example):
        token_ids = self.inference_engine.pre_process_inputs(vllm_prompt, valid_length_each_example)
        outputs = self.inference_engine.generate(
            prompts=None, sampling_params=self.sampling_params, prompt_token_ids=token_ids, use_tqdm=False
        )
        return outputs

    @enable_pynative_async
    def _init_cache_engine(self):
        self.inference_engine.init_cache_engine()
