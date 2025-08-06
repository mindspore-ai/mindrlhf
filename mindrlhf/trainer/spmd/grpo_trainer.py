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
""" GRPO Trainer """
import os
import time

import numpy as np
from mindformers import MindFormerConfig
from mindformers import logger
from mindformers.utils.tensorboard import get_tensorboard_writer, _set_tensorboard_writer
import mindspore as ms
from mindspore import Tensor
from mindspore._c_expression import MSContext
from mindspore.common.api import _pynative_executor
from mindspore.communication import get_rank, get_group_size
from omegaconf import DictConfig, OmegaConf

from mindrlhf.configs import GRPOConfigGenerator
from mindrlhf.models import TokenizerFactory
from mindrlhf.tools.host_monitor import ResourceMonitor
from mindrlhf.trainer.spmd.grpo_experience_maker import GRPOExperienceMaker
from mindrlhf.utils import (
    set_perf_stats,
    TimeConsumingCollector,
    convert_index_json_total,
    profiler_start,
    profiler_step,
    set_infer_dp_size,
)
from mindrlhf.utils.reshard_optimizer import ReshardOptimizer, Parallel
from mindrlhf.worker.infer_worker import InferWorker
from mindrlhf.worker.old_policy_worker import get_old_policy_worker, set_enable_old_policy
from mindrlhf.worker.ref_worker import RefWorker
from mindrlhf.worker.train_worker import TrainWorker
from mindrlhf.worker.transform_worker import TransformWorker


class GRPOTrainer:
    """GRPO Trainer"""

    def __init__(self, config_from_yaml: DictConfig, **kwargs):
        """
        GRPOTrainer initialization.

        Args:
            config_from_yaml (DictConfig): RL config.
            **kwargs (Dict): Key-word args.
        """
        self.no_patch_tensor_shape = kwargs.get("no_patch_tensor_shape")
        self.grpo_config = GRPOConfigGenerator.create_config(config_from_yaml)
        self._init_testing_env()
        self._init_deterministic()
        self._set_hccl_op_expansion_mode()
        self._set_vllm_generation_config()
        set_perf_stats(self.grpo_config.rl_config.performance_stats)
        set_enable_old_policy(self.grpo_config)
        set_infer_dp_size(self.grpo_config)

        self.reshard_optimizer = self._init_reshard_optimizer()
        self.tensor_writer = self._init_tensorboard()
        self.tokenizer = TokenizerFactory.init_tokenizer(self.grpo_config)
        self.host_monitor = ResourceMonitor(
            self.grpo_config.monitor_config.host_monitor_interval,
            self.grpo_config.monitor_config.host_monitor_steps,
            self.grpo_config.monitor_config.host_memory_protection,
            self.grpo_config.monitor_config.host_max_memory_threshold,
        )
        self.host_monitor.start()
        self.infer = InferWorker(
            grpo_config=self.grpo_config, tokenizer=self.tokenizer, tensor_writer=self.tensor_writer
        )
        self.ref = RefWorker(grpo_config=self.grpo_config, tensor_writer=self.tensor_writer)
        self.train = TrainWorker(grpo_config=self.grpo_config, tensor_writer=self.tensor_writer)
        self.old_policy = get_old_policy_worker(grpo_config=self.grpo_config)
        logger.info("GRPOTrainer: finish init workers")

        # rename parameters in safetensors
        if self.grpo_config.rl_config.load_ckpt_format == "hf_safetensors":
            self.rename_safetensors_weights()

        self._compile()
        self.transform = TransformWorker(
            self.grpo_config,
            self.train.sft_model_config_train,
            self.train.model(),
            self.infer.model(),
            self.ref.model(),
            self.old_policy.model(),
        )

        self._load_checkpoint()
        if not self.grpo_config.generate_config.load:
            self.transform.reshard_params(0)

        # Init train control variables.
        self.i_step, self.n_epoch, self.start_step, self.start_epoch, self.total_time = 0, 0, 0, 0, 0
        self.world_group_size = get_group_size()
        self.experience_maker = GRPOExperienceMaker(
            self.train,
            self.infer,
            self.ref,
            self.old_policy,
            self.grpo_config,
            self.tokenizer,
            self.tensor_writer,
            self.i_step,
        )
        self.step_num = self.experience_maker.step_num

        self.infer.refresh_policy_model_phase()

    def _init_testing_env(self):
        """Initiate environment for testing(ut/st) when MINDRLHF_TEST is set."""
        if os.getenv("MINDRLHF_TEST") == "1":
            print(f"124 set MINDRLHF_TEST to {os.getenv('MINDRLHF_TEST')}")
            # Set vllm env.
            os.environ["MINDFORMERS_MODEL_CONFIG"] = self.grpo_config.generate_config.model_config
            from typing import Iterable, Set, Tuple

            def qwen2_load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> Set[str]:
                return None

            from mindrlhf.third_party.vllm import Qwen2ForCausalLM

            # skip load ckpt in ci
            Qwen2ForCausalLM.load_weights = qwen2_load_weights
        else:
            print(f"137 MINDRLHF_TEST is not set.")

    @staticmethod
    def _set_hccl_op_expansion_mode():
        """set communication algorithm environment variables to avoid allreduce timeouts"""
        if MSContext.get_instance().get_ascend_soc_version() == "ascend910b":
            os.environ["HCCL_OP_EXPANSION_MODE"] = "HOST"
        elif MSContext.get_instance().get_ascend_soc_version() == "ascend910_93":
            os.environ["HCCL_OP_EXPANSION_MODE"] = "AI_CPU"
        else:
            raise NotImplementedError("only support ascend910b and ascend910_93")

    def _init_tensorboard(self):
        """init tensorboard"""
        if self.grpo_config.rl_config.tensorboard and self.grpo_config.rl_config.tensorboard_dir:
            tensorboard_dir = os.path.join(self.grpo_config.rl_config.tensorboard_dir, f"rank_{get_rank()}")
            tensorboard_conf = OmegaConf.create(
                {
                    "tensorboard_dir": tensorboard_dir,
                    "tensorboard_queue_size": self.grpo_config.rl_config.tensorboard_queue_size,
                }
            )
            _set_tensorboard_writer(tensorboard_conf)
        return get_tensorboard_writer()

    def _init_deterministic(self):
        deterministic = self.grpo_config.rl_config.deterministic.lower() == "on"
        logger.info(f"set deterministic: {deterministic}")
        if deterministic:
            ms.set_deterministic(True)
            os.environ["HCCL_DETERMINISTIC"] = "true"
            os.environ["TE_PARALLEL_COMPILER"] = "1"
            os.environ["CUSTOM_MATMUL_SHUFFLE"] = "off"
            os.environ["LCCL_DETERMINISTIC"] = "1"
        ms.set_seed(self.grpo_config.rl_config.seed)

    def _init_reshard_optimizer(self):
        """Init reshard optimizer."""
        reshard_optimizer = None
        if self.grpo_config.rl_config.enable_reshard_optimizer:
            logger.info("GRPOTrainer: start init Reshard Optimizer")
            reshard_optimizer = ReshardOptimizer(
                src_parallel=Parallel(
                    dp=self.grpo_config.actor_config.parallel_config.data_parallel,
                    tp=self.grpo_config.actor_config.parallel_config.model_parallel,
                    pp=self.grpo_config.actor_config.parallel_config.pipeline_stage,
                ),
                dst_parallel=Parallel(
                    dp=self.grpo_config.generate_config.parallel_config.data_parallel,
                    tp=self.grpo_config.generate_config.parallel_config.model_parallel,
                    pp=self.grpo_config.generate_config.parallel_config.pipeline_stage,
                ),
            )
        return reshard_optimizer

    def _set_vllm_generation_config(self):
        os.environ["MINDFORMERS_MODEL_CONFIG"] = self.grpo_config.generate_config.model_config

    def __del__(self):
        if os.getenv("MINDFORMERS_MODEL_CONFIG"):
            del os.environ["MINDFORMERS_MODEL_CONFIG"]

    @property
    def use_parallel(self):
        return self.grpo_config.rl_config.use_parallel

    def _compile(self):
        """
        compile model
        """
        with TimeConsumingCollector("GRPOTrainer compile"):
            self.infer.generate_strategy(self.reshard_optimizer)
            origin_shape = Tensor.shape
            Tensor.shape = self.no_patch_tensor_shape
            self.ref.compile()
            self.old_policy.compile()
            self.train.compile()
            Tensor.shape = origin_shape

    def _load_checkpoint(self):
        """
        load checkpoint files
        """
        if self.grpo_config.rl_config.resume_training:
            epoch_step_info = self.train.reload_ckpt()
            if epoch_step_info is None:
                raise ValueError("epoch/step info not read")
            self.ref.reload_ckpt()
            self.transform.reshard_params(0)
            epoch_num = epoch_step_info["epoch_num"]
            data_skip_steps = epoch_step_info["step_num"]
            if epoch_num > 0:
                logger.info(f"epoch in resume training is: {epoch_num}.")
                self.n_epoch = epoch_num
                self.start_epoch = epoch_num
            if data_skip_steps > 0:
                logger.info(f"Skip step in resume training is: {data_skip_steps}.")
                self.i_step = data_skip_steps
                self.start_step = data_skip_steps
            return

        with TimeConsumingCollector("GRPOTrainer load checkpoint"):
            self.infer.load_checkpoint()
            self.ref.load_checkpoint()
            self.old_policy.load_checkpoint()
            self.train.load_checkpoint()

    def _reshard_train_to_infer(self):
        """Reshard train model parameters to infer model."""
        if self.grpo_config.rl_config.reshard_mem_opt_level == 1:
            self.train.offload_model()
            if self.train.model_on_device:
                raise RuntimeError(
                    "when reshard_mem_opt_level is equal to 1, train model must not on device before transform param"
                )
            if self.infer.on_device:
                raise RuntimeError(
                    "when reshard_mem_opt_level is equal to 1, infer model must not on device before transform param"
                )
            self.old_policy.check_not_on_device()
        else:
            self.infer.load(skip_kv_cache=True)
            self.old_policy.load()
            if not self.train.model_on_device:
                raise RuntimeError(
                    "when reshard_mem_opt_level is equal to 0, train model must on device before transform param"
                )
            if not self.infer.on_device:
                raise RuntimeError(
                    "when reshard_mem_opt_level is equal to 0, infer model must on device before transform param"
                )

        if self.transform.sync_ref_model and ((self.i_step + 1) % self.transform.ref_model_sync_steps == 0):
            # in some work, ref update may have a 'bad' effect
            if self.grpo_config.rl_config.reshard_mem_opt_level == 0:
                self.ref.load()
            input_on_device_flag_dict = {
                "policy2infer": (self.train.model_on_device, self.infer.on_device),
                "policy2ref": (self.train.model_on_device, self.ref.on_device),
                "policy2old": (self.train.model_on_device, self.old_policy.on_device),
            }
            self.transform.reshard_params(self.i_step, input_on_device_flag_dict)
            if self.grpo_config.rl_config.reshard_mem_opt_level == 0:
                self.ref.offload()
        else:
            input_on_device_flag_dict = {
                "policy2infer": (self.train.model_on_device, self.infer.on_device),
                "policy2ref": (self.train.model_on_device, self.ref.on_device),
                "policy2old": (self.train.model_on_device, self.old_policy.on_device),
            }
            self.transform.reshard_params(self.i_step, input_on_device_flag_dict)

        if self.grpo_config.rl_config.reshard_mem_opt_level == 0:
            if not self.train.model_on_device:
                raise RuntimeError(
                    "when reshard_mem_opt_level is equal to 0, train model must on device after transform param"
                )
            if not self.infer.on_device:
                raise RuntimeError(
                    "when reshard_mem_opt_level is equal to 0, infer model must on device after transform param"
                )
            self.train.offload_model()
            self.old_policy.check_on_device()
            self.old_policy.offload()
            self.infer.load_kvcache()
        else:
            if self.train.model_on_device:
                raise RuntimeError(
                    "when reshard_mem_opt_level is equal to 1, train model must not on device after transform param"
                )
            if self.infer.on_device:
                raise RuntimeError(
                    "when reshard_mem_opt_level is equal to 1, infer model must not on device after transform param"
                )
            self.old_policy.check_not_on_device()

    def run_grpo_train(self):
        """
        Main entry of MindRLHF GRPO training.
        """
        logger.info(
            f"Start training epoch num:{self.grpo_config.rl_config.epochs}, step num:{self.step_num}, "
            f"generation num:{self.grpo_config.rl_config.num_generations}"
        )
        np.set_printoptions(threshold=1024)
        while self.n_epoch < self.grpo_config.rl_config.epochs:
            grpo_profiler = profiler_start(self.grpo_config.profiler_config, role="grpo_all_stage")
            while self.i_step < self.step_num:
                if self.i_step % self.grpo_config.rl_config.save_ckpt_interval == 0:
                    self.train.save_checkpoints(
                        epochs=self.n_epoch, steps=self.i_step, start_epoch=self.start_epoch, start_step=self.start_step
                    )
                    self.ref.save_checkpoints(
                        epochs=self.n_epoch, steps=self.i_step, start_epoch=self.start_epoch, start_step=self.start_step
                    )
                self.host_monitor.update_current_step(self.i_step)
                logger.info(f"epoch: {self.n_epoch}, step: {self.i_step} start")
                with TimeConsumingCollector(f"whole epoch {self.n_epoch} train stage") as perf_collector:
                    with TimeConsumingCollector("make_experience"):
                        self.experience_maker.make_experience(
                            num_rollouts=self.grpo_config.rl_config.num_rollouts,
                            num_generations=self.grpo_config.rl_config.num_generations,
                        )
                    with TimeConsumingCollector("load train optimizer"):
                        self.train.load_optimizer()
                    with TimeConsumingCollector("load train model"):
                        self.train.load_model()
                    with TimeConsumingCollector("train model"):
                        update_profiler = profiler_start(
                            self.grpo_config.profiler_config, role="actor_update", profiler_iteration=self.n_epoch
                        )
                        self.train.train()
                        profiler_step(update_profiler)
                    with TimeConsumingCollector("offload train optimizer"):
                        self.train.offload_optimizer()
                    with TimeConsumingCollector("reshard train to infer"):
                        self._reshard_train_to_infer()
                self.total_time += perf_collector.duration
                logger.info(
                    "step processed tokens {}, tokens/s/p {}".format(
                        self.experience_maker.step_total_tokens,
                        self.experience_maker.step_total_tokens / perf_collector.duration / self.world_group_size,
                    )
                )
                logger.info(
                    "total processed tokens {}, total tokens/s/p {}".format(
                        self.experience_maker.total_processed_tokens,
                        self.experience_maker.total_processed_tokens / self.total_time / self.world_group_size,
                    )
                )
                self.i_step += 1
                profiler_step(grpo_profiler)
            self.i_step = 0
            self.n_epoch += 1

        with TimeConsumingCollector("save checkpoint"):
            with TimeConsumingCollector("load train model"):
                self.train.load_model()
            self.train.save_checkpoints(epochs=self.grpo_config.rl_config.epochs, steps=self.step_num)
        self.host_monitor.stop()
        logger.info("run grpo train end")

    def rename_safetensors_weights(self):
        """rename safetensors and write output to param_name_map.json"""
        # 默认3个模型要加载的safetensors文件相同，用同一个config对象处理
        config = MindFormerConfig(self.grpo_config.actor_config.model_config)
        config.load_checkpoint = self.grpo_config.actor_config.load

        if config.model.model_config.get("qkv_concat", False):
            raise ValueError("safetensors only support qkv_concat=False for now")

        if get_rank() == 0:
            convert_func_lst = []
            convert_func_lst.append(self.infer.convert_map_dict)
            convert_func_lst.append(self.ref.convert_map_dict)
            if self.grpo_config.rl_config.enable_oldpolicy:
                convert_func_lst.append(self.old_policy.convert_map_dict)
            convert_func_lst.append(self.train.convert_map_dict)
            convert_index_json_total(config.load_checkpoint, config.load_checkpoint, convert_func_lst, False)
        else:
            # wait for rank 0 to finish
            time.sleep(10)
        ms.mint.distributed.barrier()
        _pynative_executor.sync()
