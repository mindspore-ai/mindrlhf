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

# python
import os
import time
import numpy as np

# mindspore
import mindspore as ms
from mindspore import Tensor
from mindspore.common.api import _pynative_executor
from mindspore.communication import get_rank, get_group_size

# mindformers
from mindformers import logger
from mindformers.models.llama import LlamaTokenizerFast
from mindformers import MindFormerConfig
from mindformers.models.build_tokenizer import build_tokenizer
from mindformers.utils.tensorboard import get_tensorboard_writer, _set_tensorboard_writer

# mindrlhf
from mindrlhf.utils import transfer_from_str_to_bool, set_perf_stats, TimeConsumingCollector, convert_index_json_total

from mindrlhf.worker.infer_worker import InferWorker
from mindrlhf.worker.ref_worker import RefWorker
from mindrlhf.worker.train_worker import TrainWorker
from mindrlhf.worker.old_policy_worker import OldPolicyWorker
from mindrlhf.worker.transform_worker import TransformWorker
import mindrlhf.utils.reshard_optimizer as reshard_optimizer
from mindrlhf.configs.grpo_configs import GRPOConfig, VllmMode
from mindrlhf.models.qwen2_5.qwen2_5_tokenizer import Qwen2_5Tokenizer
from mindrlhf.trainer.spmd.grpo_experience_maker import GRPOExperienceMaker


class GRPOTrainer:
    """GRPO Trainer"""

    def __init__(self, no_patch_tensor_shape, args=None):
        """Initialize"""
        self.args = args
        self._init_grpo_configs(args)
        self._set_vllm_generation_config()
        self.no_patch_tensor_shape = no_patch_tensor_shape

        # ================== Initial Tensorboard ==================
        if self.grpo_config.rl_config.tensorboard and self.grpo_config.rl_config.tensorboard_dir:
            self.grpo_config.rl_config.tensorboard_dir = os.path.join(
                self.grpo_config.rl_config.tensorboard_dir, f"rank_{get_rank()}"
            )
            _set_tensorboard_writer(self.grpo_config.rl_config)
        self.tensor_writer = get_tensorboard_writer()
        setattr(self.args, "tensor_writer", self.tensor_writer)

        self.reshard_optimizer = None
        if self.grpo_config.rl_config.enable_reshard_optimizer:
            logger.info("GRPOTrainer: start init Reshard Optimizer")

            self.reshard_optimizer = reshard_optimizer.ReshardOptimizer(
                src_parallel=reshard_optimizer.Parallel(
                    dp=self.grpo_config.actor_config.parallel_config.data_parallel,
                    tp=self.grpo_config.actor_config.parallel_config.model_parallel,
                    pp=self.grpo_config.actor_config.parallel_config.pipeline_stage,
                ),
                dst_parallel=reshard_optimizer.Parallel(
                    dp=self.grpo_config.generate_config.parallel_config.data_parallel,
                    tp=self.grpo_config.generate_config.parallel_config.model_parallel,
                    pp=self.grpo_config.generate_config.parallel_config.pipeline_stage,
                ),
            )
            reshard_optimizer.OPT_COMMUNICATION_GROUPS = self.reshard_optimizer.opt_communication_groups

        logger.info("GRPOTrainer: start init workers")
        self.infer = InferWorker(grpo_config=self.grpo_config, sft_path_infer=self.sft_path_infer, args=self.args)

        self.ref = RefWorker(grpo_config=self.grpo_config, sft_path_ref=self.sft_path_ref, args=self.args)

        self.train = TrainWorker(grpo_config=self.grpo_config, sft_path_train=self.sft_path_train, args=self.args)
        self.old_policy = OldPolicyWorker(
            grpo_config=self.grpo_config, sft_path_train=self.sft_path_train, args=self.args
        )
        logger.info(f"config of sft_model_config_train {self.train.sft_model_config_train}")
        if self.grpo_config.rl_config.packing:
            if self.grpo_config.rl_config.pack_num < 1:
                raise ValueError("pack_num must >= 1!")
            logger.info(
                f"Set packing_sample_length to train worker seq_length: "
                f"{self.train.sft_model_config_train.seq_length}."
            )
        else:
            self.grpo_config.rl_config.packing = True
            self.grpo_config.rl_config.pack_num = 1
            logger.warning(f"Set packing False, reset packing True and pack_num = 1.")
        logger.info("GRPOTrainer: finish init workers")

        self.reshard_mem_opt_level = self.grpo_config.rl_config.reshard_mem_opt_level
        if self.reshard_mem_opt_level not in [0, 1]:
            raise ValueError(f"reshard_mem_opt_level can only be 0 or 1, but got {self.reshard_mem_opt_level}")
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
        self.i_step = 0
        self.n_epoch = 0
        self.start_step = 0
        self.start_epoch = 0
        self.total_time = 0
        self._load_checkpoint()
        if not self.grpo_config.generate_config.load:
            self.transform.reshard_params(0)

        if self.grpo_config.rl_config.save_ckpt_interval <= 0:
            raise ValueError(
                f"save_ckpt_interval should be lager than 0, but got "
                f"{self.grpo_config.rl_config.save_ckpt_interval}"
            )
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
        self.step_num = self.experience_maker.get_step_num()

        if self.infer.use_vllm == VllmMode.ORIGIN:
            self.infer.grpo_model_infer.grpo_model.policy_model.model.set_train(False)
        else:
            self.infer.grpo_model_infer.grpo_model.policy_model.set_train(False)
        self.ref.ref_model.model.set_train(False)

        self.infer.refresh_policy_model_phase()

    def _set_vllm_generation_config(self):
        os.environ["MINDFORMERS_MODEL_CONFIG"] = self.grpo_config.generate_config.model_config

    def __del__(self):
        if os.getenv("MINDFORMERS_MODEL_CONFIG"):
            del os.environ["MINDFORMERS_MODEL_CONFIG"]

    @staticmethod
    def _set_args_to_config(args, grpo_config: GRPOConfig):
        """set args to config"""
        if args.dataset_file is not None:
            grpo_config.rl_config.dataset_file = args.dataset_file
        if args.tokenizer_dir is not None:
            grpo_config.rl_config.tokenizer_dir = args.tokenizer_dir
        if args.actor_checkpoint_path is not None:
            grpo_config.actor_config.load = args.actor_checkpoint_path
        if args.ref_checkpoint_path is not None:
            grpo_config.ref_config.load = args.ref_checkpoint_path
        if args.generate_checkpoint_path is not None:
            grpo_config.generate_config.load = args.generate_checkpoint_path
        if args.verifier_function is not None:
            if "," in args.verifier_function:
                verifier_function = args.verifier_function.split(",")
            else:
                verifier_function = [args.verifier_function]
            grpo_config.reward_config.verifier_function = verifier_function
        if args.verifier_weight is not None:
            if "," in args.verifier_weight:
                verifier_weight = args.verifier_weight.split(",")
                verifier_weight = [float(_) for _ in verifier_weight]
            else:
                verifier_weight = [float(args.verifier_weight)]
            grpo_config.reward_config.verifier_weight = verifier_weight
        if args.tensorboard is not None:
            tensorboard = transfer_from_str_to_bool(args.tensorboard)
            grpo_config.rl_config.tensorboard = tensorboard
        if args.save_checkpoint_dir is not None:
            grpo_config.actor_config.save = args.save_checkpoint_dir
        return grpo_config

    def _init_grpo_configs(self, args):
        """init grpo configs"""
        logger.info(f"GRPOTrainer: _init_grpo_configs {args} in main task")
        # init grpo config
        grpo_config = GRPOConfig(args.config)
        grpo_config = self._set_args_to_config(args, grpo_config)
        set_perf_stats(grpo_config)
        if grpo_config.generate_config.use_vllm not in range(len(VllmMode)):
            logger.warning(f"use_vllm should be 0, 1 or 2, but got {grpo_config.generate_config.use_vllm}. Reset to 0.")
            grpo_config.generate_config.use_vllm = 0
        grpo_config.generate_config.use_vllm = VllmMode(grpo_config.generate_config.use_vllm)
        logger.info(
            f"vllm mode: {grpo_config.generate_config.use_vllm}, "
            f"hf_config_path: {grpo_config.generate_config.hf_config_path}"
        )
        if (
            grpo_config.rl_config.save_prompt_completions_data
            and grpo_config.rl_config.save_prompt_completions_interval <= 0
        ):
            logger.warning(
                f"save_prompt_completions_interval should be positive, "
                f"but got {grpo_config.rl_config.save_prompt_completions_interval}. "
                f"Set save_prompt_completions_data to False."
            )
            grpo_config.rl_config.save_prompt_completions_data = False

        # for worker
        if args.custom_model_name == "qwen":
            args.vocab_path = os.path.join(grpo_config.rl_config.tokenizer_dir, "vocab.json")
            args.merges_file_path = os.path.join(grpo_config.rl_config.tokenizer_dir, "merges.txt")
            self.tokenizer = Qwen2_5Tokenizer(
                args.vocab_path, args.merges_file_path, add_bos_token=False, add_eos_token=False
            )
        elif args.custom_model_name == "deepseek":
            args.tokenizer_path = grpo_config.rl_config.tokenizer_dir
            self.tokenizer = LlamaTokenizerFast(
                tokenizer_file=args.tokenizer_path, add_bos_token=False, add_eos_token=False
            )
        elif args.custom_model_name == "llama":
            args.vocab_path = grpo_config.rl_config.tokenizer_dir
            sft_config_infer = MindFormerConfig(grpo_config.generate_config.model_config)
            sft_config_infer.processor.tokenizer.tokenizer_file = args.vocab_path
            sft_config_infer.processor.tokenizer.vocab_file = args.vocab_path
            self.tokenizer = build_tokenizer(sft_config_infer.processor.tokenizer)
        else:
            raise ValueError(f"model_name should in ['qwen', 'deepseek'], but get {args.custom_model_name}")
        self.grpo_config = grpo_config
        self.use_parallel = transfer_from_str_to_bool(self.grpo_config.rl_config.use_parallel)
        self.sft_path_infer = grpo_config.generate_config.model_config
        self.sft_path_train = grpo_config.actor_config.model_config
        self.sft_path_ref = grpo_config.ref_config.model_config
        if isinstance(self.grpo_config.rl_config.seed, int):
            ms.set_seed(self.grpo_config.rl_config.seed)

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
        if self.args.resume_training:
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
        if self.reshard_mem_opt_level == 1:
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
            self.infer.load()
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
            if self.reshard_mem_opt_level == 0:
                self.ref.load()
            input_on_device_flag_dict = {
                "policy2infer": (self.train.model_on_device, self.infer.on_device),
                "policy2ref": (self.train.model_on_device, self.ref.on_device),
                "policy2old": (self.train.model_on_device, self.old_policy.on_device),
            }
            self.transform.reshard_params(self.i_step, input_on_device_flag_dict)
            if self.reshard_mem_opt_level == 0:
                self.ref.offload()
        else:
            input_on_device_flag_dict = {
                "policy2infer": (self.train.model_on_device, self.infer.on_device),
                "policy2ref": (self.train.model_on_device, self.ref.on_device),
                "policy2old": (self.train.model_on_device, self.old_policy.on_device),
            }
            self.transform.reshard_params(self.i_step, input_on_device_flag_dict)

        if self.reshard_mem_opt_level == 0:
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
            while self.i_step < self.step_num:
                if self.i_step % self.grpo_config.rl_config.save_ckpt_interval == 0:
                    self.train.save_checkpoints(
                        epochs=self.n_epoch, steps=self.i_step, start_epoch=self.start_epoch, start_step=self.start_step
                    )
                    self.ref.save_checkpoints(
                        epochs=self.n_epoch, steps=self.i_step, start_epoch=self.start_epoch, start_step=self.start_step
                    )

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
                        self.train.train()
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
            self.i_step = 0
            self.n_epoch += 1

        with TimeConsumingCollector("save checkpoint"):
            with TimeConsumingCollector("load train model"):
                self.train.load_model()
            self.train.save_checkpoints(epochs=self.grpo_config.rl_config.epochs, steps=self.step_num)
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
            if self.grpo_config.rl_config.num_iterations > 1:
                convert_func_lst.append(self.old_policy.convert_map_dict)
            convert_func_lst.append(self.train.convert_map_dict)
            convert_index_json_total(config.load_checkpoint, config.load_checkpoint, convert_func_lst, False)
        else:
            # wait for rank 0 to finish
            time.sleep(10)
        ms.mint.distributed.barrier()
        _pynative_executor.sync()
