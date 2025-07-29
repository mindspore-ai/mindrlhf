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
"""Train Worker"""
import os

import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.dataset.transforms import TypeCast
from mindspore.dataset import GeneratorDataset
from mindspore.communication import get_rank
from mindspore import context
from mindspore.nn.wrap.cell_wrapper import PipelineCell, _VirtualDatasetCell, MicroBatchInterleaved
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell

from mindformers import MindFormerConfig
from mindformers.trainer.utils import load_distributed_checkpoint
from mindformers.core.callback.callback import TopkBiasBalanceCallback
from mindformers import LlamaConfig
from mindformers import logger
from mindformers.tools.resume_ckpt import get_resume_checkpoint_by_meta

from research.deepseek3.deepseek3_config import DeepseekV3Config

from mindrlhf.utils.adam import AdamWeightDecayOp
from mindrlhf.utils.utils import LearningRate, FP32StateAdamWeightDecay, TimeConsumingCollector
from mindrlhf.wrapper import TrainOneStepWithLossScaleGRPO, TrainPipelineWithLossScaleCellGRPO
from mindrlhf.utils.utils import (
    load_param_to_net,
    record_last_ckpt_to_json,
    get_checkpoint_name,
    ensure_total_ckpt_is_less_than_limit,
    load_safetensors,
)
from mindrlhf.models.grpo_models import CausalLMHybrid, GRPOModelTrain
from mindrlhf.utils.dataset import GRPOIteratorStore
from mindrlhf.worker.worker import Worker
from mindrlhf.utils.configs import set_weight_decay
from mindrlhf.configs.grpo_configs import GRPOConfig


class TrainWorker(Worker):
    """
    This class do GRPO train.
    """

    def __init__(self, grpo_config: GRPOConfig, args):
        super().__init__()
        logger.info("init TrainWorker")
        self.args = args
        self.grpo_config = grpo_config
        self.load_ckpt_format = self.grpo_config.rl_config.load_ckpt_format
        sft_config_train = MindFormerConfig(grpo_config.actor_config.model_config)
        sft_config_train.use_parallel = grpo_config.rl_config.use_parallel
        self.sft_config_train = sft_config_train
        sft_config_train.parallel_config = MindFormerConfig(**grpo_config.actor_config.parallel_config.param_dict)
        logger.info(f"actor parallel_config:{sft_config_train.parallel_config}")
        logger.info(f"grpo_config.actor_config.recompute_config:{grpo_config.actor_config.recompute_config.param_dict}")
        sft_config_train.recompute_config = grpo_config.actor_config.recompute_config.param_dict
        sft_config_train.model.model_config.seq_length = grpo_config.rl_config.seq_length
        sft_config_train.model.model_config.offset = grpo_config.actor_config.offset
        sft_config_train.model.model_config.parallel_config = sft_config_train.parallel_config

        if grpo_config.actor_config.save and get_rank() == 0:
            train_save_dir = os.path.join(grpo_config.actor_config.save, "train")
            optimizer_save_dir = os.path.join(grpo_config.actor_config.save, "optimizer")
            if not os.path.exists(train_save_dir):
                os.makedirs(train_save_dir)
            if not os.path.exists(optimizer_save_dir):
                os.makedirs(optimizer_save_dir)

        os.environ["RUN_MODE"] = sft_config_train.run_mode
        sft_config_train.model.model_config.parallel_config.recompute = sft_config_train.recompute_config
        if args.model_name in ["qwen", "llama"]:
            sft_config_train.model.model_config.use_eod_attn_mask_compression = (
                grpo_config.actor_config.use_eod_attn_mask_compression
            )
            sft_model_config_train = LlamaConfig(**sft_config_train.model.model_config)
            sft_model_config_train.model_name = "llama"
        elif args.model_name == "deepseek":
            sft_config_train.model.model_config.moe_config = sft_config_train.moe_config
            sft_model_config_train = DeepseekV3Config(**sft_config_train.model.model_config)
            sft_model_config_train.model_name = "deepseek_training"
            self.topk_bias_balance_callback = TopkBiasBalanceCallback(
                sft_model_config_train.moe_config.balance_via_topk_bias,
                sft_model_config_train.moe_config.topk_bias_update_rate,
                sft_model_config_train.moe_config.expert_num,
                sft_model_config_train.parallel_config.micro_batch_num
            )
        else:
            raise ValueError(f"model_name should in ['qwen', 'llama','deepseek'], but get {args.model_name}")
        sft_model_config_train.checkpoint_name_or_path = grpo_config.actor_config.load
        self.sft_ckpt_path_train = sft_model_config_train.checkpoint_name_or_path
        sft_model_config_train.checkpoint_name_or_path = None

        self.train_pp_stage = sft_model_config_train.parallel_config.pipeline_stage or 1
        self.enable_parallel_optimizer = (
            grpo_config.actor_config.enable_parallel_optimizer
            and grpo_config.actor_config.parallel_config.data_parallel > 1
        )
        context.set_auto_parallel_context(
            pipeline_stages=self.train_pp_stage, enable_parallel_optimizer=self.enable_parallel_optimizer
        )
        self.sft_model_config_train = sft_model_config_train
        policy_model = CausalLMHybrid(sft_model_config_train, self.grpo_config)
        self.grpo_model_train = GRPOModelTrain(grpo_config, policy_model)
        self.grpo_model_train.set_train(True)
        self.grpo_with_grad = self._init_grpo_network_and_optimizer()
        self.store = []

        self.model_on_device = True
        self.optimizer_on_device = True
        self.save_strategy_dir = grpo_config.rl_config.save_strategy_dir

        self.tensor_writer = self.args.tensor_writer
        self.global_training_step = 0
        if self.grpo_config.rl_config.save_max_ckpt_num < 1:
            raise ValueError(
                f"save_max_ckpt_num should be lager than 0, " f"but got {self.grpo_config.rl_config.save_max_ckpt_num}"
            )

    def model(self):
        """Return train model."""
        return self.grpo_model_train

    def compile(self):
        """compile and save strategy"""
        self.grpo_with_grad.set_train(True)
        self.grpo_model_train.grpo_model_train.policy_model.model.set_train(True)
        context.set_auto_parallel_context(
            pipeline_stages=self.train_pp_stage, enable_parallel_optimizer=self.enable_parallel_optimizer
        )
        if self.train_pp_stage == 1:
            # for pipeline stage 1, the micro_batch_num is not used
            train_bs = self.grpo_config.rl_config.batch_size * self.sft_model_config_train.parallel_config.data_parallel
        else:
            train_bs = (
                self.grpo_config.rl_config.batch_size
                * self.sft_model_config_train.parallel_config.micro_batch_num
                * self.sft_model_config_train.parallel_config.data_parallel
            )
        prompt_completion_ids = ms.Tensor(
            shape=(train_bs, self.grpo_config.rl_config.seq_length + 1), dtype=ms.int32
        )  # [bs, seq_len+1]
        responses_mask = ms.Tensor(
            shape=(train_bs, self.grpo_config.rl_config.seq_length), dtype=ms.int32
        )  # [bs, seq_len]
        ref_per_token_logps = ms.Tensor(
            shape=(train_bs, self.grpo_config.rl_config.seq_length), dtype=ms.float32
        )  # [bs, seq_len]
        advantages = ms.Tensor(
            shape=(train_bs, self.grpo_config.rl_config.seq_length), dtype=ms.float32
        )  # [bs, seq_len]
        actual_sequence_length = ms.Tensor(
            shape=(train_bs, self.grpo_config.rl_config.pack_num), dtype=ms.int32
        )  # [bs, packed_sample_num]
        sample_index = ms.Tensor(
            shape=(train_bs, self.grpo_config.rl_config.seq_length), dtype=ms.int32
        )  # [bs, seq_len]
        sample_valid_length = ms.Tensor(
            shape=(train_bs, self.grpo_config.rl_config.pack_num), dtype=ms.int32
        )  # [bs, packed_sample_num]
        old_per_token_logps = ms.Tensor(
            shape=(train_bs, self.grpo_config.rl_config.seq_length), dtype=ms.float32
        )  # [bs, seq_len]

        with TimeConsumingCollector("train model compile"):
            stage_name = "train"
            strategy_path = self.save_strategy_dir
            context.set_auto_parallel_context(
                strategy_ckpt_config={
                    "save_file": f"{strategy_path}/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt",
                    "only_trainable_params": True
                },
                pipeline_stages=self.train_pp_stage,
            )
            # To avoid mindspore compiler's unpacking bug and prevent duplicate compilation,
            # use positional arguments instead of keyword arguments
            inputs = [
                prompt_completion_ids,
                responses_mask,
                ref_per_token_logps,
                advantages,
                actual_sequence_length,
                sample_index,
                sample_valid_length,
                old_per_token_logps,
            ]
            self.grpo_with_grad.compile(*inputs)
            stage_name = "other"
            context.set_auto_parallel_context(
                strategy_ckpt_config={
                    "save_file": f"{self.save_strategy_dir}/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt",
                    "only_trainable_params": True
                }
            )

        sim_level = os.getenv("MS_SIMULATION_LEVEL")
        if sim_level:
            logger.info(f"start dryrun with sim_level: {sim_level}")
            self.grpo_with_grad(
                prompt_completion_ids=prompt_completion_ids,
                responses_mask=responses_mask,
                ref_per_token_logps=ref_per_token_logps,
                advantages=advantages,
                actual_sequence_length=actual_sequence_length,
                sample_index=sample_index,
                sample_valid_length=sample_valid_length,
                old_per_token_logps=old_per_token_logps,
            )
            logger.info(f"dryrun finished")
            exit(0)

    def reload_ckpt(self, formats="ckpt"):
        """reload checkpoint"""
        resume_dict = None
        if self.sft_ckpt_path_train:
            if self.load_ckpt_format in ["hf_safetensors", "ms_safetensors"]:
                formats = "safetensors"
            src_ckpt_file = os.path.join(self.sft_ckpt_path_train, f"rank_{get_rank()}")
            if not os.path.isdir(src_ckpt_file):
                raise ValueError(f"There is no *.{formats} in {self.sft_ckpt_path_train}, load failed.")
            logger.info(f"Loading latest checkpoint: {src_ckpt_file}, this may take a while.")
            meta_path = os.path.join(src_ckpt_file, "meta.json")
            if not os.path.exists(meta_path):
                raise ValueError(f"Could not find meta.json in directory {src_ckpt_file} {meta_path}")
            resume_ckpt = get_resume_checkpoint_by_meta(self.sft_ckpt_path_train, formats)
            ckpt_path = os.path.join(src_ckpt_file, resume_ckpt)

            param_dict = ms.load_checkpoint(ckpt_path, format=formats)
            resume_dict = {
                "epoch_num": int(param_dict.pop("epoch_num", 0)),
                "step_num": int(param_dict.pop("step_num", 0)),
            }
            load_param_to_net(self.grpo_model_train.grpo_model_train.policy_model, param_dict)

            parent_dir = os.path.dirname(os.path.normpath(self.sft_ckpt_path_train))
            optimizer_dir = os.path.join(parent_dir, "optimizer")
            src_opt_ckpt_file = os.path.join(optimizer_dir, f"rank_{get_rank()}")
            if not os.path.isdir(src_opt_ckpt_file):
                src_opt_ckpt_file = None
            if src_opt_ckpt_file is not None:
                logger.info(f"start load ckpt optimizer: {src_opt_ckpt_file}")
                meta_path = os.path.join(src_opt_ckpt_file, "meta.json")
                if not os.path.exists(meta_path):
                    raise ValueError(f"Could not find meta.json in directory {src_ckpt_file} {meta_path}")
                resume_ckpt = get_resume_checkpoint_by_meta(optimizer_dir, formats)
                ckpt_path = os.path.join(src_opt_ckpt_file, resume_ckpt)
                param_dict_opt = ms.load_checkpoint(ckpt_path, format=formats)
                load_param_to_net(self.grpo_with_grad.optimizer, param_dict_opt)
            self.model_on_device = True
            self.optimizer_on_device = True
        return resume_dict

    def load_checkpoint(self):
        """load_checkpoint"""
        logger.info(f"sft_ckpt_path_train:{self.sft_ckpt_path_train}")
        if not self.sft_ckpt_path_train:
            return
        if not os.path.exists(self.sft_ckpt_path_train):
            raise ValueError(f"train model checkpoint path: {self.sft_ckpt_path_train} not exists")

        if self.sft_ckpt_path_train and self.load_ckpt_format in ["ms_safetensors", "hf_safetensors"]:
            self.model_on_device = True
            self.optimizer_on_device = True
            strategy_path = os.path.join(self.save_strategy_dir, "merge_strategy", "train_policy_merged_strategy.ckpt")
            network = self.grpo_model_train.grpo_model_train.policy_model.model
            prefix = "grpo_model_train.policy_model.model."
            load_safetensors(
                self.sft_ckpt_path_train,
                self.load_ckpt_format,
                network,
                self.grpo_model_train.grpo_model_train.policy_model,
                prefix,
                strategy_path,
            )
            return
        load_ckpt_func = load_distributed_checkpoint if self.grpo_config.rl_config.use_parallel else ms.load_checkpoint
        logger.info(f"use_parallel is {self.grpo_config.rl_config.use_parallel}, {load_ckpt_func}")
        if self.sft_ckpt_path_train:
            self.model_on_device = True
            self.optimizer_on_device = True
            param_dict = load_ckpt_func(self.sft_ckpt_path_train)
            new_param_dict = {"grpo_model_train.policy_model.model." + k: v for k, v in param_dict.items()}
            logger.info(f"begin to load train policy model from: {self.sft_ckpt_path_train}")
            for _, param in self.grpo_model_train.grpo_model_train.policy_model.parameters_and_names():
                logger.info(f"train model para names:   {param.name}")
            param_not_load, ckpt_not_load = ms.load_param_into_net(
                self.grpo_model_train.grpo_model_train.policy_model, new_param_dict
            )
            logger.info(f"param not load: {param_not_load}")
            logger.info(f"ckpt not load: {ckpt_not_load}")

    def _init_grpo_network_and_optimizer(self):
        """
        Build train network.
        """

        sft_model_config = self.sft_model_config_train
        grpo_model_train = self.grpo_model_train
        grpo_config = self.grpo_config
        if sft_model_config.parallel_config.pipeline_stage > 1:
            logger.info("pipeline cell")
            grpo_with_loss_net = PipelineCell(
                MicroBatchInterleaved(grpo_model_train, grpo_config.rl_config.micro_batch_interleaved),
                sft_model_config.parallel_config.micro_batch_num,
            )
        else:
            logger.info("non-pipeline cell")
            grpo_with_loss_net = grpo_model_train
        grpo_with_loss = _VirtualDatasetCell(grpo_with_loss_net)
        lr = LearningRate(
            learning_rate=grpo_config.actor_config.lr_schedule.lr,
            end_learning_rate=grpo_config.actor_config.lr_schedule.min_lr,
            warmup_steps=grpo_config.actor_config.lr_schedule.warmup_step,
            decay_steps=grpo_config.actor_config.lr_schedule.decay_steps,
            use_cosine=grpo_config.actor_config.lr_schedule.lr_decay_style == "cosine",
        )
        params = grpo_with_loss.trainable_params()
        if self.args.model_name == "deepseek":
            group_params = set_weight_decay(params, is_use_other_params=False)
        else:
            group_params = set_weight_decay(params)

        if grpo_config.actor_config.optimizer.type == "lamb":
            optimizer = nn.Lamb(group_params, learning_rate=lr)
        elif grpo_config.actor_config.optimizer.opt_offload:
            optimizer = AdamWeightDecayOp(
                group_params,
                learning_rate=lr,
                eps=grpo_config.actor_config.optimizer.eps,
                beta1=grpo_config.actor_config.optimizer.adam_beta1,
                beta2=grpo_config.actor_config.optimizer.adam_beta2,
                param_init_type=sft_model_config.param_init_type,
            )
        else:
            optimizer = FP32StateAdamWeightDecay(
                group_params,
                learning_rate=lr,
                beta1=grpo_config.actor_config.optimizer.adam_beta1,
                beta2=grpo_config.actor_config.optimizer.adam_beta2,
                eps=grpo_config.actor_config.optimizer.eps,
            )

        loss_scale_value = grpo_config.actor_config.loss_scale_value
        update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value, scale_factor=2, scale_window=1000)

        if sft_model_config.parallel_config.pipeline_stage > 1:
            logger.info("pipeline cell")
            grpo_with_grad = TrainPipelineWithLossScaleCellGRPO(
                grpo_with_loss,
                optimizer=optimizer,
                scale_sense=update_cell,
                micro_batch_num=sft_model_config.parallel_config.micro_batch_num
            )
        else:
            logger.info("non-pipeline cell")
            grpo_with_grad = TrainOneStepWithLossScaleGRPO(
                grpo_with_loss,
                optimizer=optimizer,
                scale_sense=update_cell,
                use_clip_grad=True
            )
        return grpo_with_grad

    def _init_grpo_dataset_before_train(self):
        """
        Build dataset for graph pre-compilation.
        """
        grpo_config = self.grpo_config
        column_names = [
            "prompt_completion_ids",
            "responses_mask",
            "ref_per_token_logps",
            "advantages",
            "actual_sequence_length",
            "sample_index",
            "sample_valid_length",
            "old_per_token_logps",
        ]
        logger.info(f"store.length: {len(self.store)}")
        pipeline = GRPOIteratorStore(self.store)
        dataset = GeneratorDataset(pipeline, column_names=column_names)
        logger.info(f"dataset.len: {len(dataset)}")
        type_cast_op_int32 = TypeCast(ms.int32)
        type_cast_op_fp32 = TypeCast(ms.float32)
        dataset = dataset.map(operations=type_cast_op_int32, input_columns="prompt_completion_ids")
        dataset = dataset.map(operations=type_cast_op_int32, input_columns="responses_mask")
        dataset = dataset.map(operations=type_cast_op_fp32, input_columns="ref_per_token_logps")
        dataset = dataset.map(operations=type_cast_op_fp32, input_columns="advantages")
        dataset = dataset.map(operations=type_cast_op_int32, input_columns="actual_sequence_length")
        dataset = dataset.map(operations=type_cast_op_int32, input_columns="sample_index")
        dataset = dataset.map(operations=type_cast_op_int32, input_columns="sample_valid_length")
        dataset = dataset.map(operations=type_cast_op_fp32, input_columns="old_per_token_logps")
        micro_batch_num = 1
        if self.sft_model_config_train.parallel_config.pipeline_stage > 1:
            micro_batch_num = self.sft_model_config_train.parallel_config.micro_batch_num
        batch_size = (
            grpo_config.rl_config.batch_size
            * self.sft_model_config_train.parallel_config.data_parallel
            * micro_batch_num
        )
        logger.info(
            f"bs:{grpo_config.rl_config.batch_size}, "
            f"dp:{self.sft_model_config_train.parallel_config.data_parallel}, "
            f"micro_batch_num:{micro_batch_num}, "
            f"bs in dataset: {batch_size}"
        )
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
        return dataset

    def train(self):
        """train"""
        dataset = self._init_grpo_dataset_before_train()
        context.set_auto_parallel_context(
            pipeline_stages=self.train_pp_stage, enable_parallel_optimizer=self.enable_parallel_optimizer
        )

        def formatter(out):
            return out.asnumpy() if isinstance(out, Tensor) else out

        iterator = dataset.create_dict_iterator()
        logger.info(f"dataset size is {len(dataset)}")
        with TimeConsumingCollector(
            f"train model {dataset.dataset_size} epochs and {self.grpo_config.rl_config.num_iterations} steps"
        ):
            for epoch in range(self.grpo_config.rl_config.num_iterations):
                for step, databatch in enumerate(iterator):
                    with TimeConsumingCollector(f"train epoch {epoch} step {step}"):
                        prompt_completion_ids = databatch['prompt_completion_ids']
                        responses_mask = databatch['responses_mask']
                        ref_per_token_logps = databatch['ref_per_token_logps']
                        advantages = databatch['advantages']
                        actual_sequence_length = databatch['actual_sequence_length']
                        sample_index = databatch['sample_index']
                        sample_valid_length = databatch['sample_valid_length']
                        old_per_token_logps = databatch['old_per_token_logps']
                        inputs = [
                            prompt_completion_ids,
                            responses_mask,
                            ref_per_token_logps,
                            advantages,
                            actual_sequence_length,
                            sample_index,
                            sample_valid_length,
                            old_per_token_logps,
                        ]
                        out = self.grpo_with_grad(*inputs)
                    logger.info(
                        " loss: {} | lr: {} | is overflow: {} | loss scale: {} | grad norm: {}".format(
                            formatter(out[0]),
                            formatter(out[3]),
                            formatter(out[1]),
                            formatter(out[2]),
                            formatter(out[4])
                        )
                    )
                    if self.args.model_name == "deepseek":
                        if self.topk_bias_balance_callback.update_topk_bias_flag:
                            policy_model = self.grpo_model_train.grpo_model_train.policy_model.model
                            # pylint: disable=W0212
                            self.topk_bias_balance_callback._update_topk_bias(policy_model)
                    if self.tensor_writer:
                        self.tensor_writer.add_scalar("loss", out[0].asnumpy(), global_step=self.global_training_step)
                        self.tensor_writer.add_scalar("lr", out[3].asnumpy(), global_step=self.global_training_step)
                        self.tensor_writer.add_scalar(
                            "overflow", out[1].asnumpy(), global_step=self.global_training_step
                        )
                        self.tensor_writer.add_scalar(
                            "loss-scale", out[2].asnumpy(), global_step=self.global_training_step
                        )
                        self.tensor_writer.add_scalar(
                            "grad-norm", out[4].asnumpy(), global_step=self.global_training_step
                        )
                    self.global_training_step += 1

    def offload_optimizer(self):
        """offload optimizer"""
        if self.optimizer_on_device is False:
            return
        logger.info(f"before offload train {ms.hal.memory_stats()}")
        with TimeConsumingCollector("offload train optimizer"):
            for param in self.grpo_with_grad.optimizer.moments1:
                # pylint: disable=W0212
                param._offload()
            for param in self.grpo_with_grad.optimizer.moments2:
                # pylint: disable=W0212
                param._offload()
            if self.train_pp_stage > 1:
                for param in self.grpo_with_grad.accu_grads:
                    # pylint: disable=W0212
                    param._offload()
        logger.info(f"after offload train {ms.hal.memory_stats()}")
        self.optimizer_on_device = False

    def load_optimizer(self):
        """load optimizer"""
        if self.optimizer_on_device:
            return
        logger.info(f"before load train optimizer {ms.hal.memory_stats()}")
        with TimeConsumingCollector("load train optimizer"):
            for param in self.grpo_with_grad.optimizer.moments1:
                # pylint: disable=W0212
                param._load()
            for param in self.grpo_with_grad.optimizer.moments2:
                # pylint: disable=W0212
                param._load()
            if self.train_pp_stage > 1:
                for param in self.grpo_with_grad.accu_grads:
                    # pylint: disable=W0212
                    param._load()
        logger.info(f"after load train optimizer {ms.hal.memory_stats()}")
        self.optimizer_on_device = True

    def load_model(self):
        """load model"""
        if self.model_on_device:
            return
        logger.info(f"before load train model {ms.hal.memory_stats()}")
        with TimeConsumingCollector("load train model"):
            for param in self.grpo_with_grad.network.get_parameters(expand=True):
                # pylint: disable=W0212
                param._load()
        logger.info(f"after load train model {ms.hal.memory_stats()}")
        self.model_on_device = True

    def offload_model(self):
        """offload model"""
        if self.model_on_device is False:
            return
        logger.info(f"after offload train model {ms.hal.memory_stats()}")
        with TimeConsumingCollector("offload train model"):
            for param in self.grpo_with_grad.network.get_parameters(expand=True):
                # pylint: disable=W0212
                param._offload()
        logger.info(f"after offload train model {ms.hal.memory_stats()}")
        self.model_on_device = False

    def push_to_store(self, data):
        """Save date to global store."""
        self.store = data

    def save_checkpoints(self, epochs=0, steps=0, start_epoch=0, start_step=0, formats="ckpt"):
        """save checkpoint"""
        if epochs == start_epoch and steps == start_step:
            return
        if self.grpo_config.actor_config.save:
            if self.grpo_config.rl_config.save_ckpt_format == "safetensors":
                formats = "safetensors"
            logger.info("Save checkpoints in {}".format(self.grpo_config.actor_config.save))
            train_save_dir = os.path.join(self.grpo_config.actor_config.save, "train")
            rank_path = os.path.join(train_save_dir, f"rank_{get_rank()}")
            ckpt_file = get_checkpoint_name(
                train_save_dir, prefix="policy_model", epoch_num=epochs, step_num=steps, formats=formats
            )
            self.load_model()
            append_dict = {"epoch_num": epochs, "step_num": steps}
            # ensure ckpt number is less than `keep_checkpoint_max` after saving
            self.grpo_model_train.grpo_model_train.policy_model.init_parameters_data()
            ms.save_checkpoint(
                self.grpo_model_train.grpo_model_train.policy_model,
                ckpt_file,
                append_dict=append_dict,
                integrated_save=False,
                format=formats,
            )
            self.offload_model()
            ensure_total_ckpt_is_less_than_limit(
                ckpt_path=rank_path, limit=self.grpo_config.rl_config.save_max_ckpt_num, formats=formats
            )

            optimizer_save_dir = os.path.join(self.grpo_config.actor_config.save, "optimizer")
            rank_path_opt = os.path.join(optimizer_save_dir, f"rank_{get_rank()}")
            ckpt_file_opt = get_checkpoint_name(
                optimizer_save_dir, prefix="optimizer", epoch_num=epochs, step_num=steps, formats=formats
            )
            self.load_optimizer()
            self.grpo_with_grad.optimizer.init_parameters_data()
            ms.save_checkpoint(self.grpo_with_grad.optimizer, ckpt_file_opt, integrated_save=False, format=formats)
            self.offload_optimizer()
            ensure_total_ckpt_is_less_than_limit(
                ckpt_path=rank_path_opt, limit=self.grpo_config.rl_config.save_max_ckpt_num, formats=formats
            )
            record_last_ckpt_to_json(
                epoch=epochs,
                step=steps,
                ckpt_file=os.path.basename(ckpt_file),
                meta_json=os.path.join(rank_path, "meta.json"),
            )
            record_last_ckpt_to_json(
                epoch=epochs,
                step=steps,
                ckpt_file=os.path.basename(ckpt_file_opt),
                meta_json=os.path.join(rank_path_opt, "meta.json"),
            )
        else:
            logger.info("There is no checkpoint to save!")

    def convert_map_dict(self, source_dict, **kwargs):
        """convert_map_dict"""
        network = self.grpo_model_train.grpo_model_train.policy_model.model
        prefix = "grpo_model_train.policy_model.model."
        weight_dict = network.convert_map_dict(source_dict, **kwargs)
        new_weight_dict = {f"{prefix}{key}": value for key, value in weight_dict.items()}
        return new_weight_dict
