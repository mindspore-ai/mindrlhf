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
import math

# mindspore
import mindspore
import mindspore as ms
from mindspore.dataset.transforms import TypeCast
from mindspore.dataset import GeneratorDataset
from mindspore.communication import get_rank
from mindspore import context
from mindspore.nn.wrap.cell_wrapper import PipelineCell, _VirtualDatasetCell, MicroBatchInterleaved
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore import nn

# mindformers
from mindformers import MindFormerConfig
from mindformers.trainer.utils import load_distributed_checkpoint
from mindformers import LlamaConfig
from mindformers import logger

# mindrlhf
from mindrlhf.utils.adam import AdamWeightDecayOp
from mindrlhf.utils.utils import LearningRate, FP32StateAdamWeightDecay
from mindrlhf.wrapper import TrainOneStepWithLossScale_GRPO, TrainPipelineWithLossScaleCell_GRPO
from mindrlhf.models.grpo_models import CausalLMHybrid, GRPOModelTrain
from mindrlhf.utils.dataset import GRPOIteratorStore
from mindrlhf.worker.worker import Worker, format_time_delta


class TrainWorker(Worker):
    '''
    This class do grpo train.
    '''

    def __init__(self, grpo_config, sft_path_train, args):
        super().__init__()
        logger.info("init StfTrainWorker")
        self.args = args
        self.grpo_config = grpo_config
        sft_config_train = MindFormerConfig(sft_path_train)
        sft_config_train.use_parallel = args.use_parallel
        self.sft_config_train = sft_config_train
        sft_config_train.model.model_config.parallel_config = (
            sft_config_train.parallel_config
        )
        sft_config_train.model.model_config.parallel_config.recompute = sft_config_train.recompute_config
        sft_model_config_train = LlamaConfig(**sft_config_train.model.model_config)
        sft_model_config_train.checkpoint_name_or_path = args.load_sft_checkpoint_train
        sft_model_config_train.model_name = "llama"
        self.sft_ckpt_path_train = sft_model_config_train.checkpoint_name_or_path
        sft_model_config_train.checkpoint_name_or_path = None

        self.train_pp_stage = sft_model_config_train.parallel_config.pipeline_stage or 1
        context.set_auto_parallel_context(pipeline_stages=self.train_pp_stage)
        self.sft_model_config_train = sft_model_config_train
        policy_model = CausalLMHybrid(sft_model_config_train, self.grpo_config)
        self.grpo_model_train = GRPOModelTrain(grpo_config, policy_model)
        self.grpo_model_train.set_train(True)
        self.grpo_with_grad = self._init_grpo_network_and_optimizer()
        self.store = []

        self.model_on_device = True
        self.optimizer_on_device = True

    def model(self):
        return self.grpo_model_train

    def compile(self):
        # compile and save strategy
        self.grpo_with_grad.set_train(True)
        self.grpo_model_train.grpo_model_train.policy_model.model.set_train(True)

        sample = self.store[0]
        self.store = [sample for _ in range(self.args.pre_store_data)]
        dataset = self._init_grpo_dataset_before_train()
        data = next(dataset.create_dict_iterator())

        start_time = time.time()
        stage_name = 'train'
        context.set_auto_parallel_context(
            strategy_ckpt_config={
                "save_file":
                    f"../../strategy/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt"},
            pipeline_stages=self.train_pp_stage
        )
        self.grpo_with_grad.compile(**data)
        stage_name = 'other'
        context.set_auto_parallel_context(
            strategy_ckpt_config={
                "save_file":
                    f"../../strategy/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt"})
        logger.info(f"grpo_with_grad time: {format_time_delta(time.time() - start_time)}")

    def load_checkpoint(self):
        load_ckpt_func = load_distributed_checkpoint if self.grpo_config.use_parallel else ms.load_checkpoint
        logger.info(f"self.grpo_config.use_parallel is {self.grpo_config.use_parallel}, {load_ckpt_func}")
        if self.sft_ckpt_path_train:
            param_dict = load_ckpt_func(self.sft_ckpt_path_train)
            new_param_dict = {'grpo_model_train.policy_model.model.' + k: v for k, v in param_dict.items()}
            logger.info(f"begin to load train policy model from: {self.sft_ckpt_path_train}")
            for _, param in self.grpo_model_train.grpo_model_train.policy_model.parameters_and_names():
                logger.info(f"train model para names:   {param.name}")
            param_not_load, ckpt_not_load = mindspore.load_param_into_net(
                self.grpo_model_train.grpo_model_train.policy_model, new_param_dict)
            logger.info(f"param not load: {param_not_load}")
            logger.info(f"ckpt not load: {ckpt_not_load}")

    def _init_grpo_network_and_optimizer(self):
        """
        Build train network.
        """
        def set_weight_decay(params):
            def decay_filter(x):
                return 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()

            decay_params = list(filter(decay_filter, params))
            other_params = list(filter(lambda x: not decay_filter(x), params))
            group_params = [{
                'params': decay_params,
                'weight_decay': 1e-1
            }, {
                'params': other_params,
                'weight_decay': 0.0
            }, {
                'order_params': params
            }]
            return group_params

        sft_model_config = self.sft_model_config_train
        grpo_model_train = self.grpo_model_train
        grpo_config = self.grpo_config
        if sft_model_config.parallel_config.pipeline_stage > 1:
            logger.info("pipeline cell")
            grpo_with_loss_net = PipelineCell(MicroBatchInterleaved(grpo_model_train,
                                                                    grpo_config.micro_batch_interleaved),
                                              sft_model_config.parallel_config.micro_batch_num)
        else:
            logger.info("non-pipeline cell")
            grpo_with_loss_net = grpo_model_train
        grpo_with_loss = _VirtualDatasetCell(grpo_with_loss_net)
        lr = LearningRate(learning_rate=grpo_config.start_lr, end_learning_rate=grpo_config.end_lr,
                          warmup_steps=grpo_config.warmup_step, decay_steps=grpo_config.decay_steps)
        params = grpo_with_loss.trainable_params()
        group_params = set_weight_decay(params)

        if grpo_config.optimizer == "lamb":
            optimizer = nn.Lamb(group_params, learning_rate=lr)
        elif grpo_config.opt_offload:
            optimizer = AdamWeightDecayOp(group_params, learning_rate=lr, eps=grpo_config.eps, beta1=grpo_config.beta1,
                                          beta2=grpo_config.beta2, param_init_type=sft_model_config.param_init_type)
        else:
            optimizer = FP32StateAdamWeightDecay(group_params, learning_rate=lr, beta1=grpo_config.beta1,
                                                 beta2=grpo_config.beta2, eps=grpo_config.eps)

        loss_scale_value = math.pow(2, 12)
        update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value,
                                                 scale_factor=2, scale_window=1000)

        if sft_model_config.parallel_config.pipeline_stage > 1:
            logger.info("pipeline cell")
            grpo_with_grad = TrainPipelineWithLossScaleCell_GRPO(grpo_with_loss, optimizer=optimizer, config=sft_model_config,
                                                                 scale_update_cell=update_cell)
        else:
            logger.info("non-pipeline cell")
            grpo_with_grad = TrainOneStepWithLossScale_GRPO(grpo_with_loss, optimizer=optimizer, config=sft_model_config,
                                                            scale_update_cell=update_cell, enable_global_norm=True)
        return grpo_with_grad

    def _init_grpo_dataset_before_train(self):
        '''
        Build dataset for graph pre-compilation.
        '''
        grpo_config = self.grpo_config
        column_names = ["prompt_completion_ids", "prompts_mask", "responses_mask",
                        "ref_per_token_logps", "advantages"]
        logger.info(f"store.length: {len(self.store)}")
        pipeline = GRPOIteratorStore(self.store)
        dataset = GeneratorDataset(pipeline, column_names=column_names)
        logger.info(f"dataset.len: {len(dataset)}")
        type_cast_op_int32 = TypeCast(mindspore.int32)
        type_cast_op_fp16 = TypeCast(mindspore.float16)
        dataset = dataset.map(operations=type_cast_op_int32, input_columns="prompt_completion_ids")
        dataset = dataset.map(operations=type_cast_op_int32, input_columns="prompts_mask")
        dataset = dataset.map(operations=type_cast_op_int32, input_columns="responses_mask")
        dataset = dataset.map(operations=type_cast_op_fp16, input_columns="ref_per_token_logps")
        dataset = dataset.map(operations=type_cast_op_fp16, input_columns="advantages")
        micro_batch_num = 1
        if self.sft_model_config_train.parallel_config.pipeline_stage > 1:
            micro_batch_num = self.sft_model_config_train.parallel_config.micro_batch_num
        logger.info(
            f"##################### bs:{grpo_config.batch_size}, dp:{self.sft_model_config_train.parallel_config.data_parallel}, micro_batch_num:{micro_batch_num}")
        dataset = dataset.batch(
            batch_size=grpo_config.batch_size * self.sft_model_config_train.parallel_config.data_parallel * micro_batch_num)
        return dataset

    def train(self):
        dataset = self._init_grpo_dataset_before_train()
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True)
        context.set_auto_parallel_context(pipeline_stages=self.train_pp_stage)
        grpo_with_grad = self.grpo_with_grad
        # logger.info(f"train cell is {grpo_with_grad}")
        sink_process = mindspore.data_sink(grpo_with_grad, dataset, sink_size=self.grpo_config.sink_size)
        steps = dataset.dataset_size // self.grpo_config.sink_size
        logger.info(
            f"dataset size is {dataset.dataset_size}, sink size is {self.grpo_config.sink_size}, total steps is {steps}")
        for step in range(steps):
            ep_begin_time = time.time()
            out = sink_process()
            end_time = time.time()
            logger.info("step {}, end at {}, elapsed time {} \n------------------------------- "
                        .format(step, time.strftime('%H:%M:%S', time.localtime(end_time)), end_time - ep_begin_time))
            logger.info(" loss: {} | lr: {} | is overflow: {} | loss scale: {}"
                        .format(out[0], out[1], out[2], out[3]))

    def offload_optimizer(self):
        if self.optimizer_on_device is False:
            return
        logger.info(f'before offload stf train {ms.hal.memory_stats()}')
        for param in self.grpo_with_grad.optimizer.moments1:
            param._offload()
        for param in self.grpo_with_grad.optimizer.moments2:
            param._offload()
        if self.train_pp_stage > 1:
            for param in self.grpo_with_grad.accu_grads:
                param._offload()
        logger.info(f'after offload stf train {ms.hal.memory_stats()}')
        self.optimizer_on_device = False

    def load_optimizer(self):
        if self.optimizer_on_device:
            return
        logger.info(f'before load stf train {ms.hal.memory_stats()}')
        for param in self.grpo_with_grad.optimizer.moments1:
            param._load()
        for param in self.grpo_with_grad.optimizer.moments2:
            param._load()
        if self.train_pp_stage > 1:
            for param in self.grpo_with_grad.accu_grads:
                param._load()
        logger.info(f'after load stf train {ms.hal.memory_stats()}')
        self.optimizer_on_device = True

    def load_model(self):
        if self.model_on_device:
            return
        for param in self.grpo_with_grad.network.get_parameters(expand=True):
            param._load()
        self.model_on_device = True

    def offload_model(self):
        if self.model_on_device is False:
            return
        for param in self.grpo_with_grad.network.get_parameters(expand=True):
            param._offload()
        self.model_on_device = False

    def push_to_store(self, data):
        self.store = data

    def save_checkpoint(self, rank_id=0, steps=0):
        """ save checkpoint """
        if self.grpo_config.save_ckpt_dir:
            logger.info("Save checkpoints in {}".format(self.grpo_config.save_ckpt_dir))
            train_save_dir = os.path.join(self.grpo_config.save_ckpt_dir, 'train', f"rank_{rank_id}")
            if not os.path.exists(train_save_dir):
                os.makedirs(train_save_dir)
            grpo_filename = os.path.join(train_save_dir, "policy_model_epoch_{}.ckpt".format(steps))
            ms.save_checkpoint(self.grpo_model_train.grpo_model_train.policy_model,
                               grpo_filename, integrated_save=False)
        else:
            logger.info("There is no checkpoint to save!")
