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
""" Train Worker """

# python
import os
import time
import math

# mindspore
import mindspore as ms
from mindspore import Tensor
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
from mindrlhf.utils.utils import LearningRate, FP32StateAdamWeightDecay, print_perf_stat
from mindrlhf.wrapper import TrainOneStepWithLossScaleGRPO, TrainPipelineWithLossScaleCellGRPO
from mindrlhf.models.grpo_models import CausalLMHybrid, GRPOModelTrain
from mindrlhf.utils.dataset import GRPOIteratorStore
from mindrlhf.worker.worker import Worker


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
        self.save_strategy_dir = grpo_config.save_strategy_dir

    def model(self):
        return self.grpo_model_train

    def compile(self):
        """ compile and save strategy """
        self.grpo_with_grad.set_train(True)
        self.grpo_model_train.grpo_model_train.policy_model.model.set_train(True)
        context.set_auto_parallel_context(pipeline_stages=self.train_pp_stage)
        train_bs = self.grpo_config.batch_size * self.sft_model_config_train.parallel_config.micro_batch_num
        fake_data_1 = ms.Tensor(shape=(train_bs, self.grpo_config.seq_length + 1),
                                dtype=ms.int32)
        fake_data_2 = ms.Tensor(shape=(train_bs, self.grpo_config.seq_length),
                                dtype=ms.float32)
        fake_data_3 = ms.Tensor(shape=(train_bs,), dtype=ms.int32)

        start_time = time.time()
        stage_name = 'train'
        context.set_auto_parallel_context(
            strategy_ckpt_config={
                "save_file":
                    f"{self.save_strategy_dir}/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt"},
            pipeline_stages=self.train_pp_stage
        )
        self.grpo_with_grad.compile(prompt_completion_ids=fake_data_1, prompts_mask=fake_data_1,
                                    responses_mask=fake_data_1, ref_per_token_logps=fake_data_2,
                                    advantages=fake_data_3)
        stage_name = 'other'
        context.set_auto_parallel_context(
            strategy_ckpt_config={
                "save_file":
                    f"{self.save_strategy_dir}/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt"})
        end_time = time.time()
        print_perf_stat(start_time, end_time, "train model compile")

    def load_checkpoint(self):
        """ load checkpoint """
        load_ckpt_func = load_distributed_checkpoint if self.grpo_config.use_parallel else ms.load_checkpoint
        logger.info(f"self.grpo_config.use_parallel is {self.grpo_config.use_parallel}, {load_ckpt_func}")
        if self.sft_ckpt_path_train:
            self.model_on_device = True
            self.optimizer_on_device = True
            param_dict = load_ckpt_func(self.sft_ckpt_path_train)
            new_param_dict = {'grpo_model_train.policy_model.model.' + k: v for k, v in param_dict.items()}
            logger.info(f"begin to load train policy model from: {self.sft_ckpt_path_train}")
            for _, param in self.grpo_model_train.grpo_model_train.policy_model.parameters_and_names():
                logger.info(f"train model para names:   {param.name}")
            param_not_load, ckpt_not_load = ms.load_param_into_net(
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
            grpo_with_grad = TrainPipelineWithLossScaleCellGRPO(grpo_with_loss, optimizer=optimizer,
                                                                config=sft_model_config,
                                                                scale_update_cell=update_cell)
        else:
            logger.info("non-pipeline cell")
            grpo_with_grad = TrainOneStepWithLossScaleGRPO(grpo_with_loss, optimizer=optimizer,
                                                           config=sft_model_config, scale_update_cell=update_cell,
                                                           enable_global_norm=True)
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
        type_cast_op_int32 = TypeCast(ms.int32)
        type_cast_op_fp16 = TypeCast(ms.float16)
        dataset = dataset.map(operations=type_cast_op_int32, input_columns="prompt_completion_ids")
        dataset = dataset.map(operations=type_cast_op_int32, input_columns="prompts_mask")
        dataset = dataset.map(operations=type_cast_op_int32, input_columns="responses_mask")
        dataset = dataset.map(operations=type_cast_op_fp16, input_columns="ref_per_token_logps")
        dataset = dataset.map(operations=type_cast_op_fp16, input_columns="advantages")
        micro_batch_num = 1
        if self.sft_model_config_train.parallel_config.pipeline_stage > 1:
            micro_batch_num = self.sft_model_config_train.parallel_config.micro_batch_num
        batch_size = (grpo_config.batch_size *
                      self.sft_model_config_train.parallel_config.data_parallel *
                      micro_batch_num)
        logger.info(
            f"bs:{grpo_config.batch_size}, "
            f"dp:{self.sft_model_config_train.parallel_config.data_parallel}, "
            f"micro_batch_num:{micro_batch_num}, "
            f"bs in dataset: {batch_size}")
        dataset = dataset.batch(batch_size=batch_size)
        return dataset

    def train(self):
        """ train """
        dataset = self._init_grpo_dataset_before_train()
        context.set_auto_parallel_context(pipeline_stages=self.train_pp_stage)
        formatter = lambda out: out.asnumpy() if isinstance(out, Tensor) else out

        train_start_time = time.time()
        iterator = dataset.create_dict_iterator()
        logger.info(f"dataset size is {dataset.dataset_size}")

        for step, databatch in enumerate(iterator):
            ep_begin_time = time.time()
            out = self.grpo_with_grad(**databatch)
            end_time = time.time()
            print_perf_stat(ep_begin_time, end_time, f"train step {step}")
            logger.info(" loss: {} | lr: {} | is overflow: {} | loss scale: {}"
                        .format(formatter(out[0]), formatter(out[1]), formatter(out[2]), formatter(out[3])))
        train_end_time = time.time()
        print_perf_stat(train_start_time, train_end_time, "train model all steps")

    def offload_optimizer(self):
        """ offload optimizer """
        if self.optimizer_on_device is False:
            return
        logger.info(f'before offload stf train {ms.hal.memory_stats()}')
        start_time = time.time()
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
        end_time = time.time()
        print_perf_stat(start_time, end_time, "offload stf train optimizer")
        logger.info(f'after offload stf train {ms.hal.memory_stats()}')
        self.optimizer_on_device = False

    def load_optimizer(self):
        """ load optimizer """
        if self.optimizer_on_device:
            return
        logger.info(f'before load stf train {ms.hal.memory_stats()}')
        start_time = time.time()
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
        end_time = time.time()
        print_perf_stat(start_time, end_time, "load stf train optimizer")
        logger.info(f'after load stf train {ms.hal.memory_stats()}')
        self.optimizer_on_device = True

    def load_model(self):
        if self.model_on_device:
            return
        start_time = time.time()
        for param in self.grpo_with_grad.network.get_parameters(expand=True):
            # pylint: disable=W0212
            param._load()
        end_time = time.time()
        print_perf_stat(start_time, end_time, "load stf train model")
        self.model_on_device = True

    def offload_model(self):
        if self.model_on_device is False:
            return
        start_time = time.time()
        for param in self.grpo_with_grad.network.get_parameters(expand=True):
            # pylint: disable=W0212
            param._offload()
        end_time = time.time()
        print_perf_stat(start_time, end_time, "offload stf train model")
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
