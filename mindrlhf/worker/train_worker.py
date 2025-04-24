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
from glob import glob

# mindspore
import mindspore
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
from mindformers.tools.logger import logger

# mindrlhf
from mindrlhf.utils.adam import AdamWeightDecayOp
from mindrlhf.utils.utils import LearningRate, FP32StateAdamWeightDecay
from mindrlhf.wrapper import TrainOneStepWithLossScaleGRPO, TrainPipelineWithLossScaleCellGRPO
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
        os.environ["RUN_MODE"] = sft_config_train.run_mode
        sft_config_train.model.model_config.parallel_config.recompute = sft_config_train.recompute_config
        sft_model_config_train = LlamaConfig(**sft_config_train.model.model_config)
        sft_model_config_train.checkpoint_name_or_path = args.load_sft_checkpoint_train
        sft_model_config_train.model_name = "llama"
        self.sft_ckpt_path_train = sft_model_config_train.checkpoint_name_or_path
        sft_model_config_train.checkpoint_name_or_path = None

        self.train_pp_stage = sft_model_config_train.parallel_config.pipeline_stage or 1
        context.set_auto_parallel_context(pipeline_stages=self.train_pp_stage)
        self.sft_model_config_train = sft_model_config_train
        sft_model_config_train.name = "grpo_train"
        policy_model = CausalLMHybrid(sft_model_config_train, self.grpo_config)
        self.grpo_model_train = GRPOModelTrain(grpo_config, policy_model)
        self.grpo_model_train.set_train(True)
        self.grpo_with_grad = self._init_grpo_network_and_optimizer()
        self.store = []

        self.model_on_device = True
        self.optimizer_on_device = True

        self.tensor_writer = self.args.tensor_writer

    def model(self):
        return self.grpo_model_train

    def compile(self):
        # compile and save strategy
        self.grpo_with_grad.set_train(True)
        self.grpo_model_train.grpo_model_train.policy_model.model.set_train(True)

        context.set_auto_parallel_context(pipeline_stages=self.train_pp_stage)
        train_bs = self.grpo_config.batch_size * self.sft_model_config_train.parallel_config.micro_batch_num
        prompt_completion_ids = ms.Tensor(shape=(train_bs, self.grpo_config.seq_length+1),
                                dtype=ms.int32) # [bs, seq_len+1]
        responses_mask = ms.Tensor(shape=(train_bs, self.grpo_config.seq_length),
                                dtype=ms.int32)   # [bs, seq_len]
        ref_per_token_logps = ms.Tensor(shape=(train_bs, self.grpo_config.seq_length),
                                dtype=ms.float16) # [bs, seq_len]
        advantages = ms.Tensor(shape=(train_bs, self.grpo_config.seq_length),
                                dtype=ms.float16)  # [bs, seq_len]
        actual_seq_length = ms.Tensor(shape=(train_bs, self.grpo_config.pack_num),
                                dtype=ms.int32)  # [bs, packed_sample_num]
        sample_index = ms.Tensor(shape=(train_bs, self.grpo_config.seq_length),
                                dtype=ms.int32) #[bs, seq_len]
        sample_valid_len = ms.Tensor(shape=(train_bs, self.grpo_config.pack_num),
                                dtype=ms.int32)  #[bs, packed_sample_num]

        start_time = time.time()
        stage_name = 'train'
        strategy_path = self.grpo_config.save_strategy_dir
        context.set_auto_parallel_context(
            strategy_ckpt_config={
                "save_file":
                    f"{strategy_path}/strategy_file/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt"},
            pipeline_stages=self.train_pp_stage
        )
        # self.grpo_with_grad.compile(**data)
        self.grpo_with_grad.compile(prompt_completion_ids, responses_mask, ref_per_token_logps, advantages,
                                    actual_seq_length, sample_index, sample_valid_len)
        stage_name = 'other'
        context.set_auto_parallel_context(
            strategy_ckpt_config={
                "save_file":
                    f"{strategy_path}/strategy_file/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt"})
        logger.info(f"grpo_with_grad time: {format_time_delta(time.time() - start_time)}")

    def load_checkpoint(self):
        if self.args.load_ckpt_format == "safetensors":
            self.model_on_device = True
            self.optimizer_on_device = True
            return self._load_checkpoint_safetensors()
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
            param_not_load, ckpt_not_load = mindspore.load_param_into_net(
                self.grpo_model_train.grpo_model_train.policy_model, new_param_dict)
            logger.info(f"param not load: {param_not_load}")
            logger.info(f"ckpt not load: {ckpt_not_load}")
    
    def _load_checkpoint_safetensors(self):
        network = self.grpo_model_train.grpo_model_train.policy_model
        name_map = None
        try:
            load_checkpoint_files = glob(
                os.path.join(self.sft_ckpt_path_train, f"*.safetensors"))
            load_checkpoint_files.sort()
            name_map = network.obtain_name_map(load_checkpoint_files)
        except Exception as e:
            raise TypeError(f"Please complete abstract function obtain_name_map. Details: {e}") from e

        strategy_path =os.path.join(self.grpo_config.save_strategy_dir, 'merge_strategy/train_policy_merged_strategy.ckpt')
        ms.load_distributed_checkpoint(
            network=network,
            predict_strategy=strategy_path,
            unified_safetensors_dir=self.sft_ckpt_path_train,
            format='safetensors',
            name_map=name_map
        )


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
            grpo_with_loss_net = PipelineCell(grpo_model_train,
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
            grpo_with_grad = TrainPipelineWithLossScaleCellGRPO(grpo_with_loss, optimizer=optimizer, config=sft_model_config,
                                                                scale_update_cell=update_cell)
        else:
            logger.info("non-pipeline cell")
            grpo_with_grad = TrainOneStepWithLossScaleGRPO(grpo_with_loss, optimizer=optimizer, config=sft_model_config,
                                                           scale_update_cell=update_cell, enable_global_norm=True)
        return grpo_with_grad

    def _init_grpo_dataset_before_train(self):
        '''
        Build dataset for graph pre-compilation.
        '''
        grpo_config = self.grpo_config
        column_names = ["prompt_completion_ids", "responses_mask",
                    "ref_per_token_logps", "advantages",
                    "actual_sequence_length", "sample_index", "sample_valid_length"]
        logger.info(f"store.length: {len(self.store)}")
        pipeline = GRPOIteratorStore(self.store)
        dataset = GeneratorDataset(pipeline, column_names=column_names)
        logger.info(f"dataset.len: {len(dataset)}")
        type_cast_op_int32 = TypeCast(mindspore.int32)
        type_cast_op_fp16 = TypeCast(mindspore.float16)
        dataset = dataset.map(operations=type_cast_op_int32, input_columns="prompt_completion_ids")
        dataset = dataset.map(operations=type_cast_op_int32, input_columns="responses_mask")
        dataset = dataset.map(operations=type_cast_op_fp16, input_columns="ref_per_token_logps")
        dataset = dataset.map(operations=type_cast_op_fp16, input_columns="advantages")
        dataset = dataset.map(operations=type_cast_op_int32, input_columns="actual_sequence_length")
        dataset = dataset.map(operations=type_cast_op_int32, input_columns="sample_index")
        dataset = dataset.map(operations=type_cast_op_int32, input_columns="sample_valid_length")
        micro_batch_num = 1
        if self.sft_model_config_train.parallel_config.pipeline_stage > 1:
            micro_batch_num = self.sft_model_config_train.parallel_config.micro_batch_num
        logger.info(
            f"##################### bs:{grpo_config.batch_size}, dp:{self.sft_model_config_train.parallel_config.data_parallel}, micro_batch_num:{micro_batch_num}")
        dataset = dataset.batch(
            batch_size=grpo_config.batch_size * self.sft_model_config_train.parallel_config.data_parallel * micro_batch_num, drop_remainder=True)
        return dataset

    def train(self):
        dataset = self._init_grpo_dataset_before_train()
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True)
        context.set_auto_parallel_context(pipeline_stages=self.train_pp_stage)
        formatter = lambda out: out.asnumpy() if isinstance(out, Tensor) else out

        iterator = dataset.create_dict_iterator()
        logger.info(f"dataset size is {len(dataset)}")

        for step, databatch in enumerate(iterator):

            ep_begin_time = time.time()
            out = self.grpo_with_grad(**databatch)
            end_time = time.time()
            logger.info(
                "step {} | loss: {} | lr: {} | is overflow: {} | loss scale: {} | elapsed time {} \n-------------------------------".format(
                    step, formatter(out[0]), formatter(out[1]), formatter(out[2]), formatter(out[3]), end_time - ep_begin_time))

        if self.tensor_writer:
            self.tensor_writer.add_scalar("loss", out[0].asnumpy(), global_step=step)
            self.tensor_writer.add_scalar("lr", out[1].asnumpy(), global_step=step)
            self.tensor_writer.add_scalar("overflow", out[2].asnumpy(), global_step=step)
            self.tensor_writer.add_scalar("loss-scale", out[3].asnumpy(), global_step=step)

    def offload_optimizer(self):
        if self.optimizer_on_device is False:
            logger.info(f'no need for offload_optimizer because optimizer_on_device is False ')
            return
        logger.info(f'before offload stf train optimizer {ms.hal.memory_stats()}')
        for param in self.grpo_with_grad.optimizer.moments1:
            param._offload()
        for param in self.grpo_with_grad.optimizer.moments2:
            param._offload()
        if self.train_pp_stage > 1:
            for param in self.grpo_with_grad.accu_grads:
                param._offload()
        logger.info(f'after offload stf train optimizer {ms.hal.memory_stats()}')
        self.optimizer_on_device = False

    def load_optimizer(self):
        if self.optimizer_on_device:
            logger.info(f'no need for load_optimizer because optimizer_on_device is True ')
            return
        logger.info(f'before load stf train optimizer {ms.hal.memory_stats()}')
        for param in self.grpo_with_grad.optimizer.moments1:
            param._load()
        for param in self.grpo_with_grad.optimizer.moments2:
            param._load()
        if self.train_pp_stage > 1:
            for param in self.grpo_with_grad.accu_grads:
                param._load()
        logger.info(f'after load stf train optimizer {ms.hal.memory_stats()}')
        self.optimizer_on_device = True

    def load_model(self):
        if self.model_on_device:
            return
        logger.info(f'before load stf train model {ms.hal.memory_stats()}')
        for param in self.grpo_with_grad.network.get_parameters(expand=True):
            param._load()
        logger.info(f'after load stf train model {ms.hal.memory_stats()}')
        self.model_on_device = True

    def offload_model(self):
        if self.model_on_device is False:
            return
        logger.info(f'after offload stf train model {ms.hal.memory_stats()}')
        for param in self.grpo_with_grad.network.get_parameters(expand=True):
            param._offload()
        logger.info(f'after offload stf train model {ms.hal.memory_stats()}')
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
