# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
MindRLHF utils
"""
import os
import json
import time
import hashlib
import copy
import stat
import yaml
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
import mindspore as ms
from mindspore.common.tensor import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR, CosineDecayLR
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.context import ParallelMode
from mindspore.communication.management import get_rank, get_group_size, create_group
from mindspore.nn import AdamWeightDecay
from mindspore.common import Parameter, ParameterTuple
from mindspore.common.initializer import initializer
import mindspore.communication.management as D
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore import context
import mindspore.common.dtype as mstype

from mindformers.tools.ckpt_transform import TransformCkpt
from mindformers import logger

import mindrlhf.utils.reshard_optimizer as reshard_optimizer
from mindrlhf.configs.grpo_configs import GRPOConfig

__all__ = ['set_pipeline_parallel_context', 'is_last_stage', 'is_first_stage',
           'FP32StateAdamWeightDecay', 'TimePoint', 'LearningRate',
           'GlobalNorm', 'ClipByGlobalNorm', "transfer_from_str_to_bool",
           "ckpt_transfer_for_generate", "yaml_to_dataclass", "set_perf_stats",
           "print_perf_stat", "_get_pipeline_group", "convert_index_json_total",
           "save_prompt_completions_data", "add_metrics_to_tensorboard", "get_dp_rank",
           "get_checkpoint_name", "ensure_total_ckpt_is_less_than_limit",
           "load_param_to_net", "record_last_ckpt_to_json"]


PERF_STATS = False

def yaml_to_dataclass(file_path, dataclass_type):
    """
    Parse a YAML file into an instance of the specified dataclass.

    Parameters:
    file_path (str): Path to the YAML file.
    dataclass_type (type): The dataclass type to instantiate.

    Returns:
    dataclass_type: An instance of the dataclass.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return dataclass_type(**data)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except yaml.YAMLError as error:
        print(f"Error parsing YAML file: {error}")
        return None
    except TypeError as error:
        print(f"Data type mismatch: {error}")
        return None

def set_pipeline_parallel_context(ppo_config):
    """Set pipeline parallel context."""
    D.init()
    device_num = D.get_group_size()
    rank_id = D.get_rank()
    context.reset_auto_parallel_context()

    if hasattr(ppo_config.parallel_config, 'optimizer_shard'):
        optimizer_shard = bool(ppo_config.parallel_config.get("optimizer_shard"))
    elif hasattr(ppo_config.parallel, 'enable_parallel_optimizer'):
        optimizer_shard = bool(ppo_config.parallel.get("enable_parallel_optimizer"))
    else:
        optimizer_shard = True

    context.set_auto_parallel_context(
        parallel_mode=ppo_config.parallel_mode, gradients_mean=False,
        full_batch=bool(ppo_config.full_batch), loss_repeated_mean=True,
        device_num=device_num,
        enable_parallel_optimizer=optimizer_shard,
        pipeline_stages=ppo_config.parallel_config.get("pipeline_stage"),
        enable_alltoall=bool(ppo_config.enable_alltoall),
        strategy_ckpt_save_file='strategy.ckpt')
    set_algo_parameters(elementwise_op_strategy_follow=True)
    _set_multi_subgraphs()
    return rank_id, device_num


def is_last_stage(pipeline_stage):
    device_num = D.get_group_size()
    rank = D.get_rank()
    per_stage_num = int(device_num / pipeline_stage)
    return device_num - 1 - per_stage_num < rank <= device_num - 1


def is_first_stage(pipeline_stage):
    device_num = D.get_group_size()
    rank = D.get_rank()
    per_stage_num = int(device_num / pipeline_stage)
    return rank < per_stage_num


class FP32StateAdamWeightDecay(AdamWeightDecay):
    r"""
        This class is almost same with the mindspore's AdamWeightDecay implements, the
        only difference is the optimizer's state will be always initialized with float32,
        where the original AdamWeightDecay will initialize the optimizer's state with float16,
        if the parameters are initialized with fp16.
        This setting will avoid overflow in training PanGu-Alpha model using fp16.
    """

    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(FP32StateAdamWeightDecay, self).__init__(params, learning_rate=learning_rate,
                                                       beta1=beta1,
                                                       beta2=beta2,
                                                       eps=eps,
                                                       weight_decay=weight_decay)

        self.moments1 = self.clone_state(self.parameters, prefix='adam_m', init='zeros')
        self.moments2 = self.clone_state(self.parameters, prefix='adam_v', init='zeros')

    def clone_state(self, parameter_tuple, prefix, init):
        r"""
            parameter_tuple: ParameterTuple. The parameters of the network
            prefix: str. The prefix name of the parameters
            init: str. The initialization method
        """
        new = []
        for old_param in parameter_tuple:
            new_state = Parameter(initializer(init, shape=old_param.shape, dtype=mstype.float32))
            new_state.param_info = old_param.param_info.clone()
            if hasattr(old_param.param_info, "cloned_obj"):
                old_param.param_info.cloned_obj.append(new_state)
            else:
                old_param.param_info.cloned_obj = [new_state]
            new_state.is_init = False
            new_state.set_data(initializer(init, shape=old_param.shape, dtype=mstype.float32))
            new_state.name = prefix + '.' + new_state.name
            new.append(new_state)
        return ParameterTuple(new)


get_square_sum = C.MultitypeFuncGraph("get_square_sum")


@get_square_sum.register("Tensor", "Number")
def _get_square_sum(grad, value):
    norm = P.ReduceSum(False)(F.square(grad), ()) / value
    norm = F.expand_dims(F.cast(norm, mstype.float32), 0)
    return norm


apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")


@apply_global_norm.register("Bool", "Tensor", "Tensor", "Tensor")
def _apply_global_norm(enable_grad_fp16, clip_norm, global_norm, grad):
    if enable_grad_fp16:
        grad = P.Cast()(grad * clip_norm / global_norm, mstype.float16)
    else:
        grad = grad * clip_norm / global_norm
    return grad


def _get_model_parallel_group(mp):
    """

    Calculate the communication group of model parallel dim in one pipeline stage

    """
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    stage_id = rank // per_stage_device_nums
    local_stage_rank_id = rank % per_stage_device_nums
    index = local_stage_rank_id // mp
    group = range(0, mp)
    rank_str_list = [str(x + index * mp + stage_id * per_stage_device_nums) for x in group]
    rank_list_str = "-".join(rank_str_list)
    rank_list = [x + index * mp + stage_id * per_stage_device_nums for x in group]
    return rank_list, rank_list_str


def _get_pipeline_group():
    """

    Calculate the communication group between all pipeline stages

    """
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    local_stage_rank_id = rank % per_stage_device_nums
    group = range(0, stage_nums)
    rank_list = [local_stage_rank_id + x * per_stage_device_nums for x in group]
    rank_str_list = [str(local_stage_rank_id + x * per_stage_device_nums) for x in group]
    rank_list_str = "-".join(rank_str_list)
    return rank_list, rank_list_str


class GlobalNorm(nn.Cell):
    """
    Calculate the global norm value of given tensors
    """

    def __init__(self, params, config):
        super(GlobalNorm, self).__init__()
        self.norm = nn.Norm()
        self.hyper_map = C.HyperMap()
        self.is_pipeline = context.get_auto_parallel_context("pipeline_stages") > 1
        self.is_data_parallel = context.get_auto_parallel_context("parallel_mode") == ParallelMode.DATA_PARALLEL
        self.config = config
        self.group_size = 1
        if self.is_data_parallel:
            self.merge_op = P.identity()
        else:
            self.merge_op = P.AllReduce()
        if self.is_pipeline:
            if context.get_auto_parallel_context("enable_parallel_optimizer"):
                self.group_size = get_group_size() // config.parallel_config.pipeline_stage
            else:
                self.group_size = config.parallel_config.model_parallel
            group_list, group_name = _get_model_parallel_group(self.group_size)
            # In avoid of the group name too long
            hashed = hashlib.md5(group_name.encode()).hexdigest()[:48]
            print(f"Creating hash value for the group_name hash({group_name})={hashed}")
            group_name = str(hashed)
            create_group(group_name, group_list)
            self.allreduce = P.AllReduce(group=group_name)
            pipeline_group_list, pipeline_group_name = _get_pipeline_group()
            hashed = hashlib.md5(pipeline_group_name.encode()).hexdigest()[:48]
            print(f"Creating hash value for the group_name hash({pipeline_group_name})={hashed}")
            pipeline_group_name = str(hashed)
            create_group(pipeline_group_name, pipeline_group_list)
            self.allreduce2 = P.AllReduce(group=pipeline_group_name)
        else:
            self.group_size = get_group_size()
        self.allreduce_group_size = ()
        if self.is_data_parallel:
            self.allreduce_group_size = (1,) * len(params)
        else:
            self.allreduce_group_size = self._get_scale_for_gradient_norm(params)

    def construct(self, grads):
        """Calculate global norm construct"""
        square_sum = self.hyper_map(get_square_sum, grads, self.allreduce_group_size)
        square_reduce_sum = F.addn(square_sum)
        if self.is_pipeline:
            stage_square_reduce_sum = self.allreduce(square_reduce_sum)
            global_square_reduce_sum = self.allreduce2(stage_square_reduce_sum)
            global_norms = F.sqrt(global_square_reduce_sum)
        else:
            global_norms = F.sqrt(self.merge_op(square_reduce_sum))
        return grads, global_norms

    def _get_scale_for_gradient_norm(self, params):
        """
        get scale for gradient norm
        """
        allreduce_group_size = ()
        for x in params:
            if "projection.bias" not in x.name and "layernorm" not in x.name and "embedding_table" not in x.name:
                allreduce_group_size = allreduce_group_size + (1.0,)
            elif "embedding_table" not in x.name:
                allreduce_group_size = allreduce_group_size + (self.group_size * 1.0,)
            else:
                if not self.config.parallel_config.vocab_emb_dp and "position_embedding.embedding_table" not in x.name \
                        and "top_query_embedding_table" not in x.name:
                    allreduce_group_size = allreduce_group_size + \
                                           (self.config.parallel_config.data_parallel * 1.0,)
                else:
                    allreduce_group_size = allreduce_group_size + (self.group_size * 1.0,)
        return allreduce_group_size


class ClipByGlobalNorm(nn.Cell):
    """

    Clip grads by global norm

    """

    def __init__(self, params, config, clip_norm=1.0):
        super(ClipByGlobalNorm, self).__init__()
        self.global_norm = GlobalNorm(params, config)
        self.clip_norm = Tensor([clip_norm], mstype.float32)
        self.hyper_map = C.HyperMap()
        if config.param_init_type == mstype.float16:
            self.enable_grad_fp16 = True
        else:
            self.enable_grad_fp16 = False

    def construct(self, grads):
        """Clip grads by global norm construct"""
        grads, global_norm_value = self.global_norm(grads)
        cond = P.GreaterEqual()(global_norm_value, self.clip_norm)
        global_norm = F.select(cond, global_norm_value, self.clip_norm)
        grads = self.hyper_map(F.partial(apply_global_norm, self.enable_grad_fp16, self.clip_norm, global_norm), grads)
        return grads, global_norm_value


class LearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for PanguAlpha network.
    """

    def __init__(self,
                 learning_rate,
                 end_learning_rate,
                 warmup_steps,
                 decay_steps,
                 power=1.0,
                 use_cosine=True):
        super(LearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate,
                                          decay_steps, power)
        self.cosine_decay_lr = CosineDecayLR(end_learning_rate, learning_rate,
                                             decay_steps)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()
        self.use_cosine = use_cosine

    def construct(self, global_step):
        """dynamic learning rate"""
        if not self.use_cosine:
            decay_lr = self.decay_lr(global_step)
        else:
            decay_lr = self.cosine_decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step),
                                  mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr


class TimePoint:
    """A helper function for recording the time spend."""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def set_start(self):
        """Set the start time point"""
        self.start_time = time.time()

    def set_end(self):
        """Set the end time point"""
        self.end_time = time.time()

    def get_spend_time(self):
        """Get the time spend between end and start"""
        return self.end_time - self.start_time


def get_testing_dataset_path(dataset_name):
    """Get dataset path for testing."""
    dataset_dict = {
        "cvalues_1024": "/path/cvalues_1024.mindrecord",
        "cvalues_2048": "/path/cvalues_2048.mindrecord",
    }
    dataset = dataset_dict.get(dataset_name)
    if dataset is None:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    return dataset


def get_valid_length_each_example(input_ids, pad_token_id):
    """get valid length and max length in a batch"""
    batch_size = input_ids.shape[0]
    valid_length_each_example = []
    for i in range(batch_size):
        # As the nonzero returns the index and we need length
        valid_length_each_example.append(
            np.max(np.argwhere(input_ids[i] != pad_token_id))
            + 1
        )
    valid_length_each_example = np.array(valid_length_each_example)
    max_length = np.max(valid_length_each_example)
    return valid_length_each_example, max_length


def transfer_from_str_to_bool(input_str):
    """transfer from string to bool"""
    if isinstance(input_str, bool):
        return input_str
    if input_str.lower() == "true":
        return True
    if input_str.lower() == "false":
        return False
    raise ValueError(f"input_str {input_str} is not supported.")


def get_strategy(startegy_path, rank_id=None):
    """Merge strategy if strategy path is dir

    Args:
        startegy_path (str): The path of stategy.
        rank_id (int): The rank id of device.

    Returns:
        None or strategy path
    """
    if not startegy_path or startegy_path == "None":
        return None

    if not os.path.exists(startegy_path):
        raise ValueError(f'{startegy_path} not found!')

    if os.path.isfile(startegy_path):
        return startegy_path

    if os.path.isdir(startegy_path):
        if rank_id:
            merge_path = os.path.join(startegy_path, f'merged_ckpt_strategy_{rank_id}.ckpt')
        else:
            merge_path = os.path.join(startegy_path, f'merged_ckpt_strategy.ckpt')

        if os.path.exists(merge_path):
            os.remove(merge_path)

        ms.merge_pipeline_strategys(startegy_path, merge_path)
        return merge_path

    return None


def ckpt_transfer_for_generate(load_sft_checkpoint):
    transform_ckpt = TransformCkpt(
        rank_id=get_rank(),
        world_size=get_group_size(),
        transform_process_num=get_group_size()
    )

    generate_ckpt = "./generate_ckpt"
    return transform_ckpt(
        src_checkpoint=load_sft_checkpoint,
        dst_checkpoint_dir=generate_ckpt,
        src_strategy="./train_policy_strategy/",
        dst_strategy="./generate_policy_strategy/",
        prefix="./generate_policy_"
    )


def set_perf_stats(grpo_config: GRPOConfig):
    global PERF_STATS
    if grpo_config.rl_config.performance_stats:
        logger.info("grpo performance statistics is on")
        PERF_STATS = True


def print_perf_stat(start_time, end_time, perf_object_str):
    global PERF_STATS
    if PERF_STATS:
        logger.info(f"Performance Statistics: {perf_object_str} costs {end_time - start_time}s")


def convert_index_json_total(load_checkpoint, converted_dir, convert_map_dict_lst, is_qkv_concat):
    """convert mapping file if exists"""
    index_path = os.path.join(load_checkpoint, 'model.safetensors.index.json')
    if not os.path.exists(index_path):
        logger.warning(f"The given path contains no 'model.safetensors.index.json' file.")
        return
    with open(index_path, 'r') as f:
        data = json.load(f)
    weight_map = data.get("weight_map")
    total_weight_map = {}
    for convert_map_dict_func in convert_map_dict_lst:
        new_weight_map = copy.deepcopy(weight_map)
        converted_weight_map = convert_map_dict_func(new_weight_map, qkv_concat=is_qkv_concat)
        total_weight_map.update(converted_weight_map)
    flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(os.open(os.path.join(converted_dir, 'param_name_map.json'), flags_, 0o750), 'w') as f:
        json.dump(total_weight_map, f, indent=2)
        logger.info(f"Converted file param_name_map.json")

def get_unique_filename(save_file_path):
    root, ext = os.path.splitext(save_file_path)
    counter = 1
    while True:
        if not os.path.exists(save_file_path):
            return save_file_path
        save_file_path = f"{root}-{counter}{ext}"
        counter += 1

def save_prompt_completions_data(save_file_dir, global_step, **kwargs):
    """Save prompt completions data."""
    if get_rank() == 0:
        if not os.path.exists(save_file_dir):
            os.makedirs(save_file_dir)
        file_path = os.path.join(save_file_dir, f'prompt_completions_step_{global_step}.json')
        file_path = get_unique_filename(file_path)
        data_dict = []
        first_key = next(iter(kwargs))
        data_length = len(kwargs[first_key])

        for i in range(data_length):
            new_row = {}
            for key, values in kwargs.items():
                if values is None or len(values)==0:
                    new_row[key] = None
                else:
                    value = values[i]
                    if value is None:
                        new_row[key] = None
                    elif hasattr(value, 'item') and callable(getattr(value, 'item', None)):
                        new_row[key] = value.item()
                    else:
                        new_row[key] = value
            data_dict.append(new_row)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=4)
            logger.info(f"Saved data to {file_path}")
        except OSError as e:
            logger.warning(f"Save File Error: {e}")


def get_dp_rank(data_parallel):
    """ Get Data Parallel Rank ID """
    rank_id = get_rank()
    if reshard_optimizer.OPT_COMMUNICATION_GROUPS:
        for group in reshard_optimizer.OPT_COMMUNICATION_GROUPS["dp"]:
            if rank_id in group:
                dp_rank_id = group.index(rank_id)
                break

        if dp_rank_id is None:
            raise ValueError(f"Rank {rank_id} not found in any DP group: "
                             f"{reshard_optimizer.OPT_COMMUNICATION_GROUPS}")
    else:
        dp_rank_id = rank_id // (get_group_size() // data_parallel)
    return dp_rank_id

def add_metrics_to_tensorboard(tensor_writer, metrics, global_step):
    for key, value in metrics.items():
        tensor_writer.add_scalar(key, value, global_step=global_step)

def get_checkpoint_name(ckpt_path, prefix: str = "network", epoch_num: int = None, step_num: int = None,
                        formats: str = "ckpt"):
    """
    Get checkpoint file name of model.
    The layout of the ckpt_path will be like:
    ckpt_path/
    ├── rank_0
    │   ├── network_rank_0-0_1.ckpt
    │   └── network_rank_0-0_2.ckpt
    """
    rank = get_rank()
    # ensure ckpt path exist
    ckpt_path = os.path.normpath(os.path.abspath(ckpt_path))
    ckpt_local_path = os.path.join(ckpt_path, f"rank_{rank}")
    os.makedirs(ckpt_local_path, exist_ok=True)
    ckpt_file = os.path.join(ckpt_local_path, f"{prefix}_rank_{rank}-{epoch_num}_{step_num}.{formats}")
    return ckpt_file

def ensure_total_ckpt_is_less_than_limit(ckpt_path: str, limit: int = 5, formats: str = "ckpt"):
    """
    make sure the provided path contain less than limited number of checkpoint file
    Args:
        ckpt_path (str): Checkpoint file path.
        limit (int): limited number of checkpoint file. Default: 5
        formats (str): checkpoint format. Default: 'ckpt'
    """
    ckpt_list = [
        checkpoint
        for checkpoint in os.listdir(ckpt_path)
        if checkpoint.endswith(f'.{formats}')
    ]
    # ckpt_list: [oldest, ..., newest]
    ckpt_list = sorted(ckpt_list, key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)))
    ckpt_num = len(ckpt_list)
    if ckpt_num >= limit:
        for rm_ckpt_name in ckpt_list[: (ckpt_num - limit)]:
            logger.warning(f"Current checkpoint file exceed keep_checkpoint_max, "
                           f"removing {rm_ckpt_name}")
            rm_ckpt_path = os.path.join(ckpt_path, rm_ckpt_name)
            os.remove(rm_ckpt_path)

def load_param_to_net(net, param_dict):
    """load param to net."""
    param_not_load, ckpt_not_load = ms.load_param_into_net(net, param_dict)
    if param_not_load:
        logger.warning(f"When loading ckpt into the net, param_not_load:{param_not_load}")
    if ckpt_not_load:
        logger.warning(f"When loading ckpt into the net, ckpt_not_load:{ckpt_not_load}")

def record_last_ckpt_to_json(epoch: int, step: int, ckpt_file: str, meta_json: str):
    """record last ckpt info to json"""
    meta_data = {
        "last_epoch": epoch,
        "last_step": step,
        "last_ckpt_file": ckpt_file
    }
    flags = os.O_WRONLY | os.O_CREAT
    mode = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(meta_json, flags, mode), 'w', encoding="utf-8") as fp:
        json.dump(meta_data, fp)
