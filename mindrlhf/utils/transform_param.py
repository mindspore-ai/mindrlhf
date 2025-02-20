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
"""Resharding Weight"""
import numpy as np

from mindspore import context
from mindspore.nn import Cell
import mindspore.log as logger
from mindspore.communication import get_rank, get_group_size, create_group
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common import dtype as msdtype
from mindspore.parallel.shard import Layout
from mindspore.parallel._parallel_serialization import _build_searched_strategy, _convert_to_list
from mindspore.parallel._auto_parallel_context import _get_auto_parallel_context_func_map, \
    _set_auto_parallel_context_func_map


def _get_used_dev_mat(dev_mat, tensormap):
    used_dev_mat = []
    for i in range(len(dev_mat)):
        idx = len(dev_mat) - i - 1
        used_dev_mat.append(idx in tensormap)
    return used_dev_mat


# pylint: disable=R1705
def _get_dev_mat_for_opt_shard(opt_shard, dev_mat, tensor_map):
    """generate device matrix for opt shard scenario"""
    if opt_shard == 0:
        return dev_mat, -1, tensor_map, True
    used_dev_num = int(np.prod(np.array([dev_mat[len(dev_mat) - i - 1] if i != -1 else 1 for i in tensor_map])))
    total_dev_num = int(np.prod(np.array(dev_mat)))
    if opt_shard == -1 or used_dev_num * opt_shard == total_dev_num:
        # 饱和切分不需要调整dev mat
        return dev_mat, -1, tensor_map, True
    # 非饱和根据场景调整dev mat
    remain_dev_num = total_dev_num // (used_dev_num * opt_shard)
    real_remain_dev_num = 1
    used_dev_mat_mask = _get_used_dev_mat(dev_mat, tensor_map)
    counter = -1
    for i, value in enumerate(dev_mat):
        if used_dev_mat_mask[i]:
            # 当前维度已经被使用，不被计算在有优化器并行中
            continue
        if real_remain_dev_num == remain_dev_num:
            # 现有dev mat已经满足需求，不需要调整
            # 通过counter值修改tensor map
            return dev_mat, counter, tensor_map, False
        elif real_remain_dev_num < remain_dev_num:
            real_remain_dev_num *= value
        else:
            # 对counter位置的维度进行拆分
            splitted_dev_value = dev_mat[counter]
            new_dev_mat_value_first = remain_dev_num // (real_remain_dev_num // splitted_dev_value)
            new_dev_mat_value_second = splitted_dev_value // new_dev_mat_value_first
            new_dev_mat = dev_mat[:counter] + \
                [new_dev_mat_value_first, new_dev_mat_value_second] + dev_mat[counter + 1:]
            new_tensor_map = [value if value < len(dev_mat) - counter - 1 else value + 1 for value in tensor_map]
            return new_dev_mat, counter, new_tensor_map, False
        counter += 1
    if real_remain_dev_num == remain_dev_num:
        # 现有dev mat已经满足需求，不需要调整
        # 通过counter值修改tensor map
        return dev_mat, counter, tensor_map, False
    else:
        # 对counter位置的维度进行拆分
        splitted_dev_value = dev_mat[counter]
        new_dev_mat_value_first = remain_dev_num // (real_remain_dev_num // splitted_dev_value)
        new_dev_mat_value_second = splitted_dev_value // new_dev_mat_value_first
        new_dev_mat = dev_mat[:counter] + \
            [new_dev_mat_value_first, new_dev_mat_value_second] + dev_mat[counter + 1:]
        new_tensor_map = [value if value < len(dev_mat) - counter - 1 else value + 1 for value in tensor_map]
        return new_dev_mat, counter, new_tensor_map, False


def _get_tensor_map_for_opt_shard(dev_mat, tensor_map, full_opt_shard, counter, alias_name):
    """generate tensor map for opt shard scenario"""
    used_dev_mat = _get_used_dev_mat(dev_mat, tensor_map)
    if full_opt_shard:
        unused_idx = [len(used_dev_mat) - i - 1 for i in range(len(used_dev_mat)) if not used_dev_mat[i]]
    else:
        unused_idx = []
        real_idx = 0
        for i in range(len(used_dev_mat)):
            if used_dev_mat[i]:
                continue
            if real_idx > counter:
                unused_idx.append(len(used_dev_mat) - i - 1)
            if not used_dev_mat[i]:
                real_idx += 1

    opt_shard_dim = unused_idx
    if tensor_map[0] != -1:
        opt_shard_dim = [tensor_map[0]] + opt_shard_dim
    opt_shard_name = tuple([alias_name[len(alias_name) - i - 1] if i != -1 else "None" for i in opt_shard_dim])
    og_name = [alias_name[len(alias_name) - i - 1] if i != -1 else "None" for i in tensor_map[1:]]
    new_tensor_map = tuple([opt_shard_name] + og_name)
    return new_tensor_map


def _layout_process(input_stra_layout_list):
    """return the layout list."""
    output_layout_list = []
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    for i, stra_layout in enumerate(input_stra_layout_list):
        opt_shard = stra_layout[5]
        dev_mat = stra_layout[0]
        tensor_map = stra_layout[1]
        new_dev_mat, counter, new_tensor_map, full_opt_shard = _get_dev_mat_for_opt_shard(opt_shard, dev_mat,
                                                                                          tensor_map)
        alias_name = [alphabet[i] for i in range(len(new_dev_mat))]
        if opt_shard != 0:
            # 开了优化器并行，需要修改new_tensor_map
            new_tensor_map = _get_tensor_map_for_opt_shard(new_dev_mat, new_tensor_map, full_opt_shard, counter,
                                                           alias_name)
        else:
            new_tensor_map = tuple([alias_name[len(alias_name) - i - 1] if i != -1 else "None" for i in new_tensor_map])
        all_dev_num = get_group_size()
        local_dev_num = int(np.prod(np.array(new_dev_mat)))
        if all_dev_num != local_dev_num:
            # 其中一个网络开启了pp，导致实际设备数和计算出的设备数不一致
            remain = all_dev_num // local_dev_num
            new_dev_mat.insert(0, remain)
            alias_name.insert(0, 'pp_remain_dev')
        new_dev_mat = tuple(new_dev_mat)
        alias_name = tuple(alias_name)
        layout = Layout(new_dev_mat, alias_name)
        final_layout = layout(*new_tensor_map)
        logger.debug("The final layout is ", final_layout.to_dict())
        output_layout_list.append(final_layout)
    return output_layout_list


def _ops_process(input_layout_list, ops_type):
    """return the ops list, which contains the assign or identity ops."""
    output_ops_list = []
    for final_layout in input_layout_list:
        if ops_type == "assign":
            ops = P.Assign().shard(in_strategy=(final_layout, final_layout), out_strategy=(final_layout,))
        elif ops_type == "identity":
            ops = P.Identity().shard(in_strategy=(final_layout,), out_strategy=(final_layout,))
        else:
            raise ValueError("The ops_type should be assign or identity.")
        ops.add_prim_attr("self_define_shard", True)
        output_ops_list.append(ops)
    return output_ops_list


class _AllReduceCell(Cell):
    """AllReduce param from different stages in PP scenario"""
    def __init__(self, src_param_info_list):
        super().__init__()
        self.allreduce_list = []
        self.param_list = []
        self.has_data_in_current_stage_list = []
        self.param_shape = []
        self.zeros = P.Zeros()
        for param_info in src_param_info_list:
            if param_info.need_all_reduce:
                self.has_data_in_current_stage_list.append(param_info.has_data_in_current_stage)
                self.allreduce_list.append(param_info.get_all_reduce_op())
                self.param_list.append(param_info.get_param_data())
                og_shape = param_info.get_param_data().shape
                if param_info.has_data_in_current_stage:
                    self.param_shape.append(og_shape)
                else:
                    stra = param_info.get_param_strategy()
                    self.param_shape.append(tuple([og_shape[i] // stra[i] for i in range(len(stra))]))

    def construct(self):
        """run allreduce to obtain the data from different stages"""
        output_param_list = []
        for i in range(len(self.allreduce_list)):
            if self.has_data_in_current_stage_list[i]:
                all_reduce_value = self.param_list[i]
                output_param_list.append(self.allreduce_list[i](all_reduce_value))
            else:
                all_reduce_value = self.zeros(self.param_shape[i], msdtype.float32)
                src_param = self.allreduce_list[i](all_reduce_value)
                output_param_list.append(src_param)
        return output_param_list


class _ParameterInfo:
    """Parameter info, it stores the param name, param value, src/dst info and from/to pp stages"""
    def __init__(self, param_name, param_data, src_stra_info, dst_stra_info, from_pp_stages, to_pp_stages):
        self._from_dev_num_in_stage = get_group_size() // from_pp_stages
        to_dev_num_in_stage = get_group_size() // to_pp_stages
        self._current_rank_id = get_rank()
        self._param_name = param_name
        self._param_data = param_data
        from_stage_id = src_stra_info[6][0]
        to_stage_id = dst_stra_info[6][0]
        self._src_stra_info = src_stra_info
        self._dst_stra_info = dst_stra_info
        self._from_rank_id = list(
            range(self._from_dev_num_in_stage * from_stage_id, self._from_dev_num_in_stage * (from_stage_id + 1)))
        self._to_rank_id = list(range(to_dev_num_in_stage * to_stage_id, to_dev_num_in_stage * (to_stage_id + 1)))
        self._need_transform = False
        self._need_all_reduce = False
        self._all_reduce_group = None
        self._strategy = None
        self._has_data_in_current_stage = self._current_rank_id in self._from_rank_id
        self._has_data_in_current_stage_dst = self._current_rank_id in self._to_rank_id
        self._diff_rank_id = [rank_id for rank_id in self._to_rank_id if rank_id not in self._from_rank_id]
        if self._has_data_in_current_stage or self._current_rank_id in self._diff_rank_id:
            if self._current_rank_id in self._to_rank_id:
                self._need_transform = True
            self._create_all_reduce_group()
        logger.debug(f"Parameter info is {self.to_dict()}")

    def get_all_reduce_op(self):
        """get all reduce op"""
        return P.AllReduce(group=self._all_reduce_group)

    def get_param_data(self):
        """get parameter value"""
        return self._param_data

    def get_param_strategy(self):
        """convert devmat and tensormap to strategy"""
        if self._strategy is None:
            dev_mat = self._src_stra_info[0]
            tensor_map = self._src_stra_info[1]
            self._strategy = [dev_mat[len(dev_mat) - i - 1] if i != -1 else 1 for i in tensor_map]
            opt_shard = self._src_stra_info[5]
            if opt_shard != 0:
                if opt_shard != -1:
                    self._strategy[0] = self._strategy[0] * opt_shard
                else:
                    remain_dev_mat = [dev_mat[dev_mat_idx] for dev_mat_idx in dev_mat if len(dev_mat) - dev_mat_idx - 1]
                    remain_prod_dev = np.prod(np.array(remain_dev_mat))
                    self._strategy[0] = self._strategy[0] * remain_prod_dev
        return tuple(self._strategy)

    def to_dict(self):
        return {"param_name": self._param_name, "need_transform": self._need_transform,
                "need_all_reduce": self._need_all_reduce,
                "diff_rank_id": self._diff_rank_id}

    @property
    def need_transform(self):
        return self._need_transform

    @property
    def need_all_reduce(self):
        return self._need_all_reduce

    @property
    def param_name(self):
        return self._param_name

    @property
    def has_data_in_current_stage(self):
        return self._has_data_in_current_stage

    @property
    def has_data_in_current_stage_dst(self):
        return self._has_data_in_current_stage_dst

    @property
    def src_stra_info(self):
        return self._src_stra_info

    @property
    def dst_stra_info(self):
        return self._dst_stra_info

    def _create_all_reduce_group(self):
        """create all reduce group"""
        if len(self._to_rank_id) > len(self._from_rank_id):
            self._need_all_reduce = True
            current_rank_stage_id = self._current_rank_id // self._from_dev_num_in_stage
            end_stage = self._from_dev_num_in_stage * (current_rank_stage_id + 1)
            rank_pos_in_stage = [rank_id for rank_id in range(self._from_dev_num_in_stage * current_rank_stage_id,
                                                              end_stage)].index(self._current_rank_id)
            all_reduce_rank_list = [self._from_rank_id[rank_pos_in_stage]]
            while rank_pos_in_stage < len(self._diff_rank_id):
                all_reduce_rank_list.append(self._diff_rank_id[rank_pos_in_stage])
                rank_pos_in_stage += self._from_dev_num_in_stage
            all_reduce_rank_list.sort()
            logger.debug(f"For param {self._param_name}, its group is {all_reduce_rank_list}.")
            str_rank_list = '-'.join([str(rank) for rank in all_reduce_rank_list])
            self._all_reduce_group = f"{self._param_name}_group-{str_rank_list}"
            create_group(self._all_reduce_group, all_reduce_rank_list)


class _ReshardParameters(Cell):
    """reshard parameter class"""
    def __init__(self, src_param_name_intersection, dst_param_name_intersection):
        super().__init__()
        src_stra_layout_list = []
        self.src_net_param_list = []
        dst_stra_layout_list = []
        self.dst_net_param_list = []
        for i, src_param_info in enumerate(src_param_name_intersection):
            src_param_name = src_param_info.param_name
            src_layout = src_param_info.src_stra_info
            if src_layout is None:
                raise ValueError(f"Can not find the layout for {src_param_name} in src_network.")
            src_stra_layout_list.append(src_layout)
            self.src_net_param_list.append(src_param_info.get_param_data())
            dst_param_info = dst_param_name_intersection[i]
            dst_param_name = dst_param_info.param_name
            dst_layout = dst_param_info.dst_stra_info
            if dst_layout is None:
                raise ValueError(f"Can not find the layout for {dst_param_name} in dst_network.")
            dst_stra_layout_list.append(dst_layout)
            self.dst_net_param_list.append(dst_param_info.get_param_data())

        from_layout_list = _layout_process(src_stra_layout_list)
        to_layout_list = _layout_process(dst_stra_layout_list)
        self.from_identity_list = _ops_process(from_layout_list, "identity")
        self.to_identity_list = _ops_process(to_layout_list, "identity")
        self.to_assign_list = _ops_process(to_layout_list, "assign")

    def construct(self):
        bool_value = True
        for i in range(len(self.from_identity_list)):
            src_param_identity = self.from_identity_list[i](self.src_net_param_list[i])
            src_param_identity = self.to_identity_list[i](src_param_identity)
            src_param_identity = ops.cast(src_param_identity, self.dst_net_param_list[i].dtype)
            bool_value = F.depend(bool_value, self.to_assign_list[i](self.dst_net_param_list[i], src_param_identity))
        return bool_value


class _ReshardParametersPP(Cell):
    """reshard parameter class"""
    def __init__(self, src_param_name_intersection, dst_param_name_intersection):
        super().__init__()

        src_stra_layout_list = []
        self.src_net_param_list = []
        dst_stra_layout_list = []
        self.dst_net_param_list = []
        self.virtual_dataset_layout_flag = []
        for i, src_param_info in enumerate(src_param_name_intersection):
            # 当前卡上的数据只有在dst的rank中有时，才需要倒换
            # 也有可能是当前卡并不需要倒换，但是需要allreduce
            if src_param_info.need_transform or src_param_info.need_all_reduce:
                src_param_name = src_param_info.param_name
                src_layout = src_param_info.src_stra_info
                if src_layout is None:
                    raise ValueError(f"Can not find the layout for {src_param_name} in src_network.")
                src_stra_layout_list.append(src_layout)
                self.src_net_param_list.append(src_param_info.get_param_data())
                # 如果param在当前stage没有数据，则需要通过allreduce获取数据。
                if not src_param_info.has_data_in_current_stage:
                    self.virtual_dataset_layout_flag.append(True)
                else:
                    self.virtual_dataset_layout_flag.append(False)
            dst_param_info = dst_param_name_intersection[i]
            dst_param_name = dst_param_info.param_name
            dst_layout = dst_param_info.dst_stra_info
            if dst_layout is None:
                raise ValueError(f"Can not find the layout for {dst_param_name} in dst_network.")
            if dst_param_info.has_data_in_current_stage_dst:
                dst_stra_layout_list.append(dst_layout)
                self.dst_net_param_list.append(dst_param_info.get_param_data())

        from_layout_list = _layout_process(src_stra_layout_list)
        to_layout_list = _layout_process(dst_stra_layout_list)
        self.from_identity_list = _ops_process(from_layout_list, "identity")
        self.from_assign_list = _ops_process(from_layout_list, "assign")
        self.to_identity_list = _ops_process(to_layout_list, "identity")
        self.to_assign_list = _ops_process(to_layout_list, "assign")
        self.dataset_layout = tuple([from_layout_list[i] for i in range(len(from_layout_list))])

    def construct(self, *param_list):
        bool_value = True
        for i in range(len(self.from_identity_list)):
            src_param_identity = self.to_identity_list[i](param_list[i])
            src_param_identity = ops.cast(src_param_identity, self.dst_net_param_list[i].dtype)
            bool_value = F.depend(bool_value, self.to_assign_list[i](self.dst_net_param_list[i], src_param_identity))
        return bool_value


class TransformParametersD2D:
    """
    Transform parameter from source network's layout to destination network's layout. All the parameters will do
    transformation from device to device.

    Args:
         src_network (Cell): The network provides the source parameters.
         dst_network (Cell): The parameters in destination network will be assigned by source network
            after transformation.
        src_strategy_path (str): The path of source merged strategy end with ckpt.
        dst_strategy_path (str): The path of destination merged strategy end with ckpt.
        match_func (function): Check whether two input parameters are matched. It takes source param and dest param
            as input and return boolean. Default value is None, which means that the parameters name must equal.
            Default: None

    Raises:
        TypeError: The output of `match_func` is not a boolean.
        ValueError: `src_network` and `dst_network` does not have matched parameter.
        ValueError: `src_strategy_path` or `dst_strategy_path` is None.
    """

    def __init__(self, src_network, dst_network, src_strategy_path, dst_strategy_path, match_func=None):
        if src_strategy_path is not None:
            src_strategy = _build_searched_strategy(src_strategy_path)
            src_stra_info = _convert_to_list(src_strategy, get_rank())
        else:
            raise ValueError("The src_strategy_path should not be None.")
        if dst_strategy_path is not None:
            dst_strategy = _build_searched_strategy(dst_strategy_path)
            dst_stra_info = _convert_to_list(dst_strategy, get_rank())
        else:
            raise ValueError("The dst_strategy_path should not be None.")

        # 获取src和dst的pp stage个数
        src_pp_stages = max(stra_info[6][0] for stra_info in src_stra_info.values()) + 1
        dst_pp_stages = max(stra_info[6][0] for stra_info in dst_stra_info.values()) + 1

        # 检查src和dst是否是同总卡数倒换
        self._check_total_dev_num(src_stra_info, src_pp_stages, dst_stra_info, dst_pp_stages)

        # 将存在strategy里面的param给存到dict中，不存在strategy.ckpt中的param不做倒换
        src_net_param_dict = {}
        for name, param in src_network.parameters_and_names():
            if name in src_stra_info.keys():
                src_net_param_dict[name] = param
        dst_net_param_dict = {}
        for name, param in dst_network.parameters_and_names():
            if name in dst_stra_info.keys():
                dst_net_param_dict[name] = param

        # 获取src和dst的param的交集
        src_param_name_intersection = []
        dst_param_name_intersection = []
        for src_name, src_value in src_net_param_dict.items():
            for dst_name, dst_value in dst_net_param_dict.items():
                # 如果有match_func则通过match_func来判断两个网络中的param是否匹配
                if match_func is not None:
                    match_flag = match_func(src_name, dst_name)
                    if not isinstance(match_flag, bool):
                        raise TypeError("The return value of match_func should be bool.")
                    if match_flag:
                        src_param_info = _ParameterInfo(src_name, src_value, src_stra_info[src_name],
                                                        dst_stra_info[dst_name], src_pp_stages, dst_pp_stages)
                        dst_param_info = _ParameterInfo(dst_name, dst_value, src_stra_info[src_name],
                                                        dst_stra_info[dst_name], src_pp_stages, dst_pp_stages)
                        src_param_name_intersection.append(src_param_info)
                        dst_param_name_intersection.append(dst_param_info)
                        break
                # 如果match_func为None，则需要src_name和dst_name完全匹配
                else:
                    if src_name == dst_name:
                        src_param_info = _ParameterInfo(src_name, src_value, src_stra_info[src_name],
                                                        dst_stra_info[dst_name], src_pp_stages, dst_pp_stages)
                        dst_param_info = _ParameterInfo(dst_name, dst_value, src_stra_info[src_name],
                                                        dst_stra_info[dst_name], src_pp_stages, dst_pp_stages)
                        src_param_name_intersection.append(src_param_info)
                        dst_param_name_intersection.append(dst_param_info)
                        break
        self._log_not_in_intersection_param(src_param_name_intersection, src_net_param_dict, "src_network")
        self._log_not_in_intersection_param(dst_param_name_intersection, dst_net_param_dict, "dst_network")
        if not src_param_name_intersection or not dst_param_name_intersection:
            raise ValueError("The intersection of src_network and dst_network is empty.")
        if len(src_param_name_intersection) != len(dst_param_name_intersection):
            raise ValueError("The length of src_param_name_intersection and dst_param_name_intersection is not equal.")

        self.pp_trans = False
        # 如果src_pp_stages小于等于dst_pp_stages，则dst的各src的各pp stage上的参数可以直接倒换到dst对应的卡上
        # rank0 01 11           01
        # rank1 02 12           02
        #  pp1   ------>  pp2
        # rank2 03 13           11
        # rank3 04 14           12
        if src_pp_stages <= dst_pp_stages:
            self._reshard_parameters = _ReshardParameters(src_param_name_intersection, dst_param_name_intersection)
        # 如果src_pp_stages大于dst_pp_stages，则需要通过AllReduce把对应卡的权重给通信过来才能倒换
        # 需要把src rank0的01给通信到src的rank2上，让rank2持有param0的数据。同理rank1的02给通信到rank3上
        # rank0 01           01 11
        # rank1 02           02 12
        # pp2 ------->  pp1
        # rank2 11           03 13
        # rank3 12           04 14
        else:
            self.pp_trans = True
            self._reshard_parameters_pp = _ReshardParametersPP(src_param_name_intersection, dst_param_name_intersection)
            self.dataset_layout = self._reshard_parameters_pp.dataset_layout
            self.all_reduce_cell = _AllReduceCell(src_param_name_intersection)
        self._auto_paralell_context_value_map = {}
        self._pipeline_config = {}

    def transform(self):
        """transform the parameters from source network layout to dest network layout and assign the parameter to
        dest network"""
        self._get_auto_parallel_context()
        if self.pp_trans:
            output_param_list = self.all_reduce_cell()
            context.set_auto_parallel_context(device_num=get_group_size(),
                                              parallel_mode="semi_auto_parallel",
                                              dataset_strategy=self.dataset_layout)
            self._reshard_parameters_pp(*output_param_list)
        else:
            context.set_auto_parallel_context(device_num=get_group_size(),
                                              parallel_mode="semi_auto_parallel")
            self._reshard_parameters()
        self._set_auto_parallel_context()

    def _get_auto_parallel_context(self):
        """get auto parallel context before reset"""
        from mindspore.context import reset_auto_parallel_context
        for key, value in _get_auto_parallel_context_func_map.items():
            if key == "pipeline_interleave":
                self._pipeline_config[key] = value()
            elif key == "pipeline_scheduler":
                self._pipeline_config[key] = value()
            else:
                self._auto_paralell_context_value_map[key] = value()
        # reset auto parallel context to prevent context influence transform
        reset_auto_parallel_context()

    def _set_auto_parallel_context(self):
        """set auto parallel context after transformation"""
        # set the same auto parallel context after transform
        from mindspore.context import reset_auto_parallel_context
        reset_auto_parallel_context()
        for key, value in self._auto_paralell_context_value_map.items():
            # list is empty or full_batch_is_set is not needed to set
            if (isinstance(value, list) and not value) or (key == "full_batch_is_set"):
                continue
            _set_auto_parallel_context_func_map[key](value)
        _set_auto_parallel_context_func_map["pipeline_config"](self._pipeline_config)

    def _log_not_in_intersection_param(self, param_intersection, all_param, debug_string):
        """find the param in all_param but not in param_intersection"""
        param_intersection_name = [
            param.param_name for param in param_intersection]
        for param_name in all_param.keys():
            if param_name not in param_intersection_name:
                logger.warning(f"The param {param_name} of {debug_string} is not in the intersection of src_network "
                               f"and dst_network.")

    def _check_total_dev_num(self, src_stra_info, src_pp_stages, dst_stra_info, dst_pp_stages):
        _, first_src_stra_info = next(iter(src_stra_info.items()))
        src_dev_num = np.prod(np.array(first_src_stra_info[0])) * src_pp_stages
        _, first_dst_stra_info = next(iter(dst_stra_info.items()))
        dst_dev_num = np.prod(np.array(first_dst_stra_info[0])) * dst_pp_stages
        if src_dev_num != dst_dev_num:
            raise ValueError(f"src network device number must equal to dest network device number,"
                             f"but got src {src_dev_num}, dst {dst_dev_num}")