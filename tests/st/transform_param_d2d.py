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
"""transform param from device to device test case"""
import time
import numpy as np

import mindspore as ms
from mindspore.communication.management import init, get_rank
from mindspore.parallel.shard import Layout
from mindspore import nn, Tensor, Parameter, context, ops

from mindrlhf.utils import TransformParametersD2D

init()


class MatMulCell(nn.Cell):
    """MatMulCell"""
    def __init__(self, in_strategy, seed):
        super().__init__()
        np.random.seed(seed)
        self.matmul = ops.MatMul().shard(in_strategy=in_strategy)
        self.param = Parameter(Tensor(np.random.rand(8, 8), ms.float32))

    def construct(self, x):
        out = self.matmul(x, self.param)
        return out


class AddCell(nn.Cell):
    """AddCell"""
    def __init__(self, in_strategy, seed):
        super().__init__()
        np.random.seed(seed)
        self.add = ops.Add().shard(in_strategy=in_strategy)
        self.param = Parameter(Tensor(np.random.rand(8, 8), ms.float32))

    def construct(self, x):
        out = self.add(x, self.param)
        return out


class SrcNet(nn.Cell):
    """SrcNet"""
    def __init__(self, matmul_in_strategy, add_in_strategy, seed):
        super().__init__()
        self.matmul_src = MatMulCell(matmul_in_strategy, seed)
        self.add_src = AddCell(add_in_strategy, seed + 1)

    def construct(self, x):
        out = self.matmul_src(x)
        out = self.add_src(out)
        return out


class DstNet(nn.Cell):
    """DstNet"""
    def __init__(self, matmul_in_strategy, add_in_strategy, seed):
        super().__init__()
        self.matmul_dst = MatMulCell(matmul_in_strategy, seed)
        self.add_dst = AddCell(add_in_strategy, seed + 1)

    def construct(self, x):
        out = self.matmul_dst(x)
        out = self.add_dst(out)
        return out


class SrcNetPP(nn.Cell):
    """SrcNetPP"""
    def __init__(self, matmul_in_strategy, add_in_strategy, seed):
        super().__init__()
        self.matmul_src = MatMulCell(matmul_in_strategy, seed)
        self.add0_src = AddCell(add_in_strategy, seed + 1)
        self.add1_src = AddCell(add_in_strategy, seed + 2)
        self.add2_src = AddCell(add_in_strategy, seed + 3)

    def construct(self, x):
        out = self.matmul_src(x)
        out = self.add0_src(out)
        out = self.add1_src(out)
        out = self.add2_src(out)
        return out


class DstNetPP(nn.Cell):
    """DstNetPP"""
    def __init__(self, matmul_in_strategy, add_in_strategy, seed):
        super().__init__()
        self.matmul_dst = MatMulCell(matmul_in_strategy, seed)
        self.add0_dst = AddCell(add_in_strategy, seed + 1)
        self.add1_dst = AddCell(add_in_strategy, seed + 2)
        self.add2_dst = AddCell(add_in_strategy, seed + 3)

    def construct(self, x):
        out = self.matmul_dst(x)
        out = self.add0_dst(out)
        out = self.add1_dst(out)
        out = self.add2_dst(out)
        return out


def match_func(src_name, dst_name):
    src_name_split = src_name.split(".")[0]
    dst_name_split = dst_name.split(".")[0]
    if src_name_split.split("_")[0] == dst_name_split.split("_")[0]:
        return True
    return False


def test_transform_d2d_no_pp_1():
    """
    Feature: transform param no pp scenario
    Description: dpmp transform
    Expectation: Run success
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel", full_batch=True,
                                      strategy_ckpt_config={"save_file":
                                                                f"./no_pp_1_src/src_strategy_{get_rank()}.ckpt"})
    np.random.seed(10)
    x = Tensor(np.random.rand(8, 8), ms.float32)
    src_matmul_in_strategy = ((2, 4), (4, 1))
    src_add_in_strategy = ((8, 1), (8, 1))
    src_net = SrcNet(src_matmul_in_strategy, src_add_in_strategy, 5)
    src_out = src_net(x)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel", full_batch=True,
                                      strategy_ckpt_config={"save_file":
                                                                f"./no_pp_1_dst/dst_strategy_{get_rank()}.ckpt"})
    dst_matmul_in_strategy = ((1, 2), (2, 4))
    dst_add_in_strategy = ((4, 2), (4, 2))
    dst_net = DstNet(dst_matmul_in_strategy, dst_add_in_strategy, 6)
    dst_net(x)
    time.sleep(10)
    src_merged_stra = "./no_pp_1_src_merge/merged_strategy.ckpt"
    dst_merge_stra = "./no_pp_1_dst_merge/merged_strategy.ckpt"
    ms.merge_pipeline_strategys("./no_pp_1_src/", src_merged_stra)
    ms.merge_pipeline_strategys("./no_pp_1_dst/", dst_merge_stra)
    time.sleep(10)

    transform_param_d2d = TransformParametersD2D(src_net, dst_net, src_merged_stra, dst_merge_stra, match_func)
    transform_param_d2d.transform()
    dst_out = dst_net(x)
    context.reset_auto_parallel_context()
    assert np.allclose(src_out.asnumpy(), dst_out.asnumpy(), rtol=1e-4, atol=1e-4)


def test_transform_d2d_no_pp_2():
    """
    Feature: transform param no pp scenario
    Description: dpmp from zero to no zero transform
    Expectation: Run success
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel", full_batch=True,
                                      enable_parallel_optimizer=True,
                                      parallel_optimizer_config={"parallel_optimizer_threshold": 0},
                                      strategy_ckpt_config={"save_file":
                                                                f"./no_pp_2_src/src_strategy_{get_rank()}.ckpt"})
    np.random.seed(10)
    x = Tensor(np.random.rand(8, 8), ms.float32)
    src_matmul_in_strategy = ((2, 4), (4, 1))
    src_add_in_strategy = ((8, 1), (8, 1))
    src_net = SrcNet(src_matmul_in_strategy, src_add_in_strategy, 5)
    src_out = src_net(x)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel", full_batch=True,
                                      strategy_ckpt_config={"save_file":
                                                                f"./no_pp_2_dst/dst_strategy_{get_rank()}.ckpt"})
    dst_matmul_in_strategy = ((1, 2), (2, 4))
    dst_add_in_strategy = ((4, 2), (4, 2))
    dst_net = DstNet(dst_matmul_in_strategy, dst_add_in_strategy, 6)
    dst_net(x)
    time.sleep(10)
    src_merged_stra = "./no_pp_2_src_merge/merged_strategy.ckpt"
    dst_merge_stra = "./no_pp_2_dst_merge/merged_strategy.ckpt"
    ms.merge_pipeline_strategys("./no_pp_2_src/", src_merged_stra)
    ms.merge_pipeline_strategys("./no_pp_2_dst/", dst_merge_stra)
    time.sleep(10)

    transform_param_d2d = TransformParametersD2D(src_net, dst_net, src_merged_stra, dst_merge_stra, match_func)
    transform_param_d2d.transform()
    dst_out = dst_net(x)
    context.reset_auto_parallel_context()
    assert np.allclose(src_out.asnumpy(), dst_out.asnumpy(), rtol=1e-4, atol=1e-4)


def test_transform_d2d_no_pp_3():
    """
    Feature: transform param no pp scenario
    Description: dpmp from zero to zero transform
    Expectation: Run success
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel", full_batch=True,
                                      enable_parallel_optimizer=True,
                                      parallel_optimizer_config={"parallel_optimizer_threshold": 0},
                                      strategy_ckpt_config={"save_file":
                                                                f"./no_pp_3_src/src_strategy_{get_rank()}.ckpt"})
    np.random.seed(10)
    x = Tensor(np.random.rand(8, 8), ms.float32)
    src_matmul_in_strategy = ((2, 4), (4, 1))
    src_add_in_strategy = ((8, 1), (8, 1))
    src_net = SrcNet(src_matmul_in_strategy, src_add_in_strategy, 5)
    src_out = src_net(x)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel", full_batch=True,
                                      strategy_ckpt_config={"save_file":
                                                                f"./no_pp_3_dst/dst_strategy_{get_rank()}.ckpt"},
                                      enable_parallel_optimizer=True,
                                      parallel_optimizer_config={"parallel_optimizer_threshold": 0})
    dst_matmul_in_strategy = ((4, 1), (1, 2))
    dst_add_in_strategy = ((4, 2), (4, 2))
    dst_net = DstNet(dst_matmul_in_strategy, dst_add_in_strategy, 6)
    dst_net(x)
    time.sleep(10)
    src_merged_stra = "./no_pp_3_src_merge/merged_strategy.ckpt"
    dst_merge_stra = "./no_pp_3_dst_merge/merged_strategy.ckpt"
    ms.merge_pipeline_strategys("./no_pp_3_src/", src_merged_stra)
    ms.merge_pipeline_strategys("./no_pp_3_dst/", dst_merge_stra)
    time.sleep(10)

    transform_param_d2d = TransformParametersD2D(src_net, dst_net, src_merged_stra, dst_merge_stra, match_func)
    transform_param_d2d.transform()
    dst_out = dst_net(x)
    context.reset_auto_parallel_context()
    assert np.allclose(src_out.asnumpy(), dst_out.asnumpy(), rtol=1e-4, atol=1e-4)


def test_transform_d2d_no_pp_4():
    """
    Feature: transform param no pp scenario
    Description: dpmp transform
    Expectation: Run success
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel", full_batch=True,
                                      strategy_ckpt_config={"save_file":
                                                                f"./no_pp_4_src/src_strategy_{get_rank()}.ckpt"},
                                      parallel_optimizer_config={"parallel_optimizer_threshold": 0})
    np.random.seed(10)
    x = Tensor(np.random.rand(8, 8), ms.float32)
    src_matmul_in_strategy = ((2, 4), (4, 1))
    src_add_in_strategy = ((8, 1), (8, 1))
    src_net = SrcNet(src_matmul_in_strategy, src_add_in_strategy, 5)
    src_out = src_net(x)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel", full_batch=True,
                                      strategy_ckpt_config={"save_file":
                                                                f"./no_pp_4_dst/dst_strategy_{get_rank()}.ckpt"},
                                      enable_parallel_optimizer=True,
                                      parallel_optimizer_config={"parallel_optimizer_threshold": 0,
                                                                 "optimizer_weight_shard_size": 4})
    layout = Layout((2, 2, 2), ("dp", "mp", "sp"))
    dst_matmul_in_strategy = (layout("dp", "mp"), layout("mp", "None"))
    dst_add_in_strategy = ((4, 2), (4, 2))
    dst_net = DstNet(dst_matmul_in_strategy, dst_add_in_strategy, 6)
    dst_net(x)
    time.sleep(10)
    src_merged_stra = "./no_pp_4_src_merge/merged_strategy.ckpt"
    dst_merge_stra = "./no_pp_4_dst_merge/merged_strategy.ckpt"
    ms.merge_pipeline_strategys("./no_pp_4_src/", src_merged_stra)
    ms.merge_pipeline_strategys("./no_pp_4_dst/", dst_merge_stra)
    time.sleep(10)

    transform_param_d2d = TransformParametersD2D(src_net, dst_net, src_merged_stra, dst_merge_stra, match_func)
    transform_param_d2d.transform()
    dst_out = dst_net(x)
    context.reset_auto_parallel_context()
    assert np.allclose(src_out.asnumpy(), dst_out.asnumpy(), rtol=1e-4, atol=1e-4)


def test_transform_d2d_no_pp_5():
    """
    Feature: transform param no pp scenario (layout)
    Description: dpmp transform
    Expectation: Run success
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel", full_batch=True,
                                      strategy_ckpt_config={"save_file":
                                                                f"./no_pp_5_src/src_strategy_{get_rank()}.ckpt"},
                                      parallel_optimizer_config={"parallel_optimizer_threshold": 0})
    np.random.seed(10)
    x = Tensor(np.random.rand(8, 8), ms.float32)
    src_matmul_in_strategy = ((2, 4), (4, 1))
    src_add_in_strategy = ((8, 1), (8, 1))
    src_net = SrcNet(src_matmul_in_strategy, src_add_in_strategy, 5)
    src_out = src_net(x)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel", full_batch=True,
                                      strategy_ckpt_config={"save_file":
                                                                f"./no_pp_5_dst/dst_strategy_{get_rank()}.ckpt"},
                                      enable_parallel_optimizer=True,
                                      parallel_optimizer_config={"parallel_optimizer_threshold": 0,
                                                                 "optimizer_weight_shard_size": 4})
    layout = Layout((2, 2, 2), ("dp", "mp", "sp"))
    dst_matmul_in_strategy = (layout("dp", ("mp", "sp")), layout(("mp", "sp"), "None"))
    dst_add_in_strategy = ((4, 2), (4, 2))
    dst_net = DstNet(dst_matmul_in_strategy, dst_add_in_strategy, 6)
    dst_net(x)
    time.sleep(10)
    src_merged_stra = "./no_pp_5_src_merge/merged_strategy.ckpt"
    dst_merge_stra = "./no_pp_5_dst_merge/merged_strategy.ckpt"
    ms.merge_pipeline_strategys("./no_pp_5_src/", src_merged_stra)
    ms.merge_pipeline_strategys("./no_pp_5_dst/", dst_merge_stra)
    time.sleep(10)

    transform_param_d2d = TransformParametersD2D(src_net, dst_net, src_merged_stra, dst_merge_stra, match_func)
    transform_param_d2d.transform()
    dst_out = dst_net(x)
    context.reset_auto_parallel_context()
    assert np.allclose(src_out.asnumpy(), dst_out.asnumpy(), rtol=1e-4, atol=1e-4)


def test_transform_d2d_pp_1():
    """
    Feature: transform param pp scenario
    Description: dpmp transform
    Expectation: Run success
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel", full_batch=True,
                                      strategy_ckpt_config={"save_file":
                                                                f"./pp_1_src/src_strategy_{get_rank()}.ckpt"},
                                      pipeline_stages=2)
    np.random.seed(10)
    x = Tensor(np.random.rand(8, 8), ms.float32)
    src_matmul_in_strategy = ((2, 2), (2, 1))
    src_add_in_strategy = ((4, 1), (4, 1))
    src_net = SrcNetPP(src_matmul_in_strategy, src_add_in_strategy, 5)
    src_net.matmul_src.pipeline_stage = 0
    src_net.add0_src.pipeline_stage = 0
    src_net.add1_src.pipeline_stage = 1
    src_net.add2_src.pipeline_stage = 1
    src_out = src_net(x)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel", full_batch=True,
                                      strategy_ckpt_config={"save_file":
                                                                f"./pp_1_dst/dst_strategy_{get_rank()}.ckpt"},
                                      enable_parallel_optimizer=True,
                                      parallel_optimizer_config={"parallel_optimizer_threshold": 0,
                                                                 "optimizer_weight_shard_size": 2})
    layout = Layout((2, 2, 2), ("dp", "mp", "sp"))
    dst_matmul_in_strategy = (layout("dp", "mp"), layout("mp", "None"))
    dst_add_in_strategy = ((4, 2), (4, 2))
    dst_net = DstNetPP(dst_matmul_in_strategy, dst_add_in_strategy, 6)
    dst_net(x)
    time.sleep(10)
    src_merged_stra = "./pp_1_src_merge/merged_strategy.ckpt"
    dst_merge_stra = "./pp_1_dst_merge/merged_strategy.ckpt"
    ms.merge_pipeline_strategys("./pp_1_src/", src_merged_stra)
    ms.merge_pipeline_strategys("./pp_1_dst/", dst_merge_stra)
    time.sleep(10)

    transform_param_d2d = TransformParametersD2D(src_net, dst_net, src_merged_stra, dst_merge_stra, match_func)
    transform_param_d2d.transform()
    dst_out = dst_net(x)
    context.reset_auto_parallel_context()
    if get_rank() in [4, 5, 6, 7]:
        assert np.allclose(src_out.asnumpy(), dst_out.asnumpy(), rtol=1e-4, atol=1e-4)


def test_transform_d2d_pp_2():
    """
    Feature: transform param no pp scenario
    Description: dpmp transform
    Expectation: Run success
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel", full_batch=True,
                                      strategy_ckpt_config={"save_file":
                                                                f"./pp_2_src/src_strategy_{get_rank()}.ckpt"})
    np.random.seed(10)
    x = Tensor(np.random.rand(8, 8), ms.float32)
    src_matmul_in_strategy = ((4, 2), (2, 1))
    src_add_in_strategy = ((8, 1), (8, 1))
    src_net = SrcNetPP(src_matmul_in_strategy, src_add_in_strategy, 5)
    src_out = src_net(x)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel", full_batch=True,
                                      strategy_ckpt_config={"save_file":
                                                                f"./pp_2_dst/dst_strategy_{get_rank()}.ckpt"},
                                      pipeline_stages=2)
    layout = Layout((2, 2), ("dp", "mp"))
    dst_matmul_in_strategy = (layout("dp", "mp"), layout("mp", "None"))
    dst_add_in_strategy = ((2, 2), (2, 2))
    dst_net = DstNetPP(dst_matmul_in_strategy, dst_add_in_strategy, 6)
    dst_net.matmul_dst.pipeline_stage = 0
    dst_net.add0_dst.pipeline_stage = 0
    dst_net.add1_dst.pipeline_stage = 1
    dst_net.add2_dst.pipeline_stage = 1
    dst_net(x)
    time.sleep(10)
    src_merged_stra = "./pp_2_src_merge/merged_strategy.ckpt"
    dst_merge_stra = "./pp_2_dst_merge/merged_strategy.ckpt"
    ms.merge_pipeline_strategys("./pp_2_src/", src_merged_stra)
    ms.merge_pipeline_strategys("./pp_2_dst/", dst_merge_stra)
    time.sleep(10)

    transform_param_d2d = TransformParametersD2D(src_net, dst_net, src_merged_stra, dst_merge_stra, match_func)
    transform_param_d2d.transform()
    dst_out = dst_net(x)
    context.reset_auto_parallel_context()
    if get_rank() in [4, 5, 6, 7]:
        assert np.allclose(src_out.asnumpy(), dst_out.asnumpy(), rtol=1e-4, atol=1e-4)


def test_transform_d2d_pp_3():
    """
    Feature: transform param no pp scenario
    Description: dpmp transform
    Expectation: Run success
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel", full_batch=True,
                                      strategy_ckpt_config={"save_file":
                                                                f"./pp_3_src/src_strategy_{get_rank()}.ckpt"},
                                      pipeline_stages=2)
    np.random.seed(10)
    x = Tensor(np.random.rand(8, 8), ms.float32)
    src_matmul_in_strategy = ((2, 2), (2, 1))
    src_add_in_strategy = ((4, 1), (4, 1))
    src_net = SrcNetPP(src_matmul_in_strategy, src_add_in_strategy, 5)
    src_net.matmul_src.pipeline_stage = 0
    src_net.add0_src.pipeline_stage = 0
    src_net.add1_src.pipeline_stage = 1
    src_net.add2_src.pipeline_stage = 1
    src_out = src_net(x)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel", full_batch=True,
                                      strategy_ckpt_config={"save_file":
                                                                f"./pp_3_dst/dst_strategy_{get_rank()}.ckpt"},
                                      enable_parallel_optimizer=True,
                                      pipeline_stages=2)
    layout = Layout((2, 2), ("dp", "mp"))
    dst_matmul_in_strategy = (layout("dp", "mp"), layout("mp", "None"))
    dst_add_in_strategy = ((2, 2), (2, 2))
    dst_net = DstNetPP(dst_matmul_in_strategy, dst_add_in_strategy, 6)
    dst_net.matmul_dst.pipeline_stage = 0
    dst_net.add0_dst.pipeline_stage = 0
    dst_net.add1_dst.pipeline_stage = 1
    dst_net.add2_dst.pipeline_stage = 1
    dst_net(x)
    time.sleep(10)
    src_merged_stra = "./pp_3_src_merge/merged_strategy.ckpt"
    dst_merge_stra = "./pp_3_dst_merge/merged_strategy.ckpt"
    ms.merge_pipeline_strategys("./pp_3_src/", src_merged_stra)
    ms.merge_pipeline_strategys("./pp_3_dst/", dst_merge_stra)
    time.sleep(10)

    transform_param_d2d = TransformParametersD2D(src_net, dst_net, src_merged_stra, dst_merge_stra, match_func)
    transform_param_d2d.transform()
    dst_out = dst_net(x)
    context.reset_auto_parallel_context()
    if get_rank() in [4, 5, 6, 7]:
        assert np.allclose(src_out.asnumpy(), dst_out.asnumpy(), rtol=1e-4, atol=1e-4)


def test_transform_d2d_pp_4():
    """
    Feature: transform param no pp scenario
    Description: dpmp transform
    Expectation: Run success
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel", full_batch=True,
                                      strategy_ckpt_config={"save_file":
                                                                f"./pp_4_src/src_strategy_{get_rank()}.ckpt"},
                                      pipeline_stages=4)
    np.random.seed(10)
    x = Tensor(np.random.rand(8, 8), ms.float32)
    src_matmul_in_strategy = ((2, 1), (1, 1))
    src_add_in_strategy = ((2, 1), (2, 1))
    src_net = SrcNetPP(src_matmul_in_strategy, src_add_in_strategy, 5)
    src_net.matmul_src.pipeline_stage = 0
    src_net.add0_src.pipeline_stage = 1
    src_net.add1_src.pipeline_stage = 2
    src_net.add2_src.pipeline_stage = 3
    src_out = src_net(x)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel", full_batch=True,
                                      strategy_ckpt_config={"save_file":
                                                                f"./pp_4_dst/dst_strategy_{get_rank()}.ckpt"},
                                      enable_parallel_optimizer=True,
                                      pipeline_stages=2)
    layout = Layout((2, 2), ("dp", "mp"))
    dst_matmul_in_strategy = (layout("dp", "mp"), layout("mp", "None"))
    dst_add_in_strategy = ((2, 2), (2, 2))
    dst_net = DstNetPP(dst_matmul_in_strategy, dst_add_in_strategy, 6)
    dst_net.matmul_dst.pipeline_stage = 0
    dst_net.add0_dst.pipeline_stage = 0
    dst_net.add1_dst.pipeline_stage = 1
    dst_net.add2_dst.pipeline_stage = 1
    dst_net(x)
    time.sleep(10)
    src_merged_stra = "./pp_4_src_merge/merged_strategy.ckpt"
    dst_merge_stra = "./pp_4_dst_merge/merged_strategy.ckpt"
    ms.merge_pipeline_strategys("./pp_4_src/", src_merged_stra)
    ms.merge_pipeline_strategys("./pp_4_dst/", dst_merge_stra)
    time.sleep(10)

    transform_param_d2d = TransformParametersD2D(src_net, dst_net, src_merged_stra, dst_merge_stra, match_func)
    transform_param_d2d.transform()
    dst_out = dst_net(x)
    context.reset_auto_parallel_context()
    if get_rank() in [6, 7]:
        assert np.allclose(src_out.asnumpy(), dst_out.asnumpy(), rtol=1e-4, atol=1e-4)
