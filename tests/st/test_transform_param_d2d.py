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
import os


def test_transform_d2d_no_pp_1():
    """
    Feature: transform param no pp scenario
    Description: dpmp transform
    Expectation: Run success
    """
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "
        "--master_port=10801 --join=True --log_dir=./transform_d2d_no_pp_1 "
        "pytest -s transform_param_d2d.py::test_transform_d2d_no_pp_1 "
    )
    assert return_code == 0


def test_transform_d2d_no_pp_2():
    """
    Feature: transform param no pp scenario
    Description: dpmp from zero to no zero transform
    Expectation: Run success
    """
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "
        "--master_port=10802 --join=True --log_dir=./transform_d2d_no_pp_2 "
        "pytest -s transform_param_d2d.py::test_transform_d2d_no_pp_2 "
    )
    assert return_code == 0


def test_transform_d2d_no_pp_3():
    """
    Feature: transform param no pp scenario
    Description: dpmp from zero to zero transform
    Expectation: Run success
    """
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "
        "--master_port=10803 --join=True --log_dir=./transform_d2d_no_pp_3 "
        "pytest -s transform_param_d2d.py::test_transform_d2d_no_pp_3 "
    )
    assert return_code == 0


def test_transform_d2d_no_pp_4():
    """
    Feature: transform param no pp scenario
    Description: dpmp transform
    Expectation: Run success
    """
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "
        "--master_port=10804 --join=True --log_dir=./transform_d2d_no_pp_4 "
        "pytest -s transform_param_d2d.py::test_transform_d2d_no_pp_4 "
    )
    assert return_code == 0


def test_transform_d2d_pp_1():
    """
    Feature: transform param no pp scenario
    Description: dpmp transform
    Expectation: Run success
    """
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "
        "--master_port=10806 --join=True --log_dir=./transform_d2d_pp_1 "
        "pytest -s transform_param_d2d.py::test_transform_d2d_pp_1 "
    )
    assert return_code == 0


def test_transform_d2d_pp_2():
    """
    Feature: transform param no pp scenario
    Description: dpmp transform
    Expectation: Run success
    """
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "
        "--master_port=10807 --join=True --log_dir=./transform_d2d_pp_2 "
        "pytest -s transform_param_d2d.py::test_transform_d2d_pp_2 "
    )
    assert return_code == 0


def test_transform_d2d_pp_3():
    """
    Feature: transform param no pp scenario
    Description: dpmp transform
    Expectation: Run success
    """
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "
        "--master_port=10809 --join=True --log_dir=./transform_d2d_pp_3 "
        "pytest -s transform_param_d2d.py::test_transform_d2d_pp_3 "
    )
    assert return_code == 0


def test_transform_d2d_pp_4():
    """
    Feature: transform param no pp scenario
    Description: dpmp transform
    Expectation: Run success
    """
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "
        "--master_port=10808 --join=True --log_dir=./transform_d2d_pp_4 "
        "pytest -s transform_param_d2d.py::test_transform_d2d_pp_4 "
    )
    assert return_code == 0
