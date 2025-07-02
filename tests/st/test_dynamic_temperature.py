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
"""Test dynamic temperature"""

import pytest
from mindrlhf.trainer.spmd.grpo_trainer import GRPOTrainer


@pytest.mark.parametrize(
    "cur_step, expected_temperature",
    [
        (0, 0.9),
        (1, 0.9),
        (2, 0.9),
        (3, 1.1),
        (4, 1.1),
        (5, 1.1),
        (9, 1.5),
        (10, 1.5),
        (11, 1.5),
        (12, 1.5),
        (13, 1.5),
        (14, 1.5),
    ],
)
def test_case(cur_step: int, expected_temperature: float):
    """
    Feature: dynamic temperature control
    Description: compute temperature
    Expectation: Run success
    """
    assert GRPOTrainer.get_temperature([0.9, 1.1, 1.3, 1.5], 3, cur_step) == expected_temperature
