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
""" Test Clip Higher """

import os
import yaml
import pytest

from tests.st.utils import check_log

root_path = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.clip_higher_dt_1
def test_clip_higher_dt_1():
    """
    Feature:  test Clip Higher DT 1
    Description: num_iterations=1, align to origin case
    Expectation: success.
    """
    default_yaml = "./qwen2_5/grpo_config_st.yaml"
    dt_yaml = './qwen2_5/clip_higher_dt_1.yaml'
    with open('config.yaml', 'r') as file:
        config = yaml.load(default_yaml, Loader=yaml.FullLoader)
    config["num_iterations"] = 1
    yaml_str = yaml.dump(config)
    with open(dt_yaml, 'w') as file:
        file.write(yaml_str)
    os.system(f"bash {root_path}/run_qwen_grpo_test.sh {dt_yaml}")

    log_path = f"{root_path}/qwen2_one_log/worker_0.log"
    check_pair = {"Save checkpoints in": 1}

    check_log(log_path, check_pair)


@pytest.mark.clip_higher_dt_2
def test_clip_higher_dt_2():
    """
    Feature:  test Clip Higher DT 2
    Description: default setting(num_iteration=2, epsilon_high=epsilon_low=0.2), align to GRPO
    Expectation: success.
    """
    default_yaml = "./qwen2_5/grpo_config_st.yaml"
    os.system(f"bash {root_path}/run_qwen_grpo_test.sh {default_yaml}")

    log_path = f"{root_path}/qwen2_one_log/worker_0.log"
    check_pair = {"Save checkpoints in": 1}

    check_log(log_path, check_pair)


@pytest.mark.clip_higher_dt_3
def test_clip_higher_dt_3():
    """
    Feature:  test Clip Higher DT 3
    Description: epsilon_high=0.28, epsilon_low=0.2, align to DAPO
    Expectation: success.
    """
    default_yaml = "./qwen2_5/grpo_config_st.yaml"
    dt_yaml = './qwen2_5/clip_higher_dt_3.yaml'
    with open('config.yaml', 'r') as file:
        config = yaml.load(default_yaml, Loader=yaml.FullLoader)
    config["epsilon_high"] = 0.28
    yaml_str = yaml.dump(config)
    with open(dt_yaml, 'w') as file:
        file.write(yaml_str)
    os.system(f"bash {root_path}/run_qwen_grpo_test.sh {dt_yaml}")

    log_path = f"{root_path}/qwen2_one_log/worker_0.log"
    check_pair = {"Save checkpoints in": 1}

    check_log(log_path, check_pair)
