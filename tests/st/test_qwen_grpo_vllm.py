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
"""Test qwen grpo with vllm."""
import os
import sys

from omegaconf import OmegaConf

from tests.mark_utils import arg_mark
from tests.st.utils import check_log

WORKDIR = os.path.dirname(os.path.abspath(__file__))


def setup_mindrlhf_mf_env():
    print(f"WORKDIR is {WORKDIR}")
    mindrlhf_path = os.path.join(WORKDIR, "../../")
    mindformers_path = os.path.join(WORKDIR, "mindformers")
    sys.path = [mindrlhf_path, mindformers_path] + sys.path


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_qwen_grpo_vllm():
    """
    Feature: vllm and cp
    Description: generate use vllm, train open context parallel
    Expectation: The training process is successful and the checkpoints are saved.
    """
    setup_mindrlhf_mf_env()
    train_output_dir = os.path.join(WORKDIR, "grpo_train_output")
    saved_model_config_dir = os.path.join(train_output_dir, "dump_configs")

    os.environ["MINDRLHF_TEST"] = "1"
    os.environ["DUMP_RECONSTRUCT_CONFIG_PATH"] = saved_model_config_dir
    if not os.path.exists(saved_model_config_dir):
        os.makedirs(saved_model_config_dir, exist_ok=True)

    os.system(f"bash {WORKDIR}/run_qwen_grpo_vllm_test.sh")

    log_path = os.path.join(train_output_dir, "worker_0.log")
    check_pair = {"Save checkpoints in": 1}
    check_values = ["total_ref_batch_size: ref_model_batch_size * ref_dp, 2 * 4 = 8"]

    from mindrlhf.worker import InferWorker, RefWorker, TrainWorker

    base_config_dir = os.path.realpath(
        os.path.join(os.path.join(os.path.join(WORKDIR, ".."), "data"), "qwen_grpo_base_config")
    )
    # Validate infer config.
    base_infer_config = OmegaConf.load(os.path.join(base_config_dir, InferWorker.SAVED_MODEL_CONFIG_YAML))
    infer_config = OmegaConf.load(os.path.join(saved_model_config_dir, InferWorker.SAVED_MODEL_CONFIG_YAML))
    assert (
        base_infer_config == infer_config
    ), f"infer_config is not match with base_infer_config. infer_config={infer_config}"

    # Validate ref config.
    base_ref_config = OmegaConf.load(os.path.join(base_config_dir, RefWorker.SAVED_MODEL_CONFIG_YAML))
    ref_config = OmegaConf.load(os.path.join(saved_model_config_dir, RefWorker.SAVED_MODEL_CONFIG_YAML))
    assert base_ref_config == ref_config, f"ref_config is not match with base_ref_config. ref_config={ref_config}"

    # Validate train config.
    base_train_config = OmegaConf.load(os.path.join(base_config_dir, TrainWorker.SAVED_MODEL_CONFIG_YAML))
    train_config = OmegaConf.load(os.path.join(saved_model_config_dir, TrainWorker.SAVED_MODEL_CONFIG_YAML))
    assert (
        base_train_config == train_config
    ), f"train_config is not match with base_train_config. train_config={train_config}"

    # Validate device memory usage.
    device_memory = [62370, 32089, 43500]
    host_memory = [24, 230]
    check_log(log_path, check_pair, check_values, device_memory, host_memory)
    del os.environ["MINDRLHF_TEST"]
    del os.environ["DUMP_RECONSTRUCT_CONFIG_PATH"]


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_qwen_grpo_vllm_temporary():
    """
    Feature: vllm and cp
    Description: generate use vllm, train open context parallel
    Expectation: The training process is successful and the checkpoints are saved.
    """
    test_qwen_grpo_vllm()
