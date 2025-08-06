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
"""worker"""

from abc import ABC
from dataclasses import dataclass
from enum import Enum
import os
from omegaconf import OmegaConf

import numpy as np


@dataclass
class GRPOData:
    """
    grpo data
    """

    prompt_completion_ids: np.array
    responses_mask: np.array
    ref_per_token_logps: np.array
    advantages: np.array
    actual_sequence_length: np.array
    sample_index: np.array
    sample_valid_length: np.array
    old_per_token_logps: np.array


class Worker(ABC):
    """Base worker."""

    class WorkerType(Enum):
        """Define worker type."""

        NONE = "none"
        INFER = "infer"
        REF = "ref"
        TRAIN = "train"
        TRANSFORM = "transform"
        OLD_POLICY = "old_policy"

    def __init__(self, **kwargs):
        self.config = kwargs.get("config")
        self.worker_type = kwargs.get("worker_type", self.WorkerType.NONE)
        if self.worker_type == self.WorkerType.NONE:
            raise ValueError(f"Worker type is not supported.")

    @staticmethod
    def dump_mf_conf_to_yaml(conf, filename: str):
        """
        Dump MindFormerConfig to yaml.
        It only works when DUMP_RECONSTRUCT_CONFIG_PATH is set and save yaml to
        DUMP_RECONSTRUCT_CONFIG_PATH path.

        Args:
            conf: MindFormerConfig.
            filename: To be saved yaml filename.
        """
        if not os.getenv("DUMP_RECONSTRUCT_CONFIG_PATH"):
            return
        mf_infer_conf = OmegaConf.create(conf.to_dict())
        with open(os.path.join(os.getenv("DUMP_RECONSTRUCT_CONFIG_PATH"), filename), "w") as f:
            f.writelines(OmegaConf.to_yaml(mf_infer_conf))

    @property
    def model_name(self):
        """Property of rl_config.model_name."""
        return self.config.rl_config.model_name

    @property
    def reshard_mode(self):
        """Property of rl_config.reshard_mode."""
        return self.config.rl_config.reshard_mode

    @property
    def use_parallel(self):
        """Property of rl_config.use_parallel."""
        return self.config.rl_config.use_parallel

    @property
    def load_ckpt_format(self):
        """Property of rl_config.load_ckpt_format."""
        return self.config.rl_config.load_ckpt_format

    @property
    def model_path(self):
        """Property of model_path(checkpoint)."""
        if self.worker_type == self.WorkerType.INFER:
            return self.config.generate_config.load
        if self.worker_type == self.WorkerType.REF:
            return self.config.ref_config.load
        if self.worker_type in {self.WorkerType.TRAIN, self.WorkerType.OLD_POLICY}:
            return self.config.actor_config.load
        raise ValueError(f"worker_type is not supported.")

    @property
    def micro_batch_num(self):
        """Property of micro_batch_num."""
        if self.worker_type == self.WorkerType.INFER:
            return self.config.generate_config.parallel_config.micro_batch_num
        if self.worker_type == self.WorkerType.REF:
            return self.config.ref_config.parallel_config.micro_batch_num
        if self.worker_type in {self.WorkerType.TRAIN, self.WorkerType.OLD_POLICY}:
            return self.config.actor_config.parallel_config.micro_batch_num
        raise ValueError(f"worker_type is not supported.")

    @property
    def reconstructed_model_config(self):
        if self.worker_type == self.WorkerType.INFER:
            return self.config.generate_config.reconstructed_model_config
        if self.worker_type == self.WorkerType.REF:
            return self.config.ref_config.reconstructed_model_config
        if self.worker_type in {self.WorkerType.TRAIN, self.WorkerType.OLD_POLICY}:
            return self.config.actor_config.reconstructed_model_config
        raise ValueError(f"worker_type is not supported.")

    @property
    def model_parallel(self) -> int:
        """Property of model_parallel."""
        if self.worker_type == self.WorkerType.INFER:
            return self.config.generate_config.parallel_config.model_parallel
        if self.worker_type == self.WorkerType.REF:
            return self.config.ref_config.parallel_config.model_parallel
        if self.worker_type in {self.WorkerType.TRAIN, self.WorkerType.OLD_POLICY}:
            return self.config.actor_config.parallel_config.model_parallel
        raise ValueError(f"worker_type is not supported.")

    @property
    def data_parallel(self) -> int:
        """Property of data_parallel."""
        if self.worker_type == self.WorkerType.INFER:
            return self.config.generate_config.parallel_config.data_parallel
        if self.worker_type == self.WorkerType.REF:
            return self.config.ref_config.parallel_config.data_parallel
        if self.worker_type in {self.WorkerType.TRAIN, self.WorkerType.OLD_POLICY}:
            return self.config.actor_config.parallel_config.data_parallel
        raise ValueError(f"worker_type is not supported.")

    @property
    def pipeline_stage(self) -> int:
        """Property of pipeline_stage."""
        if self.worker_type == self.WorkerType.INFER:
            return self.config.generate_config.parallel_config.pipeline_stage
        if self.worker_type == self.WorkerType.REF:
            return self.config.ref_config.parallel_config.pipeline_stage
        if self.worker_type in {self.WorkerType.TRAIN, self.WorkerType.OLD_POLICY}:
            return self.config.actor_config.parallel_config.pipeline_stage
        raise ValueError(f"worker_type is not supported.")

    @property
    def save_strategy_dir(self) -> str:
        """Property of rl_config.save_strategy_dir."""
        return self.config.rl_config.save_strategy_dir

    @property
    def is_old_policy_enabled(self) -> bool:
        """Whether old policy is enabled."""
        return self.config.rl_config.enable_oldpolicy

    @property
    def is_optimizer_parallel_enabled(self) -> bool:
        """Whether optimizer parallel is enabled."""
        return self.config.actor_config.enable_parallel_optimizer and self.data_parallel > 1


def format_time_delta(seconds):
    """format time delta to string"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{seconds:.4f}"
