# Copyright 2024 Huawei Technologies Co., Ltd
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
# limitations under the License
# ============================================================================
"""GRPO model"""
from dataclasses import dataclass

@dataclass
class GRPOConfig:
    """
    GRPO config class which defines the model size
    """
    beta: float = 0.01 # KL coefficient
    num_generations: int = 8
    grpo_epochs: int = 2

    model_name: str = ''
    align_type: str = ''
    epochs: int = 10
    total_steps: int = 100000
    batch_size: int = 1
    checkpoint_interval = 10000
    eval_interval: int = 200

    optimizer: str = 'adamw'
    lr: float = 9.0e-6
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1.0e-8
    weight_decay: float = 0.01

    sceduler_name: str = 'cosine_annealing'
    t_max: int = 100000
    eta_min: float = 5.0e-6

    num_rollouts: int = 4
    chunk_size: int = 2
    ppo_epochs: int = 1
    init_kl_coef: float = 0.1
    kl_coef: float = 0.02
    target: float = 6.0
    horizon: int = 10000
    gamma: float = 1.0
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 1.0
    pretrain_coef: float = 0.9
    scale_reward: bool = None
    ref_mean: bool = False
    ref_std: bool = False
    gen_experience_kwargs: bool = False

    sink_size: int = 2
    device_target: str = 'Ascend'
    parallel_mode: str = 'semi_auto_parallel'
    full_batch: bool = True
    enable_alltoall: bool = False
    micro_batch_interleaved: int = 1
    start_lr: float = 5e-7  # 1e-12
    end_lr: float = 1e-10  # 1e-13
    warmup_step: int = 10 # 3200
    decay_steps: int = 200000
    opt_offload: bool = False
    mind_dataset_dir: str = "/path/train.mindrecord"
    inference_micro_size: int = 1
    save_ckpt_dir: str = "./"
    save_data_file: str = ""
    sft_model_path: str = "/path/model.yaml"
    critic_model_path: str = "/path/model.yaml"
    reward_model_path: str = "/path/model.yaml"
    is_shared_backbone: bool = True
    only_save_strategy: bool = False
    use_parallel: bool = False
    sync_ref_model: bool = True
    # Whether to synchronize the reference model with the active model every `ref_model_sync_steps`"
    ref_model_sync_steps: int = 50
