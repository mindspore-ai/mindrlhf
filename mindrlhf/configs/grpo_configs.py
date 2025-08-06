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
# limitations under the License
# ============================================================================
"""GRPO model"""
from dataclasses import dataclass, field
from enum import Enum

from omegaconf import DictConfig, OmegaConf


class VllmMode(Enum):
    ORIGIN = 0
    VLLM = 1
    DEBUG = 2  # DEBUG mode: init model with vllm, but generate with mindformers


def _gen_default_verifier_weight():
    """Generate default verifier weight."""
    return [1.0, 1.0]


def _gen_default_verifier_func():
    """Generate default verifier weight."""
    return ["qwen_accuracy_reward", "format_reward"]


@dataclass
class ParallelConfig:
    """parallel config"""

    # Data parallel.
    data_parallel: int = 1
    # Model parallel.
    model_parallel: int = 4
    # Pipeline parallel.
    pipeline_stage: int = 2
    # Expert parallel.
    expert_parallel: int = 1
    # Enable sequence parallel.
    use_seq_parallel: bool = False
    # Micro batch number.
    micro_batch_num: int = 2
    # Enable vocab embedding shard on dp dimension.
    vocab_emb_dp: bool = True
    # Context parallel(Long sequence parallel).
    context_parallel: int = 1


@dataclass
class RecomputeConfig:
    """recompute config"""

    # Enable recompute.
    recompute: bool = False
    # Enable select recompute.
    select_recompute: bool = False
    # Enable communication select recompute.
    select_comm_recompute: bool = False
    # Enable optimizer parallel communication recompute.
    parallel_optimizer_comm_recompute: bool = False
    # Enable model parallel communication recompute.
    mp_comm_recompute: bool = True
    # Enable slice activation recompute.
    recompute_slice_activation: bool = False


@dataclass
class Optimizer:
    """optimizer"""

    # Optimizer.
    type: str = "adamw"
    # Adam beta1.
    adam_beta1: float = 0.9
    # Adam beta2.
    adam_beta2: float = 0.95
    # Eps.
    eps: float = 1.0e-8
    # Weight decay.
    weight_decay: float = 0.01
    # Optimizer offload.
    opt_offload: bool = False


@dataclass
class LRSchedule:
    """lr schedule"""

    # LR decay style.
    lr_decay_style: str = "cosine"
    # LR.
    lr: float = 5.0e-7
    # Min LR.
    min_lr: float = 1.0e-10
    # Warmup steps.
    warmup_step: int = 10
    # Decay steps.
    decay_steps: int = 200000


@dataclass
class ActorConfig:
    """Actor model config."""

    # Actor model path.
    load: str = ""
    # Checkpoints saved path. Enable checkpoints saving when it's not empty.
    save: str = ""
    # Actor model config.
    model_config: str = ""
    # FIXME: what means?
    offset: int = 0
    # Enable use EOD attention mask compression.
    use_eod_attn_mask_compression: bool = False
    # Loss scale.
    loss_scale_value: int = 1
    # Enable actor model parallel optimizer.
    enable_parallel_optimizer: bool = False
    # Actor model parallel config.
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    # Actor model recompute config.
    recompute_config: RecomputeConfig = field(default_factory=RecomputeConfig)
    # Actor model optimizer config.
    optimizer: Optimizer = field(default_factory=Optimizer)
    # Actor model learning rate scheduler.
    lr_schedule: LRSchedule = field(default_factory=LRSchedule)
    # moe_config: DictConfig = field(default_factory=OmegaConf.create)
    enable_alltoall: bool = False
    # Built-in attr. Non-public.
    reconstructed_model_config: DictConfig = field(default_factory=OmegaConf.create)


@dataclass
class RefConfig:
    """reference model config"""

    # Ref model config.
    model_config: str = ""
    # Ref model path.
    load: str = ""
    # Ref model batch size.
    ref_model_batch_size: int = 1
    # Ref model offset.
    offset: int = 0
    # Enable use EOD attention mask compression.
    use_eod_attn_mask_compression: bool = False

    # Whether to synchronize the reference model with the policy model every `ref_model_sync_steps`
    sync_ref_model: bool = False
    # Ref model sync steps.
    ref_model_sync_steps: int = 50

    # Ref model parallel config.
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    # Ref model recompute config.
    recompute_config: RecomputeConfig = field(default_factory=RecomputeConfig)
    # Built-in attr. Non-public.
    reconstructed_model_config: DictConfig = field(default_factory=OmegaConf.create)


def _gen_default_eos_token_id():
    """Generate default eos token ids."""
    return [151645, 151643]


@dataclass
class SamplingConfig:
    """Sampling config."""

    # BOS token id.
    bos_token_id: int = 151643
    # Pad token id.
    pad_token_id: int = 151643
    # EOS token id.
    eos_token_id: list = field(default_factory=_gen_default_eos_token_id)
    # Max decode length.
    max_tokens: int = 512
    # Min decode length.
    min_tokens: int = 2
    # Temperature.
    temperature: float = 1.0
    # Repetition penalty.
    repetition_penalty: float = 1.0
    # Top P.
    top_p: float = 1.0
    # Top K.
    top_k: int = -1
    # Detokenize.
    detokenize: bool = False
    # Log probs.
    logprobs: float = 1
    # Min p.
    min_p: float = 0.01


@dataclass
class GenerateConfig:
    """Generate model config."""

    # Model config.
    model_config: str = ""
    # Model path.
    load: str = ""
    # Inference model batch size.
    infer_model_batch_size: int = 1
    # Offset.
    offset: int = 0
    # Enable use EOD attention mask compression.
    use_eod_attn_mask_compression: bool = False
    # Inference mode.
    # 0: MindFormers.
    # 1: VLLM.
    # 2: DEBUG mode. Init model with vllm, but generate with mindformers
    use_vllm: VllmMode = VllmMode.VLLM

    # Parallel config.
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)

    # vLLM setting.
    block_size: int = 16
    max_model_len: int = 25536
    max_num_batched_tokens: int = 25536
    max_num_seqs: int = 1024
    max_prompt_length: int = 2048
    num_scheduler_steps: int = 32
    gpu_memory_utilization: float = 0.8
    trust_remote_code: bool = True
    # vLLM post-process setting.
    sampling_config: SamplingConfig = field(default_factory=SamplingConfig)
    # Built-in attr. Non-public.
    reconstructed_model_config: DictConfig = field(default_factory=OmegaConf.create)


@dataclass
class RewardConfig:
    """GRPO reward config."""

    verifier_function: list = field(default_factory=_gen_default_verifier_func)
    verifier_weight: list = field(default_factory=_gen_default_verifier_weight)


def _gen_default_jit_config():
    """Generate default jit_config."""
    return OmegaConf.create({"jit_level": "O0"})


def _gen_default_ascend_config():
    """Generate default ascend_config."""
    return OmegaConf.create({"precision_mode": "must_keep_origin_dtype"})


@dataclass
class ContextConfig:
    """context"""

    # MindSpore run mode.
    # 0: Graph Mode.
    # 1: Pynative Mode.
    mode: int = 0
    # Run device.
    device_target: str = "Ascend"
    # Max nested cells depth.
    max_call_depth: int = 10000
    # Max device memory.
    max_device_memory: str = "55GB"
    # Save graphs.
    save_graphs: bool = False
    # Save graphs path.
    save_graphs_path: str = "./graph"
    # JIT config.
    jit_config: DictConfig = field(default_factory=_gen_default_jit_config)
    # Device memory optimize level.
    memory_optimize_level: str = "O0"
    # Ascend config.
    ascend_config: DictConfig = field(default_factory=_gen_default_ascend_config)


@dataclass
class MonitorConfig:
    """Host memory monitor setting."""

    # Monitor sampling interval.
    host_monitor_interval: float = -1.0
    # Monitor sampling steps.
    host_monitor_steps: list = field(default_factory=list)
    # Enable host memory protection.
    host_memory_protection: bool = False
    # The process will be stopped when host memory usage reach
    # 'host_max_memory_threshold' when 'host_memory_protection' is True.
    host_max_memory_threshold: float = 0.95


@dataclass
class RLConfig:
    """Define RL config dataclass."""

    # model_name is used to select model.
    model_name: str = "qwen2.5"
    # Global random seed.
    seed: int = 1
    # Enable deterministic compute, options 'ON' or 'OFF'
    deterministic: str = "OFF"
    # FIXME: reserved?
    align_type: str = "rlhf_stages"
    # Dataset file path.
    dataset_file: str = ""
    # tokenizer_type is used to select tokenizer, it's same to model_name by default.
    tokenizer_type: str = "qwen2.5"
    # Tokenizer path.
    tokenizer_dir: str = ""

    # Epoch number.
    epochs: int = 1
    # Batch size number.
    batch_size: int = 1
    # Sequence length for train and inference.
    seq_length: int = 8192
    # Enable parallel training and inference.
    use_parallel: bool = True
    # Checkpoints format, it's "hf_safetensors" by default. Options: "hf_safetensors", "ms_safetensors".
    load_ckpt_format: str = "hf_safetensors"
    # MindSpore parallel mode, it's "semi_auto_parallel" by default. Options: "semi_auto_parallel", "auto_parallel".
    parallel_mode: str = "semi_auto_parallel"
    # Distributed strategy saved path.
    save_strategy_dir: str = "./strategy"
    # Save checkpoints interval.
    save_ckpt_interval: int = 500
    # Max number of saved checkpoints.
    save_max_ckpt_num: int = 5
    # Resume training.
    resume_training: bool = False

    # Pack number for pack training.
    pack_num: int = 1
    # Micro batch interleaved number for pipeline parallel.
    micro_batch_interleaved: int = 1

    # KL coefficient.
    beta: float = 0.0
    # Number of generations.
    num_generations: int = 8
    # Number of rollouts.
    num_rollouts: int = 1
    # TODO: ?
    chunk_size: int = 1

    # Enable old policy model.
    enable_oldpolicy: bool = True

    # Clip higher related setting.
    # Number of iterations.
    num_iterations: int = 1
    # Low epsilon.
    epsilon_low: float = 0.2
    # High epsilon.
    epsilon_high: float = 0.2

    # Reshard memory optimization level.
    # 0: do not optimize mem during resharding.
    # 1: offload all src and dst param during resharding.
    reshard_mem_opt_level: int = 0
    # Reshard mode.
    # 0: run reshard in PYNATIVE mode.
    # 1: run reshard in GRAPH mode.
    reshard_mode: int = 0
    # Enable reshard optimizer.
    enable_reshard_optimizer: bool = False

    # Enable MindSpore compile cache. Only used by inference, move it to Context?
    enable_compile_cache: bool = False
    # Compile cache saved path.
    compile_cache_path: str = "./compile_cache"

    # Save internal data to MindRecord path(feed in train data).
    save_data_file: str = ""

    # Enable save prompt completions data to json(raw train data).
    save_prompt_completions_data: bool = False
    # Save prompt completions data interval.
    save_prompt_completions_interval: int = 1
    # Save prompt completions data path.
    save_prompt_completions_dir: str = ""

    # Enable clip frac, KL loss, actor loss monitoring.
    enable_full_monitor: bool = False
    # Whether calculate entropy.
    calculate_entropy: bool = False

    # Enable performance collector.
    performance_stats: bool = False

    # Enable tensorboard.
    tensorboard: bool = False
    # Tensorboard dir.
    tensorboard_dir: str = ""
    # Tensorboard queue size.
    tensorboard_queue_size: int = 10


@dataclass
class ProfilerConfig:
    """profiler config"""

    profile: bool = False
    mstx: bool = False
    stage: str = "all"
    profile_save_path: str = "./profiler_data"
    profile_level: str = "level1"
    profile_with_memory: bool = False
    profile_with_cpu: bool = True
    profile_with_npu: bool = True
    profile_with_stack: bool = False
    profile_step_start: int = 0
    profile_step_end: int = 1
    profile_analysis: bool = False
    profile_ranks: str = "all"


@dataclass
class GRPOConfig:
    """
    GRPO config class which defines the model size
    """

    rl_config: RLConfig = field(default_factory=RLConfig)
    actor_config: ActorConfig = field(default_factory=ActorConfig)
    ref_config: RefConfig = field(default_factory=RefConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    generate_config: GenerateConfig = field(default_factory=GenerateConfig)
    monitor_config: MonitorConfig = field(default_factory=MonitorConfig)
    profiler_config: ProfilerConfig = field(default_factory=ProfilerConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
