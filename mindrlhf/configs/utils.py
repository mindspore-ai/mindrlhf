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
"""Manage config related operations."""
import os

from mindformers import logger
from omegaconf import DictConfig, OmegaConf

from mindrlhf.configs.grpo_configs import VllmMode, GRPOConfig


class GRPOConfigGenerator:
    """Generator of GRPO configuration."""

    @staticmethod
    def _dump_reconstructed_config(config: DictConfig, save_path: str, filename: str):
        """
        Save reconstructed config to yaml for validation.

        Args:
            config (DictConfig): Config instance.
            save_path (str): Save path.
            filename (str): Saved filename.
        """
        if not isinstance(config, DictConfig):
            raise TypeError(f"config must be type of DictConfig, but got {type(config)}")
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"yaml saved path is not found: {save_path}. Maybe the dir has not been created.")
        saved_file_path = os.path.join(save_path, filename)
        if os.path.exists(saved_file_path):
            logger.warning(f"File has been overwritten: {saved_file_path}")
        with open(os.path.join(saved_file_path), "w") as yaml_cfg:
            yaml_cfg.writelines(OmegaConf.to_yaml(config))

    @staticmethod
    def _reconstruct_infer_config(config_obj: DictConfig) -> DictConfig:
        """
        Reconstruct infer config with parallel config and other args.

        Args:
            config_obj (DictConfig): Config object.

        Returns:
            DictConfig, merged infer config.
        """
        if not os.path.exists(config_obj.generate_config.model_config):
            raise FileNotFoundError(f"Cannot find inference model config: {config_obj.generate_config.model_config}")
        reconstructed_model_config = OmegaConf.load(config_obj.generate_config.model_config)
        reconstructed_model_config.context = config_obj.context
        reconstructed_model_config.use_parallel = config_obj.rl_config.use_parallel
        reconstructed_model_config.parallel_config = config_obj.generate_config.parallel_config
        reconstructed_model_config.model.model_config.parallel_config = config_obj.generate_config.parallel_config
        reconstructed_model_config.model.model_config.seq_length = config_obj.rl_config.seq_length
        reconstructed_model_config.model.model_config.offset = config_obj.generate_config.offset
        reconstructed_model_config.model.model_config.max_decode_length = (
            config_obj.generate_config.sampling_config.max_tokens
        )
        reconstructed_model_config.model.model_config.min_decode_length = (
            config_obj.generate_config.sampling_config.min_tokens
        )
        # It could be a bug. Don't see moe_config in model_config in mindformers.
        # reconstructed_model_config.model.model_config.moe_config = reconstructed_model_config.moe_config
        config_obj.generate_config.reconstructed_model_config = reconstructed_model_config
        return config_obj

    @staticmethod
    def _reconstruct_ref_config(config_obj: DictConfig) -> DictConfig:
        """
        Reconstruct reference config with parallel config and other args.

        Args:
            config_obj (DictConfig): Config object.

        Returns:
            DictConfig, merged reference config.
        """
        if not os.path.exists(config_obj.ref_config.model_config):
            raise FileNotFoundError(f"Cannot find reference model config: {config_obj.ref_config.model_config}")
        reconstructed_model_config = OmegaConf.load(config_obj.ref_config.model_config)
        reconstructed_model_config.use_parallel = config_obj.rl_config.use_parallel
        reconstructed_model_config.parallel_config = config_obj.ref_config.parallel_config
        reconstructed_model_config.recompute_config = config_obj.ref_config.recompute_config
        reconstructed_model_config.model.model_config.seq_length = config_obj.rl_config.seq_length
        reconstructed_model_config.model.model_config.offset = config_obj.ref_config.offset
        reconstructed_model_config.model.model_config.use_past = False
        reconstructed_model_config.model.model_config.use_eod_attn_mask_compression = (
            config_obj.ref_config.use_eod_attn_mask_compression
        )
        reconstructed_model_config.model.model_config.checkpoint_name_or_path = config_obj.ref_config.load
        reconstructed_model_config.model.model_config.parallel_config = config_obj.ref_config.parallel_config
        # reconstructed_model_config.moe_config = config_obj.ref_config.moe_config
        # reconstructed_model_config.model.model_config.parallel_config.recompute = reconstructed_model_config.recompute_config
        # It could be a bug. Don't see moe_config in model_config in mindformers.
        # reconstructed_model_config.model.model_config.moe_config = reconstructed_model_config.moe_config
        config_obj.ref_config.reconstructed_model_config = reconstructed_model_config
        return config_obj

    @staticmethod
    def _reconstruct_train_config(config_obj: DictConfig) -> DictConfig:
        """
        Reconstruct train config with parallel config and other args.

        Args:
            config_obj (DictConfig): Config object.

        Returns:
            DictConfig, merged train config.
        """
        if not os.path.exists(config_obj.actor_config.model_config):
            raise FileNotFoundError(f"Cannot find train model config: {config_obj.actor_config.model_config}")
        reconstructed_model_config = OmegaConf.load(config_obj.actor_config.model_config)
        OmegaConf.set_struct(reconstructed_model_config, False)
        reconstructed_model_config.use_parallel = config_obj.rl_config.use_parallel
        reconstructed_model_config.parallel_config = config_obj.actor_config.parallel_config
        reconstructed_model_config.recompute_config = config_obj.actor_config.recompute_config
        reconstructed_model_config.model.model_config.seq_length = config_obj.rl_config.seq_length
        reconstructed_model_config.model.model_config.offset = config_obj.actor_config.offset
        reconstructed_model_config.model.model_config.use_eod_attn_mask_compression = (
            config_obj.actor_config.use_eod_attn_mask_compression
        )
        # Next two line could be a bug.
        reconstructed_model_config.model.model_config.parallel_config = config_obj.actor_config.parallel_config
        OmegaConf.set_struct(reconstructed_model_config.model.model_config.parallel_config, False)
        reconstructed_model_config.model.model_config.parallel_config.recompute = OmegaConf.to_container(
            config_obj.actor_config.recompute_config
        )
        config_obj.actor_config.reconstructed_model_config = reconstructed_model_config
        return config_obj

    @staticmethod
    def _post_process(config_obj: DictConfig) -> DictConfig:
        """
        Post process and validate constraints between params.

        Args:
            config_obj (DictConfig): Configuration.

        Returns:
            DictConfig, validated configuration.
        """
        if (
            config_obj.rl_config.save_prompt_completions_data
            and config_obj.rl_config.save_prompt_completions_interval <= 0
        ):
            raise ValueError(
                f"save_prompt_completions_interval should be positive, "
                f"but got {config_obj.rl_config.save_prompt_completions_interval}. "
                f"Set save_prompt_completions_data to False."
            )
        if not isinstance(config_obj.generate_config.use_vllm, VllmMode):
            raise ValueError(f"use_vllm should be type of VllmMode, and value should be 0, 1 or 2.")
        return config_obj

    @staticmethod
    def _validate(config_obj: DictConfig):
        """
        Validate args.

        Args:
            config_obj (DictConfig): Validate args whether legal.
        """
        if config_obj.rl_config.pack_num < 1:
            raise ValueError("pack_num must >= 1!")
        if config_obj.rl_config.reshard_mem_opt_level not in [0, 1]:
            raise ValueError(
                f"reshard_mem_opt_level can only be 0 or 1, but got {config_obj.rl_config.reshard_mem_opt_level}"
            )
        if config_obj.actor_config.save:
            if config_obj.rl_config.save_ckpt_interval <= 0:
                raise ValueError(
                    f"rl_config.save_ckpt_interval should be lager than 0, but got "
                    f"{config_obj.rl_config.save_ckpt_interval}"
                )
            if config_obj.rl_config.save_max_ckpt_num <= 0:
                raise ValueError(
                    f"rl_config.save_max_ckpt_num should be lager than 0, but got "
                    f"{config_obj.rl_config.save_max_ckpt_num}"
                )

    @classmethod
    def create_config(cls, config_from_yaml: DictConfig) -> DictConfig:
        """
        Create config instance.

        Args:
            config_from_yaml (DictConfig): Config from yaml.

        Returns:
            DictConfig, verified configuration.
        """
        OmegaConf.set_struct(config_from_yaml, False)
        base_conf = OmegaConf.structured(GRPOConfig)
        config_from_yaml = OmegaConf.merge(base_conf, config_from_yaml)
        cls._validate(config_from_yaml)
        config_from_yaml = cls._reconstruct_infer_config(config_from_yaml)
        config_from_yaml = cls._reconstruct_ref_config(config_from_yaml)
        config_from_yaml = cls._reconstruct_train_config(config_from_yaml)
        config_from_yaml = cls._post_process(config_from_yaml)
        if os.getenv("DUMP_RECONSTRUCT_CONFIG_PATH"):
            GRPOConfigGenerator._dump_reconstructed_config(
                config_from_yaml.generate_config.reconstructed_model_config,
                save_path=os.getenv("DUMP_RECONSTRUCT_CONFIG_PATH"),
                filename="reconstructed_infer_model_config.yaml",
            )
            GRPOConfigGenerator._dump_reconstructed_config(
                config_from_yaml.ref_config.reconstructed_model_config,
                save_path=os.getenv("DUMP_RECONSTRUCT_CONFIG_PATH"),
                filename="reconstructed_ref_model_config.yaml",
            )
            GRPOConfigGenerator._dump_reconstructed_config(
                config_from_yaml.actor_config.reconstructed_model_config,
                save_path=os.getenv("DUMP_RECONSTRUCT_CONFIG_PATH"),
                filename="reconstructed_train_model_config.yaml",
            )
        OmegaConf.set_struct(config_from_yaml, True)
        return config_from_yaml
