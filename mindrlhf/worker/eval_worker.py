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
"""GRPO EvalMaker"""

# python
import json
import os
import numpy as np

# mindspore
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore.dataset import MindDataset
from mindspore import Tensor

# mindformers
from mindformers import logger

# mindrlhf
from mindrlhf.worker.worker import Worker
from mindrlhf.configs.grpo_configs import VllmMode
from mindrlhf.reward.reward_fn import accuracy_reward, format_reward, qwen_accuracy_reward
from mindrlhf.reward.kk_reward_fn import kk_reward
from mindrlhf.trainer.spmd.grpo_experience_maker import GRPOExperienceMaker
from mindrlhf.utils.utils import CustomJsonEncoder


class EvalWorker(Worker):
    """
    This class do online validation
    """

    def __init__(self, grpo_config, tokenizer, infer):
        super().__init__()
        self.grpo_config = grpo_config
        self.tokenizer = tokenizer
        self.infer = infer

        self.infer_dp = self.infer.get_infer_dp()
        self._init_val_dataset()
        self._init_reward_fn()

    def _init_val_dataset(self):
        """init val dataset"""
        self.val_mind_dataset_dir = self.grpo_config.eval_config.val_dataset_file

        if self.val_mind_dataset_dir is not None and self.grpo_config.eval_config.eval_freq > 0:
            columns_to_project = ["prompt_ids", "pretrain_ids"]
            ms.dataset.config.set_seed(2023)
            dataset = (
                MindDataset(self.val_mind_dataset_dir)
                .project(columns=columns_to_project)
                .repeat(self.grpo_config.eval_config.val_n)
            )
            self.val_prompt_dataset = dataset
            self.val_prompt_dataloader = dataset.take()
            logger.info(f"Validation dataset len: {len(dataset)}")
            self.val_prompt_dataloader = self.val_prompt_dataloader.batch(
                batch_size=len(dataset), drop_remainder=True
            )
        else:
            logger.info("In main task, there is not dataset for making experience")

    def _init_reward_fn(self):
        """init reward function"""
        logger.info("EvalWorker: _init_reward_fn")
        if self.grpo_config.reward_config.val_verifier_function:
            verifier_function_list = self.grpo_config.reward_config.val_verifier_function
        else:
            verifier_function_list = ["accuracy_reward", "format_reward"]
        if self.grpo_config.reward_config.val_verifier_weight:
            verifier_weight = self.grpo_config.reward_config.val_verifier_weight
        else:
            verifier_weight = [1.0, 1.0]
        logger.info(f"verifier_function_list:{verifier_function_list}")
        logger.info(f"verifier_weight:{verifier_weight}")

        verifier_function = []
        for reward_func_str in verifier_function_list:
            if "qwen" in self.grpo_config.rl_config.model_name and reward_func_str == "accuracy_reward":
                reward_func_str = "qwen_accuracy_reward"
            if reward_func_str == "accuracy_reward":
                verifier_function.append(accuracy_reward)
            elif reward_func_str == "format_reward":
                verifier_function.append(format_reward)
            elif reward_func_str == "qwen_accuracy_reward":
                verifier_function.append(qwen_accuracy_reward)
            elif reward_func_str == "kk_reward":
                verifier_function.append(kk_reward)
            else:
                raise ValueError(f"Unsupported reward function {reward_func_str}")
        self.verifier_function = verifier_function

        # Reward weights
        if len(verifier_weight) != len(verifier_function):
            raise ValueError(
                f"Number of reward weights ({len(verifier_weight)}) must match number of reward "
                f"functions ({len(verifier_function)})"
            )
        self.verifier_weight = np.array(verifier_weight, dtype=np.float32)

    def _remove_padding(self, left_padding_prompts, right_padding_responses, pad_token_id):
        """remove padding for prompts and responses"""
        # pylint: disable=W0212
        no_padding_prompts = GRPOExperienceMaker._remove_left_padding(
            left_padding_prompts, padding_token=pad_token_id
        )
        # pylint: disable=W0212
        no_padding_responses = GRPOExperienceMaker._remove_right_padding(
            right_padding_responses, padding_token=pad_token_id
        )
        prompts = self.tokenizer.decode(no_padding_prompts, skip_special_tokens=True)
        completions = self.tokenizer.decode(no_padding_responses, skip_special_tokens=True)
        return no_padding_prompts, no_padding_responses, prompts, completions

    def validate(self, step: int):
        """
        Online validate.
        """
        logger.info(f"Validation start | step: {step}")
        self.infer.load()

        pad_token_id = self.grpo_config.generate_config.sampling_config.pad_token_id
        val_prompt_iterator = self.val_prompt_dataloader.create_tuple_iterator()
        val_all_rewards = []

        batch = next(val_prompt_iterator)
        val_prompt_tensors = Tensor(batch[0], mstype.int32).asnumpy()
        # pylint: disable=W0212
        val_solution_ids = GRPOExperienceMaker._remove_right_padding(
            Tensor(batch[1], mstype.int32).asnumpy(), padding_token=pad_token_id
        )
        val_solution = self.tokenizer.decode(val_solution_ids, skip_special_tokens=True)
        for i in range(len(val_solution)):
            val_solution[i] = "$" + val_solution[i] + "$"
        # pylint: disable=W0212
        val_input_ids_numpy = GRPOExperienceMaker._split_for_data_parallel(val_prompt_tensors, self.infer_dp)

        max_tokens = self.grpo_config.eval_config.val_max_decode_length
        if self.infer.use_vllm == VllmMode.ORIGIN:
            results = []
            input_bs = val_input_ids_numpy.shape[0] // self.infer_dp
            for idx in range(self.infer_dp):
                result = self.infer.generate(
                    val_input_ids_numpy[idx * input_bs : (idx + 1) * input_bs, :],
                    max_tokens=max_tokens,
                    is_val=True,
                )
                for res_idx in range(len(result)):
                    if len(results) == len(result):
                        results[res_idx] = np.concatenate((results[res_idx], result[res_idx]))
                    else:
                        results.append(result[res_idx])
        else:
            results = self.infer.generate(val_input_ids_numpy, max_tokens, is_val=True)

        right_padding_responses, _, left_padding_prompts, _ = self.infer.post_process_infer_outputs(
            results, is_val=True
        )
        no_padding_prompts, no_padding_responses, prompts, completions = self._remove_padding(
            left_padding_prompts, right_padding_responses, pad_token_id
        )
        rewards_per_func = np.zeros((len(no_padding_prompts), len(self.verifier_function)), dtype=np.float32)

        for i, reward_func in enumerate(self.verifier_function):
            if reward_func is accuracy_reward or reward_func is qwen_accuracy_reward:
                output_reward_func, _ = reward_func(solution=val_solution, completions=completions)
            else:
                output_reward_func = reward_func(solution=val_solution, completions=completions)
            rewards_per_func[:, i] = np.array(output_reward_func, dtype=np.float32)
        val_rewards = (rewards_per_func * self.verifier_weight[np.newaxis, :]).sum(axis=1)
        val_all_rewards.extend(val_rewards)

        val_results = [
            {k: v for k, v in zip(["prompt", "completion", "solution", "reward", "length"], values)}
            for values in zip(
                prompts, completions, val_solution, val_rewards, [len(v) for v in no_padding_responses]
            )
        ]
        logger.info(
            f"Validation results: {val_results} | len: {len(val_results)} | step: {step}"
        )

        # Currently only avg@N eval is supported
        logger.info(f"Validation avg rewards: {np.mean(val_all_rewards)} | step: {step}")
        save_path = os.path.join(
            self.grpo_config.eval_config.save_eval_result_dir, f"eval_results_step_{step}.json"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if self.grpo_config.eval_config.save_eval_result_data:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(val_results, f, ensure_ascii=False, indent=4, cls=CustomJsonEncoder)
            logger.info(f"Saved eval results to {save_path}")
        return val_results
