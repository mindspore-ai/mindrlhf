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

# python
import time
import numpy as np
from dataclasses import asdict

# mindspore
import mindspore
import mindspore as ms
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.communication import get_rank
from mindspore.mindrecord import FileWriter
from mindspore.dataset import MindDataset
from mindspore import communication as D

# mindformers
from mindformers import logger

# mindrlhf
from mindrlhf.reward.reward_fn import accuracy_reward, format_reward
from mindrlhf.configs.grpo_configs import GRPOConfig
from mindrlhf.utils import transfer_from_str_to_bool
from mindrlhf.models.qwen2.qwen2_tokenizer import Qwen2Tokenizer

# mindrlhf
from mindrlhf.worker.worker import GRPOData
from mindrlhf.worker.train_worker import TrainWorker
from mindrlhf.worker.infer_worker import InferWorker
from mindrlhf.worker.ref_worker import RefWorker
from mindrlhf.worker.transform_worker import TransformWorker


class GRPOTrainer:
    def __init__(self, args=None):
        self.args = args
        self._init_grpo_configs(args)
        self._init_reward_fn()

        logger.info("GRPOTrainer: start init workers")
        self.infer = InferWorker(grpo_config=self.grpo_config,
                                 sft_path_infer=self.sft_path_infer,
                                 args=self.args)
        # grpo_config infer 和 train 共用
        self.grpo_config = self.infer.get_updated_grpo_config()
        self.infer_dp = self.infer.get_infer_dp()
        self._init_grpo_infer_dataset()

        self.ref = RefWorker(grpo_config=self.grpo_config,
                             sft_path_infer=self.sft_path_infer,
                             args=self.args)
        self.train = TrainWorker(grpo_config=self.grpo_config,
                                 sft_path_train=self.sft_path_train,
                                 args=self.args)
        logger.info("GRPOTrainer: finish init workers")

        self._compile()
        self._load_checkpoint()
        self.transform = TransformWorker(self.grpo_config, self.train.model(),
                                         self.infer.model(), self.ref.model())

    def _init_grpo_configs(self, args):
        logger.info("GRPOTrainer: _init_grpo_configs {args} in main task")
        use_parallel = transfer_from_str_to_bool(args.use_parallel)
        # init grpo config
        grpo_config = GRPOConfig()
        grpo_config.mind_dataset_dir = args.mind_dataset_dir
        grpo_config.save_data_file = args.save_data_file
        grpo_config.save_ckpt_dir = args.save_ckpt_dir
        grpo_config.align_type = "rlhf_stages"
        grpo_config.use_parallel = use_parallel

        # for worker
        self.tokenizer = Qwen2Tokenizer(
            args.vocab_path, args.merges_file_path, add_bos_token=False, add_eos_token=False)
        self.grpo_config = grpo_config
        self.args.use_parallel = use_parallel
        self.use_parallel = use_parallel
        self.sft_path_infer = args.sft_path_infer
        self.sft_path_train = args.sft_path_train

    def _init_reward_fn(self, reward_weights=None):
        logger.info("GRPOTrainer: _init_reward_fn")
        reward_funcs = [accuracy_reward, format_reward]
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        self.reward_funcs = reward_funcs

        # Reward weights
        if reward_weights is not None:
            if len(reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(len(reward_weights))}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = np.array(reward_weights, dtype=np.float32)
        else:
            self.reward_weights = np.ones(
                (len(reward_funcs),), dtype=np.float32)

    def _init_grpo_infer_dataset(self):
        '''
        Build dataset for generating.
        '''
        logger.info(
            "GRPOTrainer: _init_grpo_infer_dataset, dataset dir {self.mind_dataset_dir}")
        self.mind_dataset_dir = self.grpo_config.mind_dataset_dir
        if self.mind_dataset_dir is not None:
            columns_to_project = ["prompt_ids", "pretrain_ids", "loss_mask"]
            ms.dataset.config.set_seed(2023)
            dataset = MindDataset(self.mind_dataset_dir).project(columns=columns_to_project)
            self.prompt_dataset = dataset
            self.prompt_dataloader = dataset.take()
            bs = self.grpo_config.chunk_size * self.infer_dp
            self.prompt_dataloader = self.prompt_dataloader.batch(batch_size=bs, drop_remainder=True)
            self.prompt_iterator = self.prompt_dataloader.create_tuple_iterator()
            self.step_num = self.prompt_dataset.get_dataset_size() // self.prompt_dataset.get_batch_size() // self.grpo_config.num_rollouts
        else:
            logger.info("In main task, there is not dataset for making experience")

    def _compile(self):
        self._make_experience(num_rollouts=1, num_generations=1, pre_run_flag=True)
        self.train.compile()

    def _load_checkpoint(self):
        self.infer.load_checkpoint()
        self.ref.load_checkpoint()
        self.train.load_checkpoint()

    def _get_batch(self):
        """ get batch """
        try:
            batch = next(self.prompt_iterator)
        except StopIteration:
            ms.dataset.config.set_seed(2023)
            self.prompt_iterator = self.prompt_dataloader.create_tuple_iterator()
            batch = next(self.prompt_iterator)
        return batch

    def _split_for_data_parallel(self, batch_inputs, data_parallel_size):
        """
        split batch_inputs for data parallel
        """
        world_size = D.get_group_size()
        rank_id = D.get_rank()
        split_size = (batch_inputs.shape[0] // data_parallel_size)
        all_other_group_size = world_size // data_parallel_size

        dp_rank_id = rank_id // all_other_group_size

        start = dp_rank_id * split_size
        stop = (dp_rank_id + 1) * split_size
        batch_inputs_for_this_rank = batch_inputs[start:stop]
        return batch_inputs_for_this_rank

    def _make_experience(self, num_rollouts: int = 1, num_generations: int = 16, pre_run_flag=False):
        def _remove_right_padding(token_ids, padding_token=0):
            """ remove_right_padding """
            trimmed_sequences = []
            for seq in token_ids:
                # 将序列转换为列表以处理不同输入类型（如numpy数组）
                seq_list = list(seq)
                # 逆序查找第一个非填充标记的位置
                last_non_pad = next((i for i in reversed(
                    range(len(seq_list))) if seq_list[i] != padding_token), None)
                # 截断右侧填充
                if last_non_pad is not None:
                    trimmed_sequences.append(seq_list[:last_non_pad + 1])
                else:
                    trimmed_sequences.append([])  # 全为填充时返回空列表
            return trimmed_sequences

        def _remove_left_padding(token_ids, padding_token=0):
            """ remove_left_padding """
            trimmed_sequences = []
            for seq in token_ids:
                # 将序列转换为列表以处理不同输入类型（如numpy数组）
                seq_list = list(seq)
                # 顺查找第一个非填充标记的位置
                last_non_pad = next(
                    (i for i in range(len(seq_list)) if seq_list[i] != padding_token), None)
                # 截断左侧填充
                if last_non_pad is not None:
                    trimmed_sequences.append(seq_list[last_non_pad:])
                else:
                    trimmed_sequences.append([])  # 全为填充时返回空列表
            return trimmed_sequences

        def _construct_inputs_for_ref_model(right_padding_responses, responses_mask, left_padding_prompts, prompts_mask):
            is_eos = right_padding_responses == self.tokenizer.eos_token_id
            eos_idx = np.full(is_eos.shape[0], is_eos.shape[1], dtype=int)
            eos_idx[is_eos.any(axis=1)] = np.argmax(is_eos.astype(int), axis=1)[is_eos.any(axis=1)]
            sequence_indices = np.arange(is_eos.shape[1])
            responses_eos_mask = (sequence_indices <= eos_idx[:, None]).astype(int)
            responses_mask = responses_mask * responses_eos_mask

            prompt_completion_ids = np.concatenate((left_padding_prompts, right_padding_responses), axis=1)
            prompts_mask = np.concatenate((prompts_mask, np.zeros_like(responses_mask, dtype=np.int32)), axis=1)
            responses_mask = np.concatenate(
                (np.zeros_like(left_padding_prompts, dtype=np.int32), responses_mask), axis=1)
            attention_mask = prompts_mask + responses_mask

            # Generate outputs.
            prompt_completion_ids_tensor = Tensor(prompt_completion_ids, dtype=mindspore.int32)
            attention_mask_tensor = Tensor(attention_mask, dtype=mindspore.int32)
            return prompt_completion_ids_tensor, attention_mask_tensor, responses_mask, prompts_mask

        ep_begin_time = time.time()
        logger.info("Make experience begin at {} \n------------------------------- "
                    .format(time.strftime('%H:%M:%S', time.localtime(ep_begin_time))))
        logger.info(f"Generate {num_generations} times")
        self.infer.grpo_model_infer.grpo_model.policy_model.model.set_train(False)
        self.ref.ref_model.model.set_train(False)
        
        
        grpo_rl_elements = []
        all_mean_grouped_rewards = []
        all_elements_compeltion_len = []
        
        for rollout in range(num_rollouts):
            batch = self._get_batch()
            prompt_tensors_full = Tensor(batch[0], mstype.int32)
            prompt_tensors = self._split_for_data_parallel(prompt_tensors_full, self.infer_dp)
            solution_ids = Tensor(batch[1], mstype.int32).asnumpy()
            solution_ids = _remove_right_padding(solution_ids, padding_token=self.grpo_config.pad_token_id)
            solution = self.tokenizer.decode(solution_ids, skip_special_tokens=True)
            for i in range(len(solution)):
                solution[i] = "$" + solution[i] + "$"
            reward_kwargs = {"solution": solution}
            logger.info(f"solution: {solution}")

            n_questions = batch[0].shape[0]
            all_rewards = np.zeros((num_generations, n_questions), dtype=np.float32)
            all_prompt_completion_ids = np.zeros(
                (num_generations, n_questions, self.grpo_config.seq_length), dtype=np.int32)
            all_prompts_mask = np.zeros((num_generations, n_questions, self.grpo_config.seq_length), dtype=np.int32)
            all_responses_mask = np.zeros((num_generations, n_questions, self.grpo_config.seq_length), dtype=np.int32)
            all_ref_per_token_logps = np.zeros(
                (num_generations, n_questions, self.grpo_config.seq_length-1), dtype=np.float32)

            for idx in range(num_generations):
                self.infer.load()
                # Step 1: generate responses and masks.
                start_time = time.time()
                logger.info("generation start at {}-------------------------------".format(
                    time.strftime('%H:%M:%S', time.localtime(start_time))))

                results = self.infer.generate(prompt_tensors)

                self.infer.offload()
                logger.info("model_infer offload")

                self.ref.load()
                logger.info("ref_model load")

                end_time = time.time()
                logger.info("generate end at {}, elapsed time {}-------------------------------".format(
                    time.strftime('%H:%M:%S', time.localtime(end_time)), end_time - start_time))

                logger.info(f"generate sequence results is {results} type {type(results)}")
                for i, ele in enumerate(results):
                    logger.info(f"{i} result is {ele}")

                right_padding_responses, responses_mask, left_padding_prompts, prompts_mask = self.infer.post_process_infer_outputs(
                    results)

                if pre_run_flag and idx == 0:
                    # only generate once
                    self.infer.generate_strategy()

                # responses_mask will be updated before ref model infer.
                prompt_completion_ids_tensor, attention_mask_tensor, responses_mask, prompts_mask = _construct_inputs_for_ref_model(
                    right_padding_responses, responses_mask, left_padding_prompts, prompts_mask)

                # Step 2: run ref model.
                start_time = time.time()
                logger.info("reference model start at {}-------------------------------".format(
                    time.strftime('%H:%M:%S', time.localtime(start_time))))

                ref_per_token_logps = self.ref.compute_ref_log_prob(
                    prompt_completion_ids_tensor, attention_mask_tensor, samples=prompt_completion_ids_tensor, save_strategy=pre_run_flag)

                end_time = time.time()
                logger.info("reference model end at {}, elapsed time {}-------------------------------".format(
                    time.strftime('%H:%M:%S', time.localtime(end_time)), end_time - start_time))
                logger.info(f"Ref log probs {ref_per_token_logps}")

                self.ref.offload()
                logger.info("ref_model offload")

                # Step 3: calculate reward.
                start_time = time.time()
                logger.info("calculate reward start at {}-------------------------------".format(
                    time.strftime('%H:%M:%S', time.localtime(start_time))))

                logger.info(
                    f"left_padding_prompts is {type(left_padding_prompts)}")
                no_padding_prompts = _remove_left_padding(left_padding_prompts, padding_token=self.grpo_config.pad_token_id)
                no_padding_responses = _remove_right_padding(
                    right_padding_responses, padding_token=self.grpo_config.pad_token_id)
                prompts = self.tokenizer.decode(no_padding_prompts, skip_special_tokens=True)
                completions = self.tokenizer.decode(no_padding_responses, skip_special_tokens=True)

                logger.info(f"prompts: \n {prompts}", )
                logger.info(f"completions: \n {completions}")
                all_elements_compeltion_len.extend([len(com) for com in completions])
                
                mean_len = np.array([len(com) for com in completions]).mean()
                logger.info(f"mean completions.length: \n {mean_len}")

                rewards_per_func = np.zeros((n_questions, len(self.reward_funcs)), dtype=np.float32)
                for i, reward_func in enumerate(self.reward_funcs):
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    rewards_per_func[:, i] = np.array(output_reward_func, dtype=np.float32)
                rewards = (rewards_per_func * self.reward_weights[np.newaxis, :]).sum(axis=1)
                logger.info(f"precision rewards are {rewards}")

                end_time = time.time()
                logger.info("calculate reward end at {}, elapsed time {}-------------------------------".format(
                    time.strftime('%H:%M:%S', time.localtime(end_time)), end_time - start_time))

                all_rewards[idx, :] = np.array(rewards, dtype=np.float32)
                all_prompt_completion_ids[idx, :, :] = prompt_completion_ids_tensor.asnumpy()
                all_prompts_mask[idx, :, :] = prompts_mask
                all_responses_mask[idx, :, :] = responses_mask
                all_ref_per_token_logps[idx, :, :] = ref_per_token_logps

            all_rewards = all_rewards.transpose((1, 0))
            logger.info(f"loaded_all_rewards: {all_rewards}")
            all_prompt_completion_ids = all_prompt_completion_ids.transpose((1, 0, 2))
            all_prompts_mask = all_prompts_mask.transpose((1, 0, 2))
            all_responses_mask = all_responses_mask.transpose((1, 0, 2))
            all_ref_per_token_logps = all_ref_per_token_logps.transpose((1, 0, 2))

            mean_grouped_rewards = all_rewards.mean(axis=1)
            std_grouped_rewards = all_rewards.std(axis=1, ddof=1)
            advantages = (all_rewards - mean_grouped_rewards[:, np.newaxis]) / (std_grouped_rewards[:, np.newaxis] + 1e-4)
            logger.info(f"mean_grouped_rewards: \n {mean_grouped_rewards}")
            all_mean_grouped_rewards.extend((mean_grouped_rewards.tolist()))

            for i in range(n_questions):
                for j in range(num_generations):
                    grpodata = GRPOData(
                        prompt_completion_ids=all_prompt_completion_ids[i, j].astype(np.int32),
                        prompts_mask=all_prompts_mask[i, j].astype(np.int32),
                        responses_mask=all_responses_mask[i, j].astype(np.int32),
                        ref_per_token_logps=all_ref_per_token_logps[i, j].astype(np.float32),
                        advantages=advantages[i, j:j+1].astype(np.float32)
                    )
                    logger.info(
                        f"precision grpo data is {all_prompt_completion_ids}\n=== {all_prompts_mask}\n=== {all_responses_mask}\n=== {all_ref_per_token_logps}\n=== {advantages}\n")
                    grpo_rl_elements.append(grpodata)

        logger.info(f"grpo_rl_elements.length: {len(grpo_rl_elements)}")
        print(f"all_elements_compeltion_len mean: {np.mean(all_elements_compeltion_len)}")
        self.train.push_to_store(grpo_rl_elements)
        logger.info(f"all_mean_grouped_rewards: {all_mean_grouped_rewards}")
        logger.info(f"Avg scores:\n {np.mean(np.array(all_mean_grouped_rewards))}")
        
        end_time = time.time()
        logger.info("Make experience, end at {}, elapsed time {} ------------------------------- ".format(
            time.strftime('%H:%M:%S', time.localtime(end_time)), end_time - ep_begin_time))
        if self.grpo_config.save_data_file:
            if get_rank() % 8 == 0:
                self._save_grpoelement(self.grpo_config.save_data_file)
        logger.info('generate over')

    def _print_data_str(self, data, name):
        decoded_str = self.tokenizer.decode(data)
        logger.info(f"{name} str is {decoded_str}")

    def _save_grpoelement(self, save_path):
        """ save grpo element """
        if save_path:
            logger.info(f"start save grpo {save_path}")
            schema = {
                "prompt_completion_ids": {"type": "int32", "shape": [-1]},
                "prompts_mask": {"type": "int32", "shape": [-1]},
                "responses_mask": {"type": "int32", "shape": [-1]},
                "ref_per_token_logps": {"type": "float32", "shape": [-1]},
                "advantages": {"type": "float32", "shape": [-1]},
            }

            writer = FileWriter(file_name=save_path, shard_num=1, overwrite=True)
            writer.add_schema(schema)
            count = 0
            for ele in self.train.store:
                count += 1
                logger.info(f"add data element {asdict(ele)}")
                writer.write_raw_data([asdict(ele)])
            writer.commit()
            logger.info(f"end save grpo {save_path}")

    def run_grpo_train(self):
        """
        Main entry of MindRLHF GRPO training.
        """
        logger.info(
            f"Start training epoch num:{self.grpo_config.epochs}, step num:{self.step_num}, generation num:{self.grpo_config.num_generations}")
        np.set_printoptions(threshold=1024)
        """
        第一次执行前, load ckpt后参数在host上, 在网络第一次执行时会将参数自动加载到device上, 不需要手动load/offload
        """
        for n in range(self.grpo_config.epochs):
            for i in range(self.step_num):

                step_begin_time = time.time()
                logger.info("step begin at {} \n------------------------------- "
                    .format(time.strftime('%H:%M:%S', time.localtime(step_begin_time))))

                logger.info(f"epoch: {n}, step: {i}")
                self._make_experience(num_rollouts=self.grpo_config.num_rollouts, num_generations=self.grpo_config.num_generations)
                self.train.load_optimizer()
                self.train.load_model()

                self.train.train()
                self.train.offload_optimizer()

                # load for reshard
                self.infer.load()
                self.ref.load()
                self.transform.reshard_params(i)

                self.train.offload_model()
                self.ref.offload()
                
                step_end_time = time.time()
                logger.info("step end at  {}, elapsed time {} \n------------------------------- ".format(
                    time.strftime('%H:%M:%S', time.localtime(step_end_time)), step_end_time - step_begin_time))

        # save checkpoint
        self.train.load_model()
        self.train.save_checkpoint(
            rank_id=get_rank(), steps=self.grpo_config.epochs)
        logger.info("run grpo train end")
