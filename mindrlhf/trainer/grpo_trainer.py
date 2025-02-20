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
# limitations under the License.
# ============================================================================
"""
MindRLHF grpo trainer
"""
import time
import os
from dataclasses import dataclass, asdict
import numpy as np
from typing import Callable

import mindspore
import mindspore.common.dtype as mstype
from mindspore import Tensor, context
from mindspore.ops import operations as P
from mindspore.dataset import MindDataset
from mindspore.mindrecord import FileWriter
from mindspore.communication.management import get_rank

from mindformers import logger
from mindformers.trainer.utils import load_distributed_checkpoint

from mindrlhf.configs.grpo_configs import GRPOConfig
from mindrlhf.models.grpo_models import CausalLMHybrid, GRPOModelInfer, GRPOModelTrain
from ..utils.utils import get_valid_length_each_example

RewardFunc = Callable[[list, list], list[float]]

@dataclass
class GRPOData:
    """
    grpo data
    """
    prompt_completion_ids: np.array
    prompts_mask: np.array
    responses_mask: np.array
    ref_per_token_logps: np.array
    advantages: np.array


def get_first_diverge_indices(preferred_comp_ids, disfavored_comp_ids):
    is_equal = Tensor(preferred_comp_ids == disfavored_comp_ids).astype('float32')
    first_diverge_indices = is_equal.sum(axis=1, dtype=mindspore.int32)
    return first_diverge_indices


class GRPOTrainer:
    """
    grpo trainer
    """
    def __init__(self,
                 grpo_config=None,
                 sft_model_config_infer=None,
                 sft_model_config_train=None,
                 ref_model_config=None,
                 reward_funcs=None,
                 reward_weights=None,
                 tokenizer=None):
        self.grpo_config = grpo_config
        self.sft_ckpt_path_infer = sft_model_config_infer.checkpoint_name_or_path
        sft_model_config_infer.checkpoint_name_or_path = None
        self.sft_ckpt_path_train = sft_model_config_train.checkpoint_name_or_path
        sft_model_config_train.checkpoint_name_or_path = None

        self.ref_ckpt_path = ref_model_config.checkpoint_name_or_path
        ref_model_config.checkpoint_name_or_path = None

        self.mind_dataset_dir = grpo_config.mind_dataset_dir
        if self.mind_dataset_dir is not None:
            columns_to_project = ["prompt_ids", "pretrain_ids", "loss_mask"]
            mindspore.dataset.config.set_seed(2023)
            dataset = MindDataset(self.mind_dataset_dir,shuffle=False).project(columns=columns_to_project)
            self.prompt_dataset = dataset
            self.prompt_dataloader = dataset.take()
            bs = grpo_config.chunk_size * sft_model_config_infer.parallel_config.data_parallel
            self.prompt_dataloader = self.prompt_dataloader.batch(batch_size=bs)
            self.prompt_iterator = self.prompt_dataloader.create_tuple_iterator()
        else:
            logger.info("In training stages, there is not dataset for making experience")

        
        self.sft_model_config_infer = sft_model_config_infer
        policy_model = CausalLMHybrid(sft_model_config_infer, self.grpo_config)
        p = policy_model.parameters_dict()
        self.grpo_model_infer = GRPOModelInfer(grpo_config, policy_model)
        self.grpo_model_infer.set_train(False)

        self.ref_model = CausalLMHybrid(ref_model_config, self.grpo_config)
        self.ref_model.model.set_train(False)
        for name, param in self.ref_model.parameters_and_names():
            param.name = name

        self.infer_pp_stage = sft_model_config_infer.parallel_config.pipeline_stage
        self.train_pp_stage = sft_model_config_train.parallel_config.pipeline_stage
        context.set_auto_parallel_context(pipeline_stages=self.train_pp_stage)
        self.sft_model_config_train = sft_model_config_train
        policy_model = CausalLMHybrid(sft_model_config_train, self.grpo_config)
        self.grpo_model_train = GRPOModelTrain(grpo_config, policy_model)
        self.grpo_model_train.set_train(True)
        context.set_auto_parallel_context(pipeline_stages=self.infer_pp_stage)

        # Reward functions
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
            self.reward_weights = np.ones((len(reward_funcs),), dtype=np.float32)

        self.tokenizer = tokenizer

        self.ref_mean = 0
        self.ref_std = 0
        self.cliprange_reward = 10.0
        self.store = []

        self.log_softmax = P.LogSoftmax(axis=-1)
        self.gather = P.GatherD()
        self.unsqueeze = P.ExpandDims()
        self.squeeze = P.Squeeze(axis=-1)
        self.depend = P.Depend()

    def load_checkpoint(self):
        """ load checkpoint """
        load_ckpt_func = load_distributed_checkpoint if self.grpo_config.use_parallel else mindspore.load_checkpoint
        if self.sft_ckpt_path_infer:
            param_dict = load_ckpt_func(self.sft_ckpt_path_infer)
            # ============= different ckpt may not need to replace name =================
            new_param_dict = {'grpo_model.policy_model.model.' + k: v for k, v in param_dict.items()}

            # ===========================================================================
            print(f"begin to load infer policy model from: {self.sft_ckpt_path_infer}", flush=True)
            print("###############")
            print("self.grpo_config.use_parallel:   ", self.grpo_config.use_parallel)
            print(new_param_dict.keys())
            for _, param in self.grpo_model_infer.grpo_model.policy_model.parameters_and_names():
                print(f"infer model para names:   {param.name}")
            param_not_load, ckpt_not_load = mindspore.load_param_into_net(self.grpo_model_infer.grpo_model.policy_model,
                                                                          new_param_dict)
            print(f"param not load: {param_not_load}", flush=True)
            print(f"ckpt not load: {ckpt_not_load}", flush=True)

        if self.sft_ckpt_path_train:
            param_dict = load_ckpt_func(self.sft_ckpt_path_train)
            new_param_dict = {'grpo_model_train.policy_model.model.' + k: v for k, v in param_dict.items()}
            print(f"begin to load train policy model from: {self.sft_ckpt_path_train}", flush=True)
            for _, param in self.grpo_model_train.grpo_model_train.policy_model.parameters_and_names():
                print(f"train model para names:   {param.name}")
            param_not_load, ckpt_not_load = mindspore.load_param_into_net(
                self.grpo_model_train.grpo_model_train.policy_model, new_param_dict)
            print(f"param not load: {param_not_load}", flush=True)
            print(f"ckpt not load: {ckpt_not_load}", flush=True)

        if self.ref_ckpt_path:
            param_dict = load_ckpt_func(self.ref_ckpt_path)
            new_param_dict = {'grpo_model.policy_model.model.' + k: v for k, v in param_dict.items()}
            # ===========================================================================
            print(f"begin to load ref model from: {self.ref_ckpt_path}", flush=True)
            for _, param in self.ref_model.parameters_and_names():
                print(f"ref model para names:   {param.name}")
            param_not_load, ckpt_not_load = mindspore.load_param_into_net(self.ref_model, new_param_dict)
            print(f"param not load: {param_not_load}", flush=True)
            print(f"ckpt not load: {ckpt_not_load}", flush=True)

    def save_checkpoint(self, rank_id=0, steps=0):
        """ save checkpoint """
        if self.grpo_config.save_ckpt_dir:
            print("Save checkpoints in {}".format(self.grpo_config.save_ckpt_dir))
            # 保存train model
            train_save_dir = os.path.join(self.grpo_config.save_ckpt_dir, 'train', f"rank_{rank_id}")
            if not os.path.exists(train_save_dir):
                os.makedirs(train_save_dir)
            grpo_filename = os.path.join(train_save_dir, "policy_model_epoch_{}.ckpt".format(steps))
            mindspore.save_checkpoint(self.grpo_model_train.grpo_model_train.policy_model, grpo_filename,
                                      integrated_save=False)
        else:
            print("There is no checkpoint to save!")

    def save_grpoelement(self, save_path):
        """ save grpo element """
        if save_path:
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
            for ele in self.store:
                count += 1
                print(asdict(ele))
                writer.write_raw_data([asdict(ele)])
            print("Total number of samples: {}".format(count))
            writer.commit()
            print("grpo element saved! Output file refer: {}".format(save_path))

    def push_to_store(self, data):
        self.store = data

    def generate(self, input_ids):
        """ Policy model generates responses for a batch of prompts. """
        input_ids_numpy = input_ids.asnumpy()
        _, max_valid_length = get_valid_length_each_example(input_ids_numpy, self.grpo_model_infer.grpo_model.pad_token_id)  # get valid length and max length in a batch

        generate_begin_time = time.time()
        outputs = self.grpo_model_infer.grpo_model.policy_model.model.generate(
            input_ids_numpy[:, :max_valid_length], max_new_tokens=self.grpo_config.max_decode_length)

        print("Generating elapsed time: ", time.time() - generate_begin_time)

        input_ids_list = input_ids_numpy.tolist()
        num_sample = len(input_ids_list)
        left_padding_prompts = np.ones((num_sample, self.grpo_config.max_prompt_length)) * self.grpo_config.pad_token_id  # 初始化存储prompt的数组，序列长度最大为max_prompt_length
        right_padding_responses = np.ones((num_sample, self.grpo_config.max_decode_length)) * self.grpo_config.pad_token_id  # 初始化存储response的数组，序列长度最大为max_decode_length
        prompt_len = (np.array(input_ids_list) != self.grpo_config.pad_token_id).astype(int).sum(1)  # 计算每个样本的prompt长度（不包含padding token)

        for i in range(num_sample):
            response = outputs[i][prompt_len[i]: prompt_len[i]+self.grpo_config.max_decode_length]  # 只包含response, 范围是从 "prompt结束位置" 到 "prompt结束位置+最大生成长度"
            right_padding_responses[i, :len(response)] = response

            left_padding_prompts[i, self.grpo_config.max_prompt_length-prompt_len[i]:] = input_ids_list[i][:prompt_len[i]]  # 整个batch的样本右对齐（左侧进行padding）

        responses_mask = (right_padding_responses != self.grpo_config.pad_token_id).astype(np.int32)
        prompts_mask = (left_padding_prompts != self.grpo_config.pad_token_id).astype(np.int32)

        return right_padding_responses.astype(np.int32), responses_mask, left_padding_prompts.astype(np.int32), prompts_mask

    def get_batch(self):
        """ get batch """
        try:
            batch = next(self.prompt_iterator)
        except StopIteration:
            mindspore.dataset.config.set_seed(2023)
            self.prompt_iterator = self.prompt_dataloader.create_tuple_iterator()
            batch = next(self.prompt_iterator)
        return batch
        
    def logprobs_of_labels(self, logits, samples):
        """
        Calculate the log value of the label
        """
        logits = logits[:, :-1, :]  # [bs, seq_len-1, vocab_size]
        samples = samples[:, 1:]  # [bs, seq_len-1]
        logprobs = self.logsoftmax_1(logits)  # [bs, seq_len-1, vocab_size]
        logprobs = self.squeeze_no_shard(
            self.gatherd(logprobs, -1, self.unsqueeze(samples, -1))
        )  # [bs, seq_len-1]

        return logprobs  # [bs, seq_len-1]    

    def make_experience(self, num_generations: int = 16, rank_id: int = 0):
        """
        make experience
        """
        context.set_auto_parallel_context(pipeline_stages=self.infer_pp_stage)
        ep_begin_time = time.time()
        print("Make experience begin at {} \n------------------------------- "
              .format(time.strftime('%H:%M:%S', time.localtime(ep_begin_time))), flush=True)

        self.grpo_model_infer.grpo_model.policy_model.model.set_train(False)
        self.ref_model.model.set_train(False)

        batch = self.get_batch()

#TODO Tiancheng:数据集格式处理和trl对齐;
#TODO Tiancheng:易用性问题，最好改成在线处理json格式

        prompt_tensors = Tensor(batch[0], mstype.int32)
        solution_ids = Tensor(batch[1], mstype.int32).asnumpy()
        solution = self.tokenizer.decode(solution_ids, skip_special_tokens=True)
        reward_kwargs = {"solution": solution}

        n_questions = batch[0].shape[0]
        all_rewards = np.zeros((num_generations, n_questions), dtype=np.float32)
        all_prompt_completion_ids = np.zeros((num_generations, n_questions, self.grpo_config.seq_length), dtype=np.int32)
        all_prompts_mask = np.zeros((num_generations, n_questions, self.grpo_config.seq_length), dtype=np.int32)
        all_responses_mask = np.zeros((num_generations, n_questions, self.grpo_config.seq_length), dtype=np.int32)
        all_ref_per_token_logps = np.zeros((num_generations, n_questions, self.grpo_config.seq_length-1), dtype=np.float32)
#TODO Tiancheng:去掉for循环改成多batch generation，拷贝输入；
        # TODO: 对比for循环是否去掉的性能
        for idx in range(num_generations):
            start_time = time.time()
            print("generation start at {}-------------------------------".format(time.strftime('%H:%M:%S', time.localtime(start_time))), flush=True)
            # right_padded_responses: [n_questions, max_decode_length]
            # responses_mask: [n_questions, max_decode_length]
            # left_padded_prompts: [n_questions, seq_length - max_decode_length]
            # prompts_mask: [n_questions, seq_length - max_decode_length]
            #TODO Tiancheng: 看一下能对齐到第几个token
            right_padding_responses, responses_mask, left_padding_prompts, prompts_mask = self.generate(prompt_tensors)

            is_eos = right_padding_responses == self.tokenizer.eos_token_id
            eos_idx = np.full(is_eos.shape[0], is_eos.shape[1], dtype=int)
            eos_idx[is_eos.any(axis=1)] = np.argmax(is_eos.astype(int), axis=1)[is_eos.any(axis=1)]
            sequence_indices = np.arange(is_eos.shape[1])
            responses_eos_mask = (sequence_indices <= eos_idx[:, None]).astype(int)
            responses_mask *= responses_eos_mask
            # Todo Tiancheng: 换成ms算子（优先级低）
            # TODO Tiancheng: 补充精度对比实验记录

            end_time = time.time()
            print("generate end at {}, elapsed time {}-------------------------------".format(time.strftime('%H:%M:%S', time.localtime(end_time)), end_time - start_time), flush=True)

            prompt_completion_ids = np.concatenate((left_padding_prompts, right_padding_responses), axis=1)  # [n_questions, seq_length]
            prompts_mask = np.concatenate((prompts_mask, np.zeros_like(responses_mask, dtype=np.int32)), axis=1)  # [n_questions, seq_length]
            responses_mask = np.concatenate((np.zeros_like(left_padding_prompts, dtype=np.int32), responses_mask), axis=1)  # [n_questions, seq_length]
            attention_mask = prompts_mask + responses_mask  # [n_questions, seq_length]

            start_time = time.time()
            print("reference model start at {}-------------------------------".format(time.strftime('%H:%M:%S', time.localtime(start_time))), flush=True)
            prompt_completion_ids_tensor = Tensor(prompt_completion_ids, dtype=mindspore.int32)  # [n_questions, seq_length]
            attention_mask_tensor = Tensor(attention_mask, dtype=mindspore.int32)  # [n_questions, seq_length]
            #TODO 把左右padding移到右边对精度
            #TODO 所有tensor都要在最后.asnumpy否则一直累积占用显存，参考PPO
            ref_per_token_logps = self.ref_model(prompt_completion_ids_tensor, attention_mask_tensor, samples=prompt_completion_ids_tensor, is_ref=True)  # [n_questions, seq_length-1]
            ref_per_token_logps = self.logprobs_of_labels(ref_per_token_logps, prompt_completion_ids_tensor)
            prompt_completion_ids_tensor = prompt_completion_ids_tensor.asnumpy()
            attention_mask_tensor = attention_mask_tensor.asnumpy()
            end_time = time.time()
            print("reference model end at {}, elapsed time {}-------------------------------".format(time.strftime('%H:%M:%S', time.localtime(end_time)), end_time - start_time), flush=True)

            start_time = time.time()
            print("calculate reward start at {}-------------------------------".format(time.strftime('%H:%M:%S', time.localtime(start_time))), flush=True)
            prompts = self.tokenizer.decode(left_padding_prompts, skip_special_tokens=True)
            completions = self.tokenizer.decode(right_padding_responses, skip_special_tokens=True)

            print("prompts: \n", prompts)
            print("completions: \n", completions)

            rewards_per_func = np.zeros((n_questions, len(self.reward_funcs)), dtype=np.float32)
            for i, reward_func in enumerate(self.reward_funcs):
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)  # [n_questions]
                rewards_per_func[:, i] = np.array(output_reward_func, dtype=np.float32) # [n_questions]
            rewards = (rewards_per_func * self.reward_weights[np.newaxis, :]).sum(axis=1)
            end_time = time.time()
            print("calculate reward end at {}, elapsed time {}-------------------------------".format(time.strftime('%H:%M:%S', time.localtime(end_time)), end_time - start_time), flush=True)


            all_rewards[idx, :] = np.array(rewards, dtype=np.float32)
            all_prompt_completion_ids[idx, :, :] = prompt_completion_ids
            all_prompts_mask[idx, :, :] = prompts_mask
            all_responses_mask[idx, :, :] = responses_mask
            all_ref_per_token_logps[idx, :, :] = ref_per_token_logps
# TODO: 定义时shape改一下，去掉transpose
        all_rewards = all_rewards.transpose((1, 0))  # [n_questions, num_generations]
        all_prompt_completion_ids = all_prompt_completion_ids.transpose((1, 0, 2))  # [n_questions, num_generations, seq_length]
        all_prompts_mask = all_prompts_mask.transpose((1, 0, 2))  # [n_questions, num_generations, seq_length]
        all_responses_mask = all_responses_mask.transpose((1, 0, 2))  # [n_questions, num_generations, seq_length]
        all_ref_per_token_logps = all_ref_per_token_logps.transpose((1, 0, 2))  # [n_questions, num_generations, seq_length-1]
# TODO: np和torch的std是否对齐
        mean_grouped_rewards = all_rewards.mean(axis=1)  # [n_questions]
        std_grouped_rewards = all_rewards.std(axis=1, ddof=1)  # [n_questions]
        advantages = (all_rewards - mean_grouped_rewards[:, np.newaxis]) / (std_grouped_rewards[:, np.newaxis] + 1e-4)  # [n_questions, num_generations]

        grpo_rl_elements = []
# TODO: 除了generation其他都代模型跑，对精度
        for i in range(n_questions):
            for j in range(num_generations):
                grpodata = GRPOData(
                    prompt_completion_ids=all_prompt_completion_ids[i, j].astype(np.int32),
                    prompts_mask=all_prompts_mask[i, j].astype(np.int32),
                    responses_mask=all_responses_mask[i, j].astype(np.int32),
                    ref_per_token_logps=all_ref_per_token_logps[i, j].astype(np.float32),
                    advantages=advantages[i, j:j+1].astype(np.float32)
                    )
                grpo_rl_elements.append(grpodata)
        self.push_to_store(grpo_rl_elements)
        end_time = time.time()
        print("Make experience, end at {}, elapsed time {} \n------------------------------- ".format(time.strftime('%H:%M:%S', time.localtime(end_time)), end_time - ep_begin_time), flush=True)
        if self.grpo_config.save_data_file:
            if rank_id % 8 == 0:
                self.save_grpoelement(self.grpo_config.save_data_file)
        print('generate over')

    def train(self, grpo_with_grad, dataset):
        """train model"""
        context.set_auto_parallel_context(pipeline_stages=self.train_pp_stage)
        sink_process = mindspore.data_sink(grpo_with_grad, dataset, sink_size=self.grpo_config.sink_size)
        steps = dataset.dataset_size // self.grpo_config.sink_size
        print(f"dataset size is {dataset.dataset_size}, sink size is {self.grpo_config.sink_size},"
              f"total steps is {steps}")
        for batch in range(steps):
            ep_begin_time = time.time()
            out = sink_process()
            end_time = time.time()
            print("steps {}, end at {}, elapsed time {} \n------------------------------- "
                  .format(batch + 1, time.strftime('%H:%M:%S', time.localtime(end_time)), end_time - ep_begin_time),
                  flush=True)
            print(" loss: {} | lr: {} | is overflow: {} | loss scale: {}"
                  .format(out[0], out[1], out[2], out[3]), flush=True)

        print('train over')

    def pre_run(self, stage_name='', input_data=None):
        """
        pre run
        """
        if self.grpo_config.only_save_strategy:
            if context.get_auto_parallel_context("parallel_mode") in ['semi_auto_parallel', 'auto_parallel',
                                                                      'hybrid_parallel']:
                fake_data = input_data['prompt_completion_ids']                
                stage_name = 'generate'
                context.set_auto_parallel_context(
                    strategy_ckpt_config={
                        "save_file":
                            f"./strategy/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt"})
                self.grpo_model_infer.grpo_model.policy_model.model.set_train(False)
                self.grpo_model_infer.compile(**input_data)

                context.set_auto_parallel_context(
                    strategy_ckpt_config={
                        "save_file":
                            f"./strategy/{stage_name}_ref_strategy/strategy_{get_rank()}.ckpt"})
                self.ref_model.compile(fake_data, samples=fake_data)

                stage_name = 'train'
                context.set_auto_parallel_context(
                    strategy_ckpt_config={
                        "save_file":
                            f"./strategy/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt"})
                self.grpo_model_train.grpo_model_train.policy_model.model.set_train(True)
                self.grpo_model_train.compile(**input_data)
                
                logger.info("Running only save strategy finish, system exit.")
                exit(0)
            else:
                logger.info("only_save_strategy is True, but stand_alone and data_parallel mode"
                            "do not have strategy file, system exit!")
                exit(0)


if __name__ == "__main__":
    context.set_context(device_target='Ascend', device_id=1, mode=mindspore.GRAPH_MODE)

    trainer = GRPOTrainer(grpo_config=GRPOConfig)
    trainer.make_experience(num_generations=2)
