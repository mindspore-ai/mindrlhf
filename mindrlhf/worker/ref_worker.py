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
""" Reference Worker """

# python
import time
import numpy as np

# mindspore
import mindspore as ms
from mindspore.communication import get_rank
from mindspore import context

# mindformers
from mindformers import MindFormerConfig
from mindformers.trainer.utils import load_distributed_checkpoint
from mindformers import LlamaConfig
from mindformers import logger

# mindrlhf
from mindrlhf.models.grpo_models import CausalLMHybrid
from mindrlhf.worker.worker import Worker
from mindrlhf.utils import print_perf_stat


class RefWorker(Worker):
    '''
    This class generates responses.
    '''

    def __init__(self, grpo_config, sft_path_infer, args):
        super().__init__()
        logger.info("init RefWorker")
        self.use_parallel = grpo_config.use_parallel
        ref_config = MindFormerConfig(sft_path_infer)
        ref_config.use_parallel = args.use_parallel
        ref_config.model.model_config.parallel_config = ref_config.parallel_config
        ref_config.model.model_config.use_past = False
        ref_model_config = LlamaConfig(**ref_config.model.model_config)
        ref_model_config.checkpoint_name_or_path = args.load_ref_checkpoint
        ref_model_config.model_name = "llama"

        self.ref_model_config = ref_model_config
        self.ref_ckpt_path = ref_model_config.checkpoint_name_or_path
        ref_model_config.checkpoint_name_or_path = None

        self.ref_model = CausalLMHybrid(ref_model_config, grpo_config)
        self.ref_model.model.set_train(False)
        for name, param in self.ref_model.parameters_and_names():
            param.name = name
        self.on_device = True
        self.grpo_config = grpo_config
        self.ref_pp_stage = ref_config.parallel_config.pipeline_stage or 1
        self.ref_dp = ref_config.parallel_config.data_parallel
        self.save_strategy_dir = grpo_config.save_strategy_dir

    def model(self):
        return self.ref_model

    def get_ref_dp(self):
        return self.ref_dp

    def compile(self):
        """ compile and save strategy """
        self.ref_model.model.set_train(False)
        context.set_auto_parallel_context(pipeline_stages=self.ref_pp_stage)
        total_ref_model_batch_size = self.grpo_config.ref_model_batch_size * self.ref_dp
        fake_data = ms.Tensor(shape=(total_ref_model_batch_size, self.grpo_config.seq_length),
                              dtype=ms.int32)
        start_time = time.time()
        stage_name = 'infer'
        context.set_auto_parallel_context(
            strategy_ckpt_config={
                "save_file":
                    f"{self.save_strategy_dir}/{stage_name}_ref_strategy/strategy_{get_rank()}.ckpt"})
        self.ref_model.compile(fake_data, samples=fake_data, is_ref=False)
        stage_name = 'other'
        context.set_auto_parallel_context(
            strategy_ckpt_config={
                "save_file":
                    f"{self.save_strategy_dir}/{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt"})
        end_time = time.time()
        print_perf_stat(start_time, end_time, "reference model compile")

    def compute_ref_log_prob(self, prompt_completion_ids_tensor, samples):
        np.set_printoptions(threshold=1024)
        context.set_auto_parallel_context(pipeline_stages=self.ref_pp_stage)
        logger.info(f"precision refmodel inputs are {prompt_completion_ids_tensor}, {samples}")

        ref_per_token_logps = self.ref_model(prompt_completion_ids_tensor,
                                             samples=samples, is_ref=False)

        logger.info(f"ref_logprobs precision is {ref_per_token_logps}")
        return ref_per_token_logps

    def offload(self):
        """ offload ref model """
        if self.on_device is False:
            return
        logger.info(f'before offload ref model {ms.hal.memory_stats()}')
        start_time = time.time()
        for param in self.ref_model.get_parameters(expand=True):
            # pylint: disable=W0212
            param._offload()
        end_time = time.time()
        print_perf_stat(start_time, end_time, "offload ref model")
        logger.info(f'after offload ref model {ms.hal.memory_stats()}')
        self.on_device = False

    def load(self):
        """ load ref model """
        if self.on_device:
            return
        logger.info(f'before load ref model {ms.hal.memory_stats()}')
        start_time = time.time()
        for param in self.ref_model.get_parameters(expand=True):
            # pylint: disable=W0212
            param._load()
        end_time = time.time()
        print_perf_stat(start_time, end_time, "load ref model")
        logger.info(f'after load ref model {ms.hal.memory_stats()}')
        self.on_device = True

    def load_checkpoint(self):
        """ load checkpoint """
        load_ckpt_func = load_distributed_checkpoint if self.use_parallel else ms.load_checkpoint
        logger.info(f"self.grpo_config.use_parallel is {self.use_parallel} {load_ckpt_func}")
        if self.ref_ckpt_path:
            self.on_device = True
            param_dict = load_ckpt_func(self.ref_ckpt_path)
            new_param_dict = {'model.' + k: v for k, v in param_dict.items()}
            # ===========================================================================
            logger.info(f"begin to load ref model from: {self.ref_ckpt_path}")
            for _, param in self.ref_model.parameters_and_names():
                logger.info(f"ref model para names:   {param.name}")
            param_not_load, ckpt_not_load = ms.load_param_into_net(self.ref_model, new_param_dict)
            logger.info(f"param not load: {param_not_load}")
            logger.info(f"ckpt not load: {ckpt_not_load}")
