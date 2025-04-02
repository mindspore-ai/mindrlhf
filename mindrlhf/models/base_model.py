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
"""
MindRLHF Base Model
"""
import importlib
import mindspore.nn as nn
from mindformers import LlamaForCausalLM
from mindformers.models.glm2 import ChatGLM2ForConditionalGeneration
from mindformers.models.gpt2 import GPT2LMHeadModel
from mindformers.models.pangualpha import PanguAlphaHeadModel
from research.qwen2_5.infer.qwen2_5 import ParallelQwenForCausalLM


def dynamic_import_from(module_name, attribute_name):
    """
    dynamic_import
    """
    module = importlib.import_module(module_name)
    attribute = getattr(module, attribute_name)
    print(f"Attribute {attribute_name} from module {module_name} successfully imported.")
    return attribute



class BaseModel(nn.Cell):
    """BaseModel"""

    def __init__(self):
        nn.Cell.__init__(self)
        self._model_list = [
            "pangu",
            "gpt2",
            "llama",
            "glm4",
            "deepseek_training",
            "deepseek_infer",
        ]

    def select_actor_model(self, model_config):
        """
        select actor model
        """
        self.model_type = None
        if not model_config.model_name:
            raise NotImplementedError("model_name in actor/reference model is None")
        for model in self._model_list:
            if model in model_config.model_name:
                self.model_type = model
        if not self.model_type:
            raise NotImplementedError(
                "only support {}".format(" ".join(self._model_list))
            )
        if self.model_type == "pangu":
            self.model = PanguAlphaHeadModel(model_config)
            self.backbone = self.model.backbone
            self.lm_head = self.model.head
        elif self.model_type == "gpt2":
            self.model = GPT2LMHeadModel(model_config)
            self.backbone = self.model.backbone
            self.lm_head = self.model.head
        elif self.model_type == "llama":
            if model_config.use_past:
                self.model = ParallelQwenForCausalLM(model_config)
            else:
                self.model = LlamaForCausalLM(model_config)
            self.backbone = self.model.model
            self.lm_head = self.model.lm_head
        elif self.model_type == "glm4":
            self.model = ChatGLM2ForConditionalGeneration(model_config)
            self.backbone = self.model.transformer
            self.lm_head = self.model.transformer.output_layer
        elif "deepseek" in self.model_type:
            if model_config.use_past:
                infer_model = dynamic_import_from('research.deepseek3.deepseek3_model_infer',
                                                  'InferenceDeepseekV3ForCausalLM')
                self.model = infer_model(model_config)
            else:
                train_model = dynamic_import_from('research.deepseek3.deepseek3_model_train',
                                                  'TrainingDeepseekV3ForCausalLM')
                self.model = train_model(model_config)
            self.backbone = self.model.model
            self.lm_head = self.model.lm_head

    def select_critic_model(self, model_config):
        """
        select critic model
        """
        self.model_type = None
        if not model_config.model_name:
            raise NotImplementedError("model_name in critic model is None")
        for model in self._model_list:
            if model in model_config.model_name:
                self.model_type = model
        if not self.model_type:
            raise NotImplementedError(
                "only support {}".format(" ".join(self._model_list))
            )
        if self.model_type == "pangu":
            self.model = PanguAlphaHeadModel(model_config)
            self.backbone = self.model.backbone
        elif self.model_type == "gpt2":
            self.model = GPT2LMHeadModel(model_config)
            self.backbone = self.model.backbone
        elif self.model_type == "llama":
            if model_config.use_past:
                self.model = ParallelQwenForCausalLM(model_config)
            else:
                self.model = LlamaForCausalLM(model_config)
            self.backbone = self.model.model
        elif self.model_type == "glm4":
            self.model = ChatGLM2ForConditionalGeneration(model_config)
            self.backbone = self.model.transformer
        elif "deepseek" in self.model_type:
            if model_config.use_past:
                infer_model = dynamic_import_from('research.deepseek3.deepseek3_model_infer',
                                                  'InferenceDeepseekV3ForCausalLM')
                self.model = infer_model(model_config)
            else:
                train_model = dynamic_import_from('research.deepseek3.deepseek3_model_train',
                                                  'TrainingDeepseekV3ForCausalLM')
                self.model = train_model(model_config)
            self.backbone = self.model.model
            self.lm_head = self.model.lm_head

    def select_reward_model(self, model_config):
        """
        select reward model
        """
        self.model_type = None
        if not model_config.model_name:
            raise NotImplementedError("model_name in reward model is None")
        for model in self._model_list:
            if model in model_config.model_name:
                self.model_type = model
        if not self.model_type:
            raise NotImplementedError(
                "only support {}".format(" ".join(self._model_list))
            )
        if self.model_type == "pangu":
            self.model = PanguAlphaHeadModel(model_config)
            self.backbone = self.model.backbone
        elif self.model_type == "gpt2":
            self.model = GPT2LMHeadModel(model_config)
            self.backbone = self.model.backbone
        elif self.model_type == "llama":
            if model_config.use_past:
                self.model = ParallelQwenForCausalLM(model_config)
            else:
                self.model = LlamaForCausalLM(model_config)
            self.backbone = self.model.model
        elif self.model_type == "glm4":
            self.model = ChatGLM2ForConditionalGeneration(model_config)
            self.backbone = self.model.transformer
        elif "deepseek" in self.model_type:
            if model_config.use_past:
                infer_model = dynamic_import_from('research.deepseek3.deepseek3_model_infer',
                                                  'InferenceDeepseekV3ForCausalLM')
                self.model = infer_model(model_config)
            else:
                train_model = dynamic_import_from('research.deepseek3.deepseek3_model_train',
                                                  'TrainingDeepseekV3ForCausalLM')
                self.model = train_model(model_config)
            self.backbone = self.model.model
            self.lm_head = self.model.lm_head
