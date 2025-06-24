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

"""MindRLHF Init"""

from mindrlhf import configs, models, wrapper, trainer, worker, utils, reward
from mindrlhf.configs import *
from mindrlhf.models import *
from mindrlhf.reward import *
from mindrlhf.wrapper import *
from mindrlhf.trainer import *
from mindrlhf.worker import *
from mindrlhf.utils import *
# from mindrlhf.third_party.vllm.qwen2 import qwen2_init

# FIXME
# from vllm_mindspore.model_executor.models.mf_models.qwen2 import Qwen2ForCausalLM

# Qwen2ForCausalLM.__init__ = qwen2_init

__all__ = []
__all__.extend(configs.__all__)
__all__.extend(models.__all__)
__all__.extend(reward.__all__)
__all__.extend(wrapper.__all__)
__all__.extend(trainer.__all__)
__all__.extend(worker.__all__)
__all__.extend(utils.__all__)
