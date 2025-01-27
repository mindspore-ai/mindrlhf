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
"""test qwen2_5"""
import os
from pathlib import Path
import pytest
from ruamel.yaml import YAML
from mindspore.mindrecord import FileReader
from mindformers.tools.download_tools import download_with_progress_bar

root_path = os.path.dirname(os.path.abspath(__file__)).split('tests')[0]


class TestQwen25DPO:
    """A test class for testing qwen2_5 dpo"""

    @staticmethod
    def setup_cmd(scripts_cmd, device_nums, master_port):
        """setup cmd"""
        cmd = f"msrun --worker_num={device_nums} " + \
              f"--local_worker_num={device_nums} " + \
              f"--master_port={master_port} " + \
              f"--log_dir=msrun_log " + \
              f"--join=True " + \
              f"--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        return cmd

    @staticmethod
    def change_yaml(config_file):
        """change yaml"""
        yaml = YAML()
        yaml.preserve_quotes = True
        with open(config_file, "r") as f:
            model_config = yaml.load(f)
        model_config["model"]["model_config"]["num_layers"] = 1
        with open(config_file, "w") as f:
            yaml.dump(model_config, f)

    @staticmethod
    def create_dataset(sh_path):
        """create dataset"""
        data_path = os.path.join(sh_path, 'intput.jsonl')
        num_processes = 8
        train_jsonl = Path(data_path).resolve()
        script_path = Path("../create_dpo_dataset.sh").resolve()
        cmd = f"bash {script_path} {num_processes} {train_jsonl}"
        os.system(cmd)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_qwen2_5_dpo_process(self):
        """test qwen2_5 dpo process"""
        download_with_progress_bar("https://www.modelscope.cn/models/Qwen/Qwen2.5-7B/resolve/master/vocab.json",
                                   f"{root_path}/checkpoint_download/qwen2_5/vocab.json")
        download_with_progress_bar("https://www.modelscope.cn/models/Qwen/Qwen2.5-7B/resolve/master/merges.txt",
                                   f"{root_path}/checkpoint_download/qwen2_5/merges.txt")
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        self.create_dataset(sh_path)
        self.change_yaml(os.path.join(root_path, "model_configs", "qwen_config", "process_qwen2_5_7b.yaml"))
        scripts_path = f"{root_path}/dpo_preprocess.py"
        dpo_data_path = os.path.join(sh_path, 'intput.jsonl')

        scripts_cmd = f"{scripts_path} --src={dpo_data_path} " + \
                      f"--dst={sh_path}/qwen25.mindrecord " + \
                      f"--config={root_path}/model_configs/qwen_config/process_qwen2_5_7b.yaml " + \
                      f"--tokenizer={root_path}/checkpoint_download/qwen2_5/vocab.json " + \
                      f"--merges_file={root_path}/checkpoint_download/qwen2_5/merges.txt " + \
                      f"--seq_len=4097 " + \
                      f"--dataset_type=cvalues " + \
                      f"--save_interval=2"
        ret = os.system(self.setup_cmd(scripts_cmd, 8, 8118))
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log/worker_*.log"
        os.system(f"python {root_path}/dpo_preprocess.py \
                    --merge True --src={sh_path} \
                    --dst {sh_path}/qwen25.mindrecord")
        reader = FileReader(file_name=f"{sh_path}/qwen25.mindrecord")
        ms_len = reader.len()
        reader.close()

        with open(f"{dpo_data_path}", 'r') as file:
            json_len = sum(1 for _ in file)

        assert ms_len == json_len
        assert os.path.isfile(f"{sh_path}/qwen25.mindrecord")

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_qwen2_5_finetune(self):
        """test qwen2_5 finetune"""
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = f"{root_path}/run_dpo.py"
        self.change_yaml(os.path.join(root_path, "model_configs", "qwen_config", "finetune_qwen2_5_7b_dpo.yaml"))
        scripts_cmd = f"{scripts_path} --config={root_path}/model_configs/qwen_config/finetune_qwen2_5_7b_dpo.yaml " + \
                      f"--run_mode=finetune " + \
                      f"--train_dataset={sh_path}/qwen25.mindrecord "

        ret = os.system(self.setup_cmd(scripts_cmd, 8, 8128))
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log/worker_*.log"
