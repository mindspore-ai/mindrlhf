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
"""
    data process
"""
import argparse
import numpy as np
import jsonlines
from tqdm import tqdm
from mindspore.mindrecord import FileWriter
from mindformers import AutoTokenizer
from mindformers import logger


def load_json_file(file_path):
    """Read data from json file"""
    raw_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for item in jsonlines.Reader(f):
            raw_data.append(item)
    return raw_data


def process_data(tokenizer, raw_data, max_prompt_length, seq_length, pad_token_id):
    """Process data"""
    template = "{prompt}{response}"

    for item in tqdm(raw_data):
        sample = {}
        prompt = template.format_map({"prompt": item["prompt"], "response": ""})
        response = template.format_map({"prompt": item["prompt"], "response": item["pos_resp"]})

        prompt_dict = tokenizer(
            prompt,
            truncation=True,
            max_length=max_prompt_length,
            add_special_tokens=False,
        )

        response_dict = tokenizer(
            response,
            truncation=True,
            max_length=seq_length,
            add_special_tokens=False,
        )

        prompt_ids = np.array(prompt_dict["input_ids"])
        prompt_len = prompt_ids.shape[-1]
        pretrain_ids = np.array(response_dict["input_ids"])
        loss_mask = np.array(response_dict["attention_mask"])
        prompt_ids = np.pad(prompt_ids, (0, max_prompt_length -
                                         prompt_ids.shape[-1]), 'constant', constant_values=(0, pad_token_id))
        pretrain_ids = np.pad(pretrain_ids, (0, seq_length -
                                             pretrain_ids.shape[-1]), 'constant', constant_values=(0, pad_token_id))
        loss_mask = np.pad(loss_mask, (0, seq_length - loss_mask.shape[-1]),
                           'constant', constant_values=(0, pad_token_id))
        loss_mask[:prompt_len] = 0.0

        sample["prompt_ids"] = prompt_ids
        sample["pretrain_ids"] = pretrain_ids
        sample["loss_mask"] = loss_mask

        yield sample


def write_mindrecord(args):
    """Write mindrecord"""
    raw_data = load_json_file(args.file_path)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    max_prompt_length = int(args.max_prompt_length)
    seq_length = int(args.seq_length)
    pad_token_id = int(args.pad_token_id)

    schema = {
        "prompt_ids": {"type": "int64", "shape": [-1]},
        "pretrain_ids": {"type": "int64", "shape": [-1]},
        "loss_mask": {"type": "int64", "shape": [-1]},
    }

    writer = FileWriter(file_name=args.output_path, shard_num=1, overwrite=True)
    writer.add_schema(schema)

    count = 0
    for sample in process_data(tokenizer, raw_data, max_prompt_length, seq_length, pad_token_id):
        count += 1
        writer.write_raw_data([sample])
    logger.info("Total number of samples: {}".format(count))

    writer.commit()
    logger.info("Transformation finished! Output file refer: {}".format(args.output_path))


def get_args():
    """Get args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tokenizer_name_or_path',
        required=True,
        help='model name or path for AutoTokenizer.')
    parser.add_argument(
        '--file_path',
        required=True,
        help='file path to raw data.')
    parser.add_argument(
        '--output_path',
        required=True,
        help='file path to output mindrecord file.'
    )
    parser.add_argument(
        '--max_prompt_length',
        default=2048,
        help='max prompt encode length.')
    parser.add_argument(
        '--seq_length',
        default=4096,
        help='encoded sequence length.')
    parser.add_argument(
        '--pad_token_id',
        default=0,
        help='pad token id.')
    args_opt = parser.parse_args()
    return args_opt


if __name__ == "__main__":
    write_mindrecord(get_args())
