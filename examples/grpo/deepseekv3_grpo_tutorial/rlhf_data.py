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
import jsonlines
import numpy as np
from mindformers.models.llama import LlamaTokenizerFast
from mindspore.mindrecord import FileWriter
from tqdm import tqdm


def load_json_file(file_path):
    "Read data from json file"
    raw_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for item in jsonlines.Reader(f):
            raw_data.append(item)
    return raw_data


def process_data(tokenizer, raw_data, max_prompt_length, seq_length, pad_token_id):
    """
    process_data
    """

    template = "{prompt}{response}"

    for item in tqdm(raw_data):
        sample = {}
        prompt = template.format_map(
            {"prompt": item["question"], "response": ""})
        response = template.format_map(
            {"prompt": item["question"], "response": item["answer"]}
        )

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
        prompt_ids = np.pad(
            prompt_ids,
            (0, max_prompt_length - prompt_ids.shape[-1]),
            "constant",
            constant_values=(0, pad_token_id),
        )
        pretrain_ids = np.pad(
            pretrain_ids,
            (0, seq_length - pretrain_ids.shape[-1]),
            "constant",
            constant_values=(0, pad_token_id),
        )
        loss_mask = np.pad(
            loss_mask,
            (0, seq_length - loss_mask.shape[-1]),
            "constant",
            constant_values=(0, pad_token_id),
        )
        loss_mask[:prompt_len] = 0.0

        sample["prompt_ids"] = prompt_ids
        sample["pretrain_ids"] = pretrain_ids
        sample["loss_mask"] = loss_mask

        yield sample


def write_mindrecord(args):
    """
    write_mindrecord
    """

    raw_data = load_json_file(args.file_path)

    tokenizer = LlamaTokenizerFast(
        tokenizer_file=args.tokenizer_file, add_bos_token=False, add_eos_token=False
    )
    tokenizer.pad_token_id = 100001
    max_prompt_length = int(args.max_prompt_length)
    seq_length = int(args.seq_length)
    if args.pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    else:
        pad_token_id = int(args.pad_token_id)

    schema = {
        "prompt_ids": {"type": "int64", "shape": [-1]},
        "pretrain_ids": {"type": "int64", "shape": [-1]},
        "loss_mask": {"type": "int64", "shape": [-1]},
    }

    writer = FileWriter(file_name=args.output_path,
                        shard_num=1, overwrite=True)
    writer.add_schema(schema)

    count = 0
    for sample in process_data(tokenizer, raw_data, max_prompt_length, seq_length, pad_token_id):
        count += 1
        writer.write_raw_data([sample])
    print("Total number of samples: {}".format(count))

    writer.commit()
    print("Transformation finished! Output file refer: {}".format(args.output_path))


def get_args():
    """
    get args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_file", required=True,
                        help="path to tokenizer.json")
    parser.add_argument("--file_path", required=True,
                        help="file path to raw data.")
    parser.add_argument(
        "--output_path", required=True, help="file path to output mindrecord file."
    )
    parser.add_argument(
        "--max_prompt_length", default=2048, help="max prompt encode length."
    )
    parser.add_argument("--seq_length", default=4096,
                        help="encoded sequence length.")
    parser.add_argument("--pad_token_id", default=None, help="pad token id.")
    args_opt = parser.parse_args()
    return args_opt


if __name__ == "__main__":
    my_args = get_args()
    write_mindrecord(my_args)
