# Qwen-PPO教程

本教程基于Qwen2引导读者跑通PPO训推一体的基本流程。

## 环境准备

| 依赖        | 推荐版本                 |
| ----------- | ------------------------ |
| CANN        | 7.5 (C20_20241024)       |
| MindSpore   | 2.5.0 (12月24日)         |
| MindFormers | 1.3.0 (dev分支 12月26日) |

## 数据预处理

采用cvalues数据集进行预处理，命令如下：

```sh
cd examples/qwen_ppo_tutorial
python rlhf_data.py \
	--vocab_path path/to/vocab.json \
	--merges_file_path path/to/merges.txt \
	--file_path path/to/input.jsonl \
	--output_path path/to/output.mindrecord \
	--max_prompt_length 4096 \
	--seq_length 8193
```

## 运行命令

```sh
msrun --worker_num=4 --local_worker_num=4 \
	--master_addr=127.0.0.1 --master_port=9123 \
	--join=True --log_dir=./qwen2_one_log \
	examples/qwen_ppo_tutorial/ppo_one_stage.py \
	--sft_path_infer ./model_configs/qwen_ppo/generate_qwen2_7b.yaml \
	--sft_path_train ./model_configs/qwen_ppo/finetune_qwen2_7b.yaml \
	--reward_path ./model_configs/qwen_ppo/generate_qwen2_7b.yaml \
	--critic_path ./model_configs/qwen_ppo/generate_qwen2_7b.yaml \
	--mind_dataset_dir path/to/output.mindrecord \
	--save_ckpt_dir path/to/save_ckpt \
	--use_parallel True \
	--load_sft_checkpoint_infer path/to/infer_ckpt \
	--load_sft_checkpoint_train path/to/train_ckpt \
	--enable_compile_cache False \
	--only_save_strategy False

# 参数说明
# only_save_strategy: 若要获取各模型的分布式策略文件，设置为True。程序将编译模型，得到策略文件并保存到 ./strategy 目录，然后直接退出。
```

​	PPOTrainer中有ppo_model_infer和ppo_model_train两个成员变量，两者分布式策略不同，加载不同的yaml进行初始化。

