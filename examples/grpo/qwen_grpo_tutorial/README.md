# Qwen-GRPO强化学习训练教程

GRPO（Group Relative Policy Optimization）是针对数学逻辑推理任务提出的强化学习优化算法。训练过程是通过学习一个策略模型，通过不断试错，策略模型与奖励模型的不断交互，策略模型会逐渐倾向于选择获得更高奖励的行为，自主探索出最佳学习路径。通过GRPO算法大规模训练，大模型在逻辑推理能力上得到显著提升。

本教程基于`Qwen2.5 7b`模型与`GSM8K Train`数据集引导读者跑通单机8卡GRPO训推一体的基本流程。

## 一、模型以及数据集获取与预处理

### 模型权文件和tokenizer获取
用户可以从[HuggingFace官方](https://huggingface.co/Qwen/Qwen2.5-7B)或[魔搭社区](https://modelscope.cn/models/Qwen/Qwen2.5-7B)下载完整预训练权重，模型对应的tokenizer文件`vocab.json`和`merges.txt`也可在上述链接中下载。

模型权重下载完成后，需要转为MindSpore使用的.ckpt文件。首先进入MindFormers路径
```shell
cd /{path}/mindformers
```
并执行以下脚本：

```shell
python convert_weight.py --model qwen2_5 --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME.ckpt --dtype bf16

# 参数说明
model:       模型名称
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
dtype:       转换权重的精度
```
脚本会将完整的.ckpt格式模型权重保存在`{path}/MS_CKPT_NAME.ckpt`路径下。

### 模型权重离线切分
当前版本的MindRLHF尚不支持权重在线切分，在使用多卡分布式训练时，需要用户手动进行[分布式权重切分](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.2/function/transform_weight.html)。
首先，在MindFormers路径下使用以下命令得到并行策略文件`/strategy`:
```bash
cd /research/qwen2_5

bash ../../scripts/msrun_launcher.sh "run_qwen2_5.py \
--config /{path}/desired_model_config.yaml \
--run_mode finetune \
--train_data /{path}/alpaca-fastchat4096.mindrecord " 8 PORT output/msrun_log False 2000
```
其中，数据文件`/{path}/alpaca-fastchat4096.mindrecord`可以按照[这份教程](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.2/quick_start/source_code_start.html)
中的指导进行生成，而模型配置文件`/{path}/desired_model_config.yaml`可以使用[finetune_qwen2_5_7b_for_strategy.yaml](model_configs/qwen_grpo/finetune_qwen2_5_7b_for_strategy.yaml)，
并手动将其中的并行策略配置更改为用户希望的并行策略。

此命令将会在`/research/qwen2_5/strategy`路径下生成并行策略文件，在下一步切分ckpt时作为dst_strategy的值传入。

随后，执行以下脚本将完整权重切分为分布式权重
```bash
cd ../..
nohup python transform_checkpoint.py \
--src_checkpoint=/{path}/MS_CKPT_NAME.ckpt \
--dst_checkpoint=/{path}/distributed_ckpt/ \
--dst_strategy=/research/qwen2_5/strategy/ > output.log 2>&1 &

# 参数说明
src_checkpoint: 原始完整权重路径
dst_checkpoint: 目标分布式权重路径
dst_strategy:   目标权重策略文件路径
```
需要注意，在GRPO算法中存在训练和推理两份模型权重，若训练和推理所使用的分布式策略不同，则需要分别切分两份分布式权重。

### 数据集文件获取与预处理
用户可以从[GSM8K Github Repo](https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/)下载得到
`GSM8K Train`数据集`train.jsonl`。下载完成后，需要转为MindSpore使用的.mindrecord文件。首先进入MindRLHF路径
```shell
cd /{path}/mindrlhf
```
并执行以下脚本：

```shell
python examples/grpo/qwen_grpo_tutorial/rlhf_data.py \
--vocab_path /{path}/vocab.json \
--merges_file_path /{path}/merges.txt \
--file_path /{path}/train.jsonl/ \
--output_path /{path}/gsm8k_train.mindrecord

# 参数说明
vocab_path:       qwen2.5 7b模型对应的tokenizer文件vocab.json路径
merges_file_path: qwen2.5 7b模型对应的tokenizer文件merges.txt路径
file_path:        GSM8K Train数据集train.jsonl文件路径
output_path:      输出.mindrecord文件路径
```
其中`vocab.json`和`merges.txt`都可以从Huggingface社区或魔搭社区对应模型页面获取。
此脚本会将`train.jsonl`转换成mindrecord的形式保存在`/{path}/gsm8k_train.mindrecord`。此数据路径将在训练拉起时作为`mind_dataset_dir`的值被传入。

## 二、GRPO算法及模型配置

### 训练/推理模型配置
训练模型的配置文件默认为`model_configs/qwen_grpo/finetune_qwen2_5_7b.yaml`,其中用户可以手动配置训练模型的并行策略：
```shell
parallel_config:
    data_parallel: 1 
    model_parallel: 4 
    pipeline_stage: 2 
    use_seq_parallel: True
    micro_batch_num: 2
    micro_batch_interleave_num: 2 
# 参数说明
data_parallel:                数据并行切分组数
model_parallel:               模型并行(tensor parallel)切分组数
pipeline_stage:               流水线并行切分组数
use_seq_parallel:             是否使用sequence parallel
micro_batch_num:              流水线并行中的micro batch number
micro_batch_interleave_num:   当model_parallel>1时,可以设置为2以加速训练
```
推理模型的配置文件默认为`model_configs/qwen_grpo/predict_qwen2_5_7b_instruct.yaml`,其中用户可以手动配置推理模型的并行策略：

```shell
parallel_config:
    data_parallel: 2
    model_parallel: 4 
    pipeline_stage: 1
# 参数说明
data_parallel:                数据并行切分组数
model_parallel:               模型并行(tensor parallel)切分组数
pipeline_stage:               流水线并行切分组数, 必须为1. 当前推理模型不支持流水线并行
```

### GRPO训练算法配置
GRPO训练算法相关配置可以在`mindrlhf/configs/grpo_configs.py`内进行修改，包括以下参数：

```shell
beta: float = 0.01
num_generations: int = 8
grpo_epochs: int = 2
start_lr: float = 5e-7 
end_lr: float = 1e-10
chunk_size: int = 2
batch_size: int = 2
sync_ref_model: bool = True
ref_model_sync_steps: int = 50

# 参数说明
grpo_epochs:            在数据集上总共训练的epochs轮数
chunk_size:             推理模型在每一步中为多少个问题生成回答
num_generations:        推理模型在每一步中为每个问题生成多少个回答
beta:                   反向训练GRPO loss中KL散度的权重
start_lr:               初始时反向训练的learning rate步长
end_lr:                 结束时反向训练的learning rate步长, 必须大于start_lr
batch_size:             反向训练的batch size
sync_ref_model:         是否每隔若干步将ref model的权重更新为最新的训练模型权重
ref_model_sync_steps:   若sync_ref_model=True, ref model权重更新的间隔步数
```


## 三、启动单机8卡GRPO训练脚本
首先进入MindRLHF路径
```shell
cd /{path}/mindrlhf
```
使用以下命令将本地mindrlhf和mindformers代码库均加入PYTHONPATH，MINDFORMERS_PAT路径中
```shell
MINDRLHF_FILE=/{path}/mindrlhf/
MINDFORMERS_FILE=/{path}/mindformers/

export PYTHONPATH="$MINDRLHF_FILE:$MINDFORMERS_FILE:$PYTHONPATH"
export MINDFORMERS_PATH="$MINDFORMERS_FILE $MINDFORMERS_PATH"
```

随后使用以下命令拉起单机8卡GRPO训练任务
```shell
msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 \
--master_port=9190 --join=False --log_dir=./qwen2_5_one_log \
examples/grpo/qwen_grpo_tutorial/grpo_one_stage.py \
--sft_path_infer ./model_configs/qwen_grpo/predict_qwen2_5_7b_instruct.yaml \
--sft_path_train ./model_configs/qwen_grpo/finetune_qwen2_5_7b.yaml \
--vocab_path /{path}/vocab.json \
--merges_file_path /{path}/merges.txt \
--mind_dataset_dir /{path}/gsm8k_train.mindrecord \
--save_data_file /{path}/grpo.mindrecord \
--save_ckpt_dir /{path}/save_ckpt \
--use_parallel True \
--load_sft_checkpoint_infer /{path}/infer_ckpt \
--load_sft_checkpoint_train /{path}/train_ckpt \
--load_ref_checkpoint /{path}/ref_ckpt \
--enable_compile_cache False 

# 参数说明
# msrun 参数
worker_num:                   总卡数
local_worker_num:             单机的卡数
master_addr:                  主节点地址
master_port:                  主节点端口
join:                         是否等待所有worker退出
log_dir:                      日志路径
# grpo_one_stage.py 参数
sft_path_infer:               推理用的模型配置文件
sft_path_train:               训练用的模型配置文件
vocab_path:                   模型对应的tokenizer文件vocab.json的路径
merges_file_path:             模型对应的tokenizer文件merges.txt的路径
mind_dataset_dir:             训练数据集mindrecord文件的路径
save_data_file:               中间推理结果的保存路径(可选)
save_ckpt_dir:                训练ckpt的保存路径
use_parallel:                 是否并行
load_sft_checkpoint_infer:    推理模型(分布式)ckpt文件路径
load_sft_checkpoint_train:    训练模型(分布式)ckpt文件路径
load_ref_checkpoint:          参考模型(分布式)ckpt文件路径
enable_compile_cache:         是否使用编译缓存
```

拉起任务后，通过以下命令查看运行日志
```shell
tail -f qwen2_5_one_log/worker_0.log
```

## 四、训练收敛曲线

基于Qwen/Qwen2.5-7B模型，使用gsm8k数据集，训练过程中主要配置项设`num_rollouts=8`，`chunk_size=2`，`lr=9.0e-6`，跑出收敛曲线如下：

![grpo_converge](https://gitee.com/mindspore/mindrlhf/tree/master/images/grpo_converge.png)
