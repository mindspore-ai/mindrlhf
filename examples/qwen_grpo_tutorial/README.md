# Qwen-GRPO强化学习训练教程
## 一、介绍
GRPO（Group Relative Policy Optimization）是针对数学逻辑推理任务提出的强化学习优化算法。训练过程是通过学习一个策略模型，通过不断试错，策略模型与奖励模型的不断交互，策略模型会逐渐倾向于选择获得更高奖励的行为，自主探索出最佳学习路径。通过GRPO算法大规模训练，大模型在逻辑推理能力上得到显著提升。

本代码仓实现了GRPO的强化学习训推流程。

当前通过获取开源镜像或者自行搭建环境，开发者可直接进行GRPO强化学习训练流程的复现，参考教程如下。

## 二、环境搭建
### 版本匹配关系
| 依赖 | 版本 |
|------|----|
| 固件&驱动 | 24.1.RC3.3 |
| CANN | 8.0 |
| Python | 3.10 |
| MindSpore | master, commit id：94ac228bae9cd6d0f00b4ce8d5857773799c4f26 |
| MindFormers | dev |
| MindRLHF | master, commit id：90977955470ea04f0e2256d6b73bc71ff62cf092 |

大家可根据以上版本自行安装。

为了方便复现，我们也提供了镜像。


### 镜像安装
1.下载Docker镜像
2.基于镜像创建容器

    docker run -itd  --privileged  --network=host \
       --shm-size 500g \
       --device=/dev/davinci0 \
       --device=/dev/davinci1 \
       --device=/dev/davinci2 \
       --device=/dev/davinci3 \
       --device=/dev/davinci4 \
       --device=/dev/davinci5 \
       --device=/dev/davinci6 \
       --device=/dev/davinci7 \
       --device=/dev/davinci_manager \
       --device=/dev/hisi_hdc \
       --device /dev/devmm_svm \
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
       -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
       -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
       -v /usr/local/sbin:/usr/local/sbin \
       -v /etc/hccn.conf:/etc/hccn.conf \
       CONTAINER_NAME:TAG \
       bash
3.进入容器 
```
docker exec -it CONTAINER_ID bash
```

## 三、使用指南
### 数据集及文件的获取
使用`examples/qwen_grpo_tutorial/rlhf_data.py`将`GSM8k.json`转换成mindrecord的形式，此数据路径为`mind_dataset_dir`的取值。此数据路径在启动训推作为`mind_dataset_dir`的值。

```
python rlhf_data.py --vocab_path /path/to/vocab.json \
--merges_file_path /path/to/merges.txt \
--file_path /path/to/raw/data/ \
--output_path /path/to/mindrecord/
```
参数说明


    vocab_path：vocab.json路径
    merges_file_path：merges.txt路径
    file_path：原始数据文件路径
    output_path：输出文件路径

其中`vocab.json`和`merges.txt`都可以从Huggingface社区对应模型页面获取。


### 权重获取
由于训练是计算密集型，生成是内存密集型，所以为了最大限度的优化性能，推理与训练阶段采用的并行策略往往不同。因此需要针对推理和训练模型分别进行切分。
下面介绍了获得不同权重切分的方法

#### MindSpore权重转换
从Huggingface下载完整权重，要转为MindSpore用的ckpt，需进入MindFormers路径下执行以下脚本
```bash
cd research/qwen2_5/

python convert_weight.py --torch_ckpt_dir /path/to/your/torch/ckpt/  \
--mindspore_ckpt_path /path/to/save/ms/ckpt --dtype bf16 --config_path \  
research/qwen2_5/finetune_qwen2_5_7b.yaml
```
参数说明


    torch_ckpt_dir：torch权重文件
    mindspore_ckpt_path： mindspore权重保存路径
    config_path：模型权重配置文件

#### 获得策略文件
```bash
cd /path/to/your/mindformers/research/qwen2_5

bash ../../scripts/msrun_launcher.sh "run_qwen2_5.py \
--config /path/to/your/desired/model/yaml \
--run_mode finetune \
--train_data /path/to/mindrecord " 8 PORT output/msrun_log False 2000
```

参数说明：


    # run_qwen2_5.py 参数
    config：模型的配置文件
    run_mode：运行模式选微调
    train_data：训练用的数据文件
    # msrun_launcher.sh 参数
    单机上卡数8
    PORT为节点PORT
    是否等待所有worker退出：False
    集群通信超时：2000
生成的策略文件在strategy下，在下一步切分ckpt时作为dst_strategy的值。

#### 获得特定切分的ckpt
```bash
nohup python transform_checkpoint.py \
--src_checkpoint=/path/to/checkpoint.ckpt \
--dst_checkpoint=/path/to/desired/ckpt/ \
--dst_strategy=/path/to/strategy/ > output.log 2>&1 &
```

参数说明


    src_checkpoint：原始权重路径
    dst_checkpoint：目标权重路径
    dst_strategy：目标权重策略文件路径

### 训练/推理模型配置
#### 训练模型配置
训练的模型的配置`finetune_qwen2_5_7b.yaml`:
```
run_mode: 'finetune'
```
并行配置:


    parallel_config:
		data_parallel: 1 # 数据并行切分为 1
		model_parallel: 4 # 模型并行切分为 4
		pipeline_stage: 2 # 流水线并行切分为 2
		use_seq_parallel: True
		micro_batch_num: 2
		vocab_emb_dp: False
		gradient_aggregation_group: 4
		micro_batch_interleave_num: 2 # mp大于1时，设为1可提升训练效率

训练相关配置在`mindrlhf/configs/grpo_configs.py`有学习率和GRPO相关的超参


    optimizer: str = 'adamw' # 优化器类型
    beta1: float = 0.9  # 优化器adamw超参，下同
    beta2: float = 0.95 
    eps: float = 1.0e-8 
    weight_decay: float = 0.01
	
	epochs: int = 100 # 训练轮数

#### 推理模型配置
推理的模型的配置`predict_qwen2_5_7b_instruct.yaml`
```
run_mode: 'predict'
```
并行配置


    parallel_config:
		data_parallel: 2 # 数据并行切分为2
		model_parallel: 4 # 模型并行切分为4
		pipeline_stage: 1 # 流水线并行切分为1
		micro_batch_num: 1
		vocab_emb_dp: False
		gradient_aggregation_group: 4
		micro_batch_interleave_num: 1


### 启动单机8卡训练脚本
用`bash run_grpo.sh`启动GRPO强化学习流程。

注意：用户需要确认将MindSpore Transformers和MindSpore RLHF的路径加入`PYTHONPATH`。

    msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 \
    --master_port=9190 --join=False --log_dir=./qwen2_5_one_log \
    examples/qwen_grpo_tutorial/grpo_one_stage.py \
    --sft_path_infer ./model_configs/qwen_grpo/predict_qwen2_5_7b_instruct.yaml \
    --sft_path_train ./model_configs/qwen_grpo/finetune_qwen2_5_7b.yaml \
    --vocab_path /path/to/your/vocab.json \
    --merges_file_path /path/to/your/merges.txt \
    --mind_dataset_dir /path/to/gsm8k.mindrecord \
    --save_data_file /path/to/grpo.mindrecord \
    --save_ckpt_dir /path/to/save/ckpt \
    --use_parallel True \
    --load_sft_checkpoint_infer /path/to/infer/ckpt \
    --load_sft_checkpoint_train /path/to/train/ckpt \
	--load_ref_checkpoint /path/to/ref/ckpt \
    --enable_compile_cache False \
    --only_save_strategy False

参数说明

    # msrun 参数
	worker_num： 总卡数
	local_worker_num： 单机的卡数
	master_addr：主节点地址
	master_port: 主节点端口
	join：是否等待所有worker退出
	log_dir: 日志路径
	# grpo_one_stage.py 参数
    sft_path_infer：推理用的模型配置
    sft_path_train：训练用的模型配置
	vocab_path: vocab.json的路径
    merges_file_path：权重合并配置
	mind_dataset_dir：训练数据文件的路径
	save_data_file：中间推理结果的保存路径（可选）
	save_ckpt_dir：训练ckpt的保存路径
	use_parallel：是否并行
    load_sft_checkpoint_infer: 推理ckpt路径
    load_sft_checkpoint_train: 训练ckpt路径
	load_ref_checkpoint： 参考模型ckpt路径
	enable_compile_cache：是否编译缓存
	only_save_strategy：是否保存策略文件


拉起任务后，通过以下命令查看运行日志
```
tail -f qwen2_5_one_log/worker_0.log
```
