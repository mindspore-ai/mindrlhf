WORKDIR=/home/wyc/rl/zpc/
export ASCEND_HOME_PATH=/home/wyc/cann/latest
export PYTHONPATH=$WORKDIR/mindrlhf:$WORKDIR/mindformers/:$PYTHONPATH
export GLOG_v=3
# export ASCEND_RT_VISIBLE_DEVICES='0,1,2,3'
export ASCEND_RT_VISIBLE_DEVICES='6,7'

# tracker
export MS_ENABLE_TRACE_MEMORY="off"
export MS_ALLOC_CONF="enable_vmm:True,memory_tracker:True"
export MS_DEV_RUNTIME_CONF="memory_statistics:True"

taskset -c 0-23 msrun --worker_num=2 --local_worker_num=2 \
--master_addr=127.0.0.1 --master_port=2333 \
--join=True --log_dir=./wyc_log main.py \
--sft_path_infer $WORKDIR/mindrlhf/model_configs/qwen_grpo/predict_qwen2_5_7b_instruct.yaml \
--sft_path_train $WORKDIR/mindrlhf/model_configs/qwen_grpo/finetune_qwen2_5_7b.yaml \
--vocab_path $WORKDIR/mindrlhf_grpo_data/ckpt/qwen2_7b/vocab.json \
--merges_file_path $WORKDIR/mindrlhf_grpo_data/ckpt/qwen2_7b/merges.txt \
--mind_dataset_dir $WORKDIR/limr_data/limr_template_qwenr1.mindrecord \
--save_data_file $WORKDIR/grpo_mindrecord/grpo_1024.mindrecord \
--save_ckpt_dir $WORKDIR/mindrlhf_grpo_data/ckpt/train \
--use_parallel True \
--enable_compile_cache False  > origin.log 2>&1 &

# 暂时不加载权重
# --load_sft_checkpoint_infer "/home/wyc/rl/zpc/mindrlhf_grpo_data/ckpt/qwen2_7b/ms_ckpt/dp1mp4/ms_qwen2_qkv_concat"