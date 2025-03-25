WORKDIR=$PWD/../
export PYTHONPATH=$WORKDIR/mindrlhf:$WORKDIR/mindformers/:$PYTHONPATH
export GLOG_v=2
# export MS_ENABLE_LCCL='off'

msrun --worker_num=8 --local_worker_num=8 \
--master_addr=127.0.0.1 --master_port=2333 \
--join=True --log_dir=./qwen2_one_log examples/grpo/qwen_grpo_tutorial/main.py \
--sft_path_infer ./model_configs/qwen_grpo/predict_qwen2_5_7b_instruct.yaml \
--sft_path_train ./model_configs/qwen_grpo/finetune_qwen2_5_7b.yaml \
--vocab_path /path/to/vocab.json \
--merges_file_path /path/to/merges.txt \
--mind_dataset_dir /path/to/grpo_data/cvalues_one.mindrecord \
--save_data_file /path/to/grpo_data/grpo_1024.mindrecord \
--save_ckpt_dir /path/to/ckpt/train \
--load_sft_checkpoint_infer "/path/to/ckpt/infer" \
--load_sft_checkpoint_train "/path/to/ckpt/train" \
--load_ref_checkpoint "/path/to/ckpt/infer" \
--use_parallel True \
--enable_compile_cache False  > origin.log 2>&1 &
