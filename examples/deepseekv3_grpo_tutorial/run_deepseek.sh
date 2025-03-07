msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 --master_port=9444  --join=True \
--log_dir=./ppo_ds3_pp2_transform_log  \
examples/qwen_grpo_tutorial/grpo_one_stage.py \
--sft_path_infer ./model_configs/qwen_grpo/predict_deepseek3_671b.yaml  \
--sft_path_train ./model_configs/qwen_grpo/finetune_deepseek3_671b.yaml  \
--vocab_path /path/vocab.json \
--merges_file_path path/merges.txt \
--mind_dataset_dir ../output_4096.mindrecord  \
--save_ckpt_dir ../ckpt \
--use_parallel True  \
--enable_compile_cache False > test.log 2>&1 &