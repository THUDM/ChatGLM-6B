PRE_SEQ_LEN=128
LR=2e-2

CUDA_VISIBLE_DEVICES=1,2 python3 main_parallel.py \
    --do_train \
    --train_file AdvertiseGen/train.json \
    --test_file AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --preprocessing_num_workers 10 \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm-6b \
    --output_dir ./output/parallel-chatglm-6b-ptuning-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 1000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --save_total_limit 1 \
    --gradient_checkpointing \
    --fp16