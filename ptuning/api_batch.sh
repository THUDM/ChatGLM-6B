PRE_SEQ_LEN=128
CHECKPOINT=adgen-chatglm-6b-pt-128-2e-2
STEP=3000

CUDA_VISIBLE_DEVICES=1 python3 api_batch.py \
    --do_predict \
    --validation_file test.json \
    --test_file test.json \
    --overwrite_cache \
    --prompt_column instruction \
    --model_name_or_path THUDM/chatglm-6b \
    --ptuning_checkpoint output/$CHECKPOINT/checkpoint-$STEP \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 128 \
    --max_target_length 128 \
    --per_device_eval_batch_size 100 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
