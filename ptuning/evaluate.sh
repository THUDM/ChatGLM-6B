PRE_SEQ_LEN=8
CHECKPOINT=adgen-chatglm-6b-pt-8-1e-2
STEP=3000

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_predict \
    --test_file AdvertiseGen/dev.json \
    --overwrite_cache \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path ./output/$CHECKPOINT/checkpoint-$STEP  \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --max_predict_samples 10 \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4
