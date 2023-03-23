'''
Author: lichuang
Date: 2023-03-23 09:18:13
Description: 将模型加载到多张GPU卡中，根据gpu的数量自动分配平均的显存占用 
'''

from transformers import AutoModel, AutoTokenizer
from accelerate import load_checkpoint_and_dispatch


def load_model_on_gpus(checkpoint_path, num_gpus=2):
    # 总共占用13GB显存,28层transformer每层0.39GB左右
    # 第一层 word_embeddings和最后一层 lm_head 层各占用1.2GB左右
    num_trans_layers = 28
    vram_per_layer = 0.39
    average = 13/num_gpus
    used = 1.2
    device_map = {'transformer.word_embeddings': 0,
                  'transformer.final_layernorm': num_gpus-1, 'lm_head': num_gpus-1}
    gpu_target = 0
    for i in range(num_trans_layers):
        if used > average-vram_per_layer/2 and gpu_target < num_gpus:
            gpu_target += 1
            used = 0
        else:
            used += vram_per_layer
        device_map['transformer.layers.%d' % i] = gpu_target

    model = AutoModel.from_pretrained(
        checkpoint_path, trust_remote_code=True)
    model = model.eval()
    model = load_checkpoint_and_dispatch(
        model, checkpoint_path, device_map=device_map, offload_folder="offload", offload_state_dict=True).half()
    return model
