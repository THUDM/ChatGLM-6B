'''
Author: lichuang
Date: 2023-03-23 09:18:13
Description: 将模型加载到多张GPU卡中，根据gpu的数量自动分配平均的显存占用 
'''
from typing import Dict

from accelerate import load_checkpoint_and_dispatch
from transformers import AutoModel


def auto_configure_device_map(num_gpus) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    device_map = {'transformer.word_embeddings': 0,
                  'transformer.final_layernorm': num_gpus - 1, 'lm_head': num_gpus - 1}

    used = 1
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.layers.{i}'] = gpu_target
        used += 1

    return device_map


def load_model_on_gpus(checkpoint_path, num_gpus=2):
    device_map = auto_configure_device_map(num_gpus)

    model = AutoModel.from_pretrained(
        checkpoint_path, trust_remote_code=True)
    model = model.eval()
    model = load_checkpoint_and_dispatch(
        model, checkpoint_path, device_map=device_map, offload_folder="offload", offload_state_dict=True).half()
    return model
