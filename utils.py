import os
from typing import Dict, Tuple, Union

from accelerate import load_checkpoint_and_dispatch
from transformers import AutoModel, AutoTokenizer


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
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


def load_model_on_gpus(checkpoint_path: Union[str, os.PathLike],
                       multi_gpu_model_cache_dir: Union[str, os.PathLike] = "./temp_model_dir",
                       num_gpus: int = 2, **kwargs):
    model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs)
    model = model.eval()

    device_map = auto_configure_device_map(num_gpus)
    try:
        model = load_checkpoint_and_dispatch(
            model, checkpoint_path, device_map=device_map, offload_folder="offload", offload_state_dict=True).half()
    except ValueError:
        # index.json not found
        print(f"index.json not found, auto fixing and saving model to {multi_gpu_model_cache_dir} ...")

        assert multi_gpu_model_cache_dir is not None, "using auto fix, cache_dir must not be None"
        model.save_pretrained(multi_gpu_model_cache_dir, max_shard_size='2GB')
        model = load_checkpoint_and_dispatch(
            model, multi_gpu_model_cache_dir, device_map=device_map,
            offload_folder="offload", offload_state_dict=True).half()

        print(f"loading model successfully, you should use checkpoint_path={multi_gpu_model_cache_dir} next time")

    return model


def load_model_and_tokenizer(checkpoint_path: Union[str, os.PathLike],
                             multi_gpu_model_cache_dir: Union[str, os.PathLike] = "./temp_model_dir",
                             num_gpus: int = 1, **kwargs) -> Tuple[AutoModel, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs)
    if num_gpus < 2:
        model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half().cuda()
        model = model.eval()
    else:
        model = load_model_on_gpus(checkpoint_path, multi_gpu_model_cache_dir, num_gpus, **kwargs)
    return model, tokenizer
