import os
from typing import Dict, Tuple, Union, Optional, List

import torch
from torch.nn import Module
from transformers import AutoModel, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer


def calculate_per_gpu_layers(gpu_list: List[int], total_layers: int) -> Dict[int, int]:
    """
    Calculate the number of layers to be allocated to each GPU based on the memory ratio.

    Args:
        gpu_list (List[int]): A list of GPU indices.
        total_layers (int): The total number of layers in the model.

    Returns:
        Dict[int, int]: A dictionary mapping GPU indices to the number of layers assigned to each GPU.

    >>> from unittest import mock
    >>> import torch
    >>> mock_get_device_properties = mock.Mock()
    >>> fake_device_properties = lambda gpu: type('', (), {'total_memory': (gpu + 1) * 1024})()
    >>> mock_get_device_properties.side_effect = fake_device_properties
    >>> torch.cuda.get_device_properties = mock_get_device_properties
    >>> calculate_per_gpu_layers([0, 1, 2], 30)
    {0: 5, 1: 10, 2: 15}
    """

    # 根据每个GPU的显存大小，计算每个GPU应分配的层数
    # 获取每个gpu的显存大小
    gpu_memory_map = {
        gpu: torch.cuda.get_device_properties(gpu).total_memory
        for gpu in gpu_list
    }

    # 计算总显存大小
    total_memory = sum(gpu_memory_map.values())

    # 计算每个GPU的显存比例
    gpu_memory_ratios = {
        gpu: memory / total_memory
        for gpu, memory in gpu_memory_map.items()
    }

    # 计算每个 GPU 应分配的层数
    per_gpu_layers = {
        gpu: int(round(total_layers * ratio))
        for gpu, ratio in gpu_memory_ratios.items()
    }

    # 修正分配误差，确保总层数为total_layers
    while True:
        diff = total_layers - sum(per_gpu_layers.values())
        if diff > 0:
            gpu_with_max_memory = max(gpu_memory_ratios, key=gpu_memory_ratios.get)
            per_gpu_layers[gpu_with_max_memory] += diff
        elif diff < 0:
            gpu_with_min_memory = min(gpu_memory_ratios, key=gpu_memory_ratios.get)
            per_gpu_layers[gpu_with_min_memory] -= -diff
        else:
            break

    return per_gpu_layers


def auto_configure_device_map(num_gpus: int = 2, gpu_list: Optional[List[int]] = None) -> Dict[str, int]:
    """
    Automatically configure the device map for model parallelism based on the number of GPUs and their memory ratios.

    Args:
        num_gpus (int): The number of GPUs to be used.
        gpu_list (Optional[List[int]]): An optional list of GPU indices. Defaults to None.

    Returns:
        Dict[str, int]: A dictionary representing the device map for model parallelism.

    >>> from unittest import mock
    >>> import torch
    >>> # mock torch.cuda.get_device_properties
    >>> mock_get_device_properties = mock.Mock()
    >>> fake_device_properties = lambda gpu: type('', (), {'total_memory': (gpu + 1) * 1024})()
    >>> mock_get_device_properties.side_effect = fake_device_properties
    >>> torch.cuda.get_device_properties = mock_get_device_properties
    >>> # mock torch.cuda.device_count
    >>> mock_device_count = mock.Mock()
    >>> mock_device_count.return_value = 3
    >>> torch.cuda.device_count = mock_device_count
    >>> for k, v in auto_configure_device_map(3).items():
    ...     print(f"{k}: {v}")
    transformer.word_embeddings: 0
    transformer.final_layernorm: 0
    lm_head: 0
    transformer.layers.0: 0
    transformer.layers.1: 0
    transformer.layers.2: 0
    transformer.layers.3: 1
    transformer.layers.4: 1
    transformer.layers.5: 1
    transformer.layers.6: 1
    transformer.layers.7: 1
    transformer.layers.8: 1
    transformer.layers.9: 1
    transformer.layers.10: 1
    transformer.layers.11: 1
    transformer.layers.12: 1
    transformer.layers.13: 2
    transformer.layers.14: 2
    transformer.layers.15: 2
    transformer.layers.16: 2
    transformer.layers.17: 2
    transformer.layers.18: 2
    transformer.layers.19: 2
    transformer.layers.20: 2
    transformer.layers.21: 2
    transformer.layers.22: 2
    transformer.layers.23: 2
    transformer.layers.24: 2
    transformer.layers.25: 2
    transformer.layers.26: 2
    transformer.layers.27: 2
    """

    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28

    if gpu_list is None:
        gpu_list = list(range(num_gpus))
    assert len(gpu_list) <= torch.cuda.device_count(), "分配的GPU数量超过了实际可用的GPU数量"

    # 获取每个gpu的承载的层数
    per_gpu_layer_dict = calculate_per_gpu_layers(gpu_list, total_layers=num_trans_layers + 2)

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    current_gpu_index = 0
    current_gpu = gpu_list[current_gpu_index]
    device_map = {
        'transformer.word_embeddings': current_gpu,
        'transformer.final_layernorm': current_gpu,
        'lm_head': current_gpu
    }

    used = 2

    # 分配剩余的层数
    for i in range(num_trans_layers):
        if used < per_gpu_layer_dict[current_gpu]:
            used += 1
        else:
            # 当前 GPU 的层数已分配完，切换到下一个 GPU
            current_gpu_index += 1
            current_gpu = gpu_list[current_gpu_index]
            used = 1

        device_map[f"transformer.layers.{i}"] = current_gpu
    return device_map


def load_model_on_gpus(checkpoint_path: Union[str, os.PathLike], num_gpus: int = 2,
                       gpu_list: Optional[List[int]] = None,
                       multi_gpu_model_cache_dir: Union[str, os.PathLike] = "./temp_model_dir",
                       device_map: Optional[Dict[str, int]] = None,
                       tokenizer: Optional[PreTrainedTokenizer] = None, **kwargs) -> Module:
    """
    Load a pretrained model on multiple GPUs.

    Args:
        checkpoint_path (Union[str, os.PathLike]): The path to the checkpoint or model directory.
        num_gpus (int, optional): The number of GPUs to use. Defaults to 2.
        gpu_list (Optional[List[int]], optional): A list of GPU indices. Defaults to None.
        multi_gpu_model_cache_dir (Union[str, os.PathLike], optional): A directory to cache the multi-GPU model.
        device_map (Optional[Dict[str, int]], optional): A dictionary representing the device map for model parallelism.
        tokenizer (Optional[PreTrainedTokenizer], optional): The tokenizer to be used with the model. Defaults to None.
        **kwargs: Additional keyword arguments for loading the model.

    Returns:
        Module: The pretrained model on multiple GPUs.
    """

    from accelerate import load_checkpoint_and_dispatch

    model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs)
    model = model.eval()

    if device_map is None:
        device_map = auto_configure_device_map(num_gpus, gpu_list)
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

        if tokenizer is not None:
            tokenizer.save_pretrained(multi_gpu_model_cache_dir)
        print(f"loading model successfully, you should use checkpoint_path={multi_gpu_model_cache_dir} next time")

    return model


def load_model_and_tokenizer(checkpoint_path: Union[str, os.PathLike], num_gpus: int = 1,
                             multi_gpu_model_cache_dir: Union[str, os.PathLike] = "./temp_model_dir",
                             gpu_list: Optional[List[int]] = None,
                             **kwargs) -> Tuple[Module, PreTrainedTokenizer]:
    """
    Load a pretrained model and its tokenizer.

    Args:
        checkpoint_path (Union[str, os.PathLike]): The path to the checkpoint or model directory.
        num_gpus (int, optional): The number of GPUs to use. Defaults to 1.
        multi_gpu_model_cache_dir (Union[str, os.PathLike], optional): A directory to cache the multi-GPU model.
        gpu_list (Optional[List[int]], optional): A list of GPU indices. Defaults to None.
        **kwargs: Additional keyword arguments for loading the model and tokenizer.

    Returns:
        Tuple[Module, PreTrainedTokenizer]: A tuple containing the loaded model and tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs)
    if num_gpus < 2:
        model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half().cuda()
        model = model.eval()
    else:
        model = load_model_on_gpus(checkpoint_path, num_gpus=num_gpus, gpu_list=gpu_list,
                                   multi_gpu_model_cache_dir=multi_gpu_model_cache_dir,
                                   tokenizer=tokenizer, **kwargs)
    return model, tokenizer
