import os
from typing import Dict, Tuple, Union, Optional

from torch.nn import Module
from transformers import AutoModel


def load_model_on_gpus(checkpoint_path: Union[str, os.PathLike], num_gpus: int = 2,
                       device_map: Optional[Dict[str, int]] = None, **kwargs) -> Module:
    if num_gpus < 2 and device_map is None:
        model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half().cuda()
    else:
        from accelerate import dispatch_model

        model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half()

        if device_map is None:
            from accelerate import infer_auto_device_map

            device_map = infer_auto_device_map(model, no_split_module_classes=["GLMBlock"])
            # e.g. Use max_memory to set the upper limit memory size of each device.
            # Huggingface suggest to save some memory of gpu0 for some reasons.
            #device_map = infer_auto_device_map(model, max_memory={0: "4GiB", 1: "10GiB", "cpu": "30GiB"}, no_split_module_classes=["GLMBlock"])
            #print(device_map)

        model = dispatch_model(model, device_map=device_map)

    return model


