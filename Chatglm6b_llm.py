import json
from langchain.llms.base import LLM
from typing import Optional, List
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import Dict, Tuple, Union, Optional


Chatglm_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE_ID = "0" if torch.cuda.is_available() else None
DEVICE = f"{Chatglm_DEVICE}:{DEVICE_ID}" if DEVICE_ID else Chatglm_DEVICE

# streaming reponse
STREAMING = True

# model name
Chatglm_MODEL = "chatglm-6b-local"

# supported LLM models
llm_model_dict = {
    "chatglm-6b-int4-qe": "THUDM/chatglm-6b-int4-qe",
    "chatglm-6b-int4": "THUDM/chatglm-6b-int4",
    "chatglm-6b-int8": "THUDM/chatglm-6b-int8",
    "chatglm-6b": "THUDM/chatglm-6b",
    "chatglm-6b-local": r"", #your local model path
}

def torch_gc(DEVICE):
    if torch.cuda.is_available():
        with torch.cuda.device(DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception as e:
            print(e)

def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus
    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    device_map = {'transformer.word_embeddings': 0, 'transformer.final_layernorm': 0, 'lm_head': 0}
    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.layers.{i}'] = gpu_target
        used += 1

    return device_map


class ChatGLM(LLM):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    # history = []
    tokenizer: object = None
    model: object = None
    history_len: int = 10
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    def __init__(self):
        super().__init__()
        
    @property
    def _llm_type(self) -> str:
        return "ChatGLM-6b"

    def _call(self, prompt: str, history: List[List[str]] = [], streaming: bool = STREAMING):  #out Tuple[str, List[List[str]]]:
        if streaming:
            for inum, (stream_resp, _) in enumerate(
                    self.model.stream_chat(
                        self.tokenizer,
                        prompt,
                        history=history[-self.history_len:-1] if self.history_len > 0 else [],
                        max_length=self.max_token,
                        temperature=self.temperature,
                    )):
                torch_gc(DEVICE)
                if inum == 0:
                    history += [[prompt, stream_resp]]
                else:
                    history[-1] = [prompt, stream_resp]
                yield stream_resp, history
        else:
            response, _ = self.model.chat(
                self.tokenizer,
                prompt,
                history=history[-self.history_len:] if self.history_len > 0 else [],
                max_length=self.max_token,
                temperature=self.temperature,
            )
            torch_gc(DEVICE)
            history += [[prompt, response]]
            yield response, history

    def load_model(self,
                   model_name_or_path: str = "THUDM/chatglm-6b",
                   llm_device=Chatglm_DEVICE,
                #    use_ptuning_v2=False,
                   device_map: Optional[Dict[str, int]] = None,
                   **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

        if torch.cuda.is_available() and llm_device.lower().startswith("cuda"):
            # 根据当前设备GPU数量决定是否进行多卡部署
            num_gpus = torch.cuda.device_count()
            if num_gpus < 2 and device_map is None:
                self.model = (AutoModel.from_pretrained(model_name_or_path,
                                                        config=model_config,
                                                        trust_remote_code=True,
                                                        **kwargs).half().cuda())
            else:
                from accelerate import dispatch_model
                model = (AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, config=model_config,
                                                   **kwargs).half())
                # 可传入device_map自定义每张卡的部署情况
                if device_map is None:
                    device_map = auto_configure_device_map(num_gpus)
                self.model = dispatch_model(model, device_map=device_map)
        else:
            self.model = (AutoModel.from_pretrained(model_name_or_path, config=model_config, trust_remote_code=True,
                                                    **kwargs).float().to(llm_device))

        self.model = self.model.eval()


if __name__ == "__main__":
    llm = ChatGLM()
    llm.load_model(
        model_name_or_path=llm_model_dict[Chatglm_MODEL],
        llm_device=Chatglm_DEVICE,
    )
    last_print_len = 0
    for resp, history in llm._call("你好", streaming=True):
        print(resp[last_print_len:], end="", flush=True)
        last_print_len = len(resp)
    for resp, history in llm._call("你好", streaming=False):
        print(resp)
