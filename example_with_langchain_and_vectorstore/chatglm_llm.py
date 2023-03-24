from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoTokenizer, AutoModel

"""ChatGLM_G is a wrapper around the ChatGLM model to fit LangChain framework. May not be an optimal implementation"""

class ChatGLM_G(LLM):

    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()
    history = []

    @property
    def _llm_type(self) -> str:
        return "ChatGLM_G"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response, updated_history = self.model.chat(self.tokenizer, prompt, history=self.history)
        print("ChatGLM: prompt: ", prompt)
        print("ChatGLM: response: ", response)
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = updated_history
        return response
    
    def __call__(self, prompt: str,  stop: Optional[List[str]] = None) -> str:
        response, updated_history = self.model.chat(self.tokenizer, prompt, history=self.history)
        print("ChatGLM: prompt: ", prompt)
        print("ChatGLM: response: ", response)
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = updated_history
        
        return response