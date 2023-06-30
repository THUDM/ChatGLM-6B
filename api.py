import json
import datetime
import torch
import uvicorn
from typing import List
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
from pydantic import BaseModel
from utils import load_model_on_gpus


devices_list = [
    'cuda:0',
    'cuda:1'
]


def _torch_gc():
    if torch.cuda.is_available():
        for item in devices_list:
            with torch.cuda.device(item):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


class Question(BaseModel):
    prompt: str
    history: List[str] = []
    max_length: int = 2048
    top_p: float = 0.7
    temperature: float = 0.95


app = FastAPI()


@app.post('/chat/')
async def chat(question: Question):
    response, history = model.chat(
        tokenizer,
        question.prompt,
        history=question.history,
        max_length=question.max_length,
        top_p=question.top_p,
        temperature=question.temperature
    )
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    _torch_gc()
    return answer


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "THUDM/chatglm-6b", trust_remote_code=True
    )
    model = load_model_on_gpus("THUDM/chatglm-6b", num_gpus=2)
    # model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    model.eval()
    uvicorn.run(app, host="127.0.0.1", port=11001, workers=1)
