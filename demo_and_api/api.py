from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModel
from pydantic import BaseModel
import uvicorn, json, datetime
import torch

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

app = FastAPI()

class Params(BaseModel):
    prompt: str = 'hello'
    history: list[list[str]] = []
    max_length: int = 2048
    top_p: float = 0.7
    temperature: float = 0.95

class Answer(BaseModel):
    status: int = 200
    time: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    response: str
    history: list[list[str]] = []

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

async def create_chat(params: Params):
    global model, tokenizer
    response, history = model.chat(tokenizer,
                                   params.prompt,
                                   history=params.history,
                                   max_length=params.max_length,
                                   top_p=params.top_p,
                                   temperature=params.temperature)
    answer_ok = Answer(response=response, history=history)
    print(answer_ok.json())
    torch_gc()
    return answer_ok

async def create_stream_chat(params: Params):
    global model, tokenizer
    for response, history in model.stream_chat(tokenizer,
                                   params.prompt,
                                   history=params.history,
                                   max_length=params.max_length,
                                   top_p=params.top_p,
                                   temperature=params.temperature):  
        answer_ok = Answer(response=response, history=history)
        # print(answer_ok.json())
        yield "\ndata: " + json.dumps(answer_ok.json())
    
    torch_gc()

@app.post("/chat")
async def post_chat(params: Params):
    answer = await create_chat(params)
    return answer

@app.post("/stream_chat")
async def post_stream_chat(params: Params):
    return StreamingResponse(create_stream_chat(params))

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
