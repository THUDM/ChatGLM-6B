from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json
from transformers import AutoModel, AutoTokenizer
from typing import List,Tuple


max_length = 4096
# 根据id获取上下文信息
def get_history(id: str) -> List[Tuple[str,str]] or None:
    if id in sessions.keys():
        length = len(json.dumps(sessions[id],indent=2))
        if length>max_length:
            sessions[id] = []
            return None
        if sessions[id] == []:
            return None
        return sessions[id]
    else:
        sessions[id] = []
        return None

# 根据id清空上下文


def clear(id: str) -> str:
    sessions[id] = []
    return '已重置'




tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2

sessions = {}


def predict(prompt: str, uid: str, max_length: int = 2048, top_p: float = 0.7, temperature: float = 0.95) -> str:
    history = get_history(uid)
    print(history)
    response, history = model.chat(tokenizer, prompt, history=history, max_length=max_length, top_p=top_p,
                                   temperature=temperature)
    sessions[uid].append((prompt, response))
    print(get_history(uid))
    return response

# while 1:
#     uid = input("uid:")
#     prompt = input('msg:')
#     msg = predict(prompt=prompt,uid = uid)
#     print(msg)


app = FastAPI()


class Item_chat(BaseModel):
    msg: str
    uid: str
@app.post("/chat")
def chat(item:Item_chat):
    msg = predict(prompt=item.msg, uid=item.uid)
    print(msg)
    return msg


class Item_claer(BaseModel):
    uid: str
@app.post("/clear")
def clear_session(item:Item_claer):
    return clear(item.uid)


uvicorn.run(app, host="0.0.0.0", port=10269)
