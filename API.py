from typing import Optional
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, time, datetime, os, platform

app = FastAPI()
@app.post("/")
async def create_item(request: Request):
    global history, model, tokenizer
    jsonPostRaw = await request.json()
    jsonPost = json.dumps(jsonPostRaw)
    jsonPostList = json.loads(jsonPost)
    prompt = jsonPostList.get('prompt')
    response, history = model.chat(tokenizer, prompt, history=history)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response":response,
        "status":200,
        "time":time
    }
    log = "["+time+"] "+'device:"'+jsonPostList.get('device')+'", prompt:"'+prompt+'", response:"'+repr(response)+'"'
    print(log)
    return answer

if __name__ == '__main__':
    uvicorn.run('API:app',host='0.0.0.0',port=8000,workers=1)

history = []
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().quantize(4).cuda()
model = model.eval()
