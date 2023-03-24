from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime, os

api = FastAPI()

@api.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    response, history = model.chat(tokenizer, prompt, history=history)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    return answer

@api.post("/clear")
async def clear_history():
    global history
    history = []
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": "history cleared",
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + 'History Cleared'
    print(log)
    return answer

@api.post("/history")
async def get_history():
    global history
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + 'Get History'
    print(log)
    return answer

if __name__ == '__main__':
    try:
        uvicorn.run('api:api', host='0.0.0.0', port=8000, workers=1)
    except KeyboardInterrupt:
        os._exit(0)

history = []
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model.eval()