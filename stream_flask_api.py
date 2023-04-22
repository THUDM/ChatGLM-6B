import os

from flask import Flask, Response, request, redirect
from transformers import AutoTokenizer, AutoModel
import json, time, datetime, hashlib, redis
import torch

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
GLM_SECRET_KEY = os.environ.get(
    "GLM_SECRET_KEY", "721d95ac31da59fa022ec8c12f72f597"
)

tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/chatglm-6b", trust_remote_code=True
)
model = (
    AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    .half()
    .cuda()
)
model.eval()

app = Flask(__name__)
mq = redis.StrictRedis()


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def wrap_answer(prompt, response, history, stop=True):
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "stop": stop,
        "status": 200,
        "time": time,
    }
    log = (
        "["
        + time
        + "] "
        + '", prompt:"'
        + prompt
        + '", response:"'
        + repr(response)
        + '"'
    )
    print(log)
    return answer


def chat(prompt, history=[], max_length=None, top_p=None, temperature=None):
    response, history = model.chat(
        tokenizer,
        prompt,
        history=history,
        max_length=max_length if max_length else 2048,
        top_p=top_p if top_p else 0.7,
        temperature=temperature if temperature else 0.95,
    )

    return wrap_answer(prompt, response, history)


def stream_chat(
    prompt, history=[], max_length=None, top_p=None, temperature=None
):
    global model, tokenizer
    for response, history in model.stream_chat(
        tokenizer,
        prompt,
        history,
        max_length=max_length if max_length else 2048,
        top_p=top_p if top_p else 0.7,
        temperature=temperature if temperature else 0.95,
    ):
        answer = wrap_answer(prompt, response, history, stop=False)
        yield answer

    yield {"stop": True, "status": 200}


@app.post("/chat")
async def chat_api():
    secret = request.headers.get("Authorization")
    if secret != GLM_SECRET_KEY:
        return {"status": 403}

    global model, tokenizer
    json_post_raw = request.json
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    stream = json_post_list.get("stream")

    if stream:
        key = hashlib.md5(
            (f"{secret}-{int(time.time() * 1000)}").encode()
        ).hexdigest()
        mq.set(f"prompt-{key}", json_post)

        return redirect(f"/stream/{key}", code=307)
    else:
        prompt = json_post_list.get("prompt")
        history = json_post_list.get("history")
        max_length = json_post_list.get("max_length")
        top_p = json_post_list.get("top_p")
        temperature = json_post_list.get("temperature")

        answer = chat(prompt, history, max_length, top_p, temperature)
    torch_gc()

    return answer


@app.route("/stream/<key>", methods=["GET", "POST"])
def sse(key):
    secret = request.headers.get("Authorization")
    if secret != GLM_SECRET_KEY:
        return {"status": 403}

    if key:
        post_data = mq.get(f"prompt-{key}")
        if post_data:
            json_post_list = json.loads(post_data)
            prompt = json_post_list.get("prompt")
            history = json_post_list.get("history")
            max_length = json_post_list.get("max_length")
            top_p = json_post_list.get("top_p")
            temperature = json_post_list.get("temperature")

            def generate():
                for answer in stream_chat(
                    prompt, history, max_length, top_p, temperature
                ):
                    yield "\n" + json.dumps(answer) + "\n"

            return Response(generate(), mimetype="text/event-stream")

    return {"status": 200}


if __name__ == "__main__":
    # pass
    app.run(threaded=True, host="0.0.0.0", port=8888)
