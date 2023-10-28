from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from sse_starlette.sse import EventSourceResponse
from transformers import AutoTokenizer, AutoModel
import uvicorn
import torch

'''
 此脚本实现模型响应结果的流式传输，让用户无需等待完整内容的响应。
 This script implements the streaming transmission of model response results, eliminating the need for users to wait for a complete response of the content.
 访问接口时它将返回event-stream流，你需要在客户端接收并处理它。
 When accessing the interface, it will return an 'event-stream' stream, which you need to receive and process on the client.

 POST http://127.0.0.1:8010
 { "input": "你好ChatGLM" }

 input: 输入内容
 max_length: 最大长度
 top_p: 采样阈值
 temperature: 抽样随机性
 history: 二维历史消息数组，eg: [["你好ChatGLM","你好，我是ChatGLM，一个基于语言模型的人工智能助手。很高兴见到你，欢迎问我任何问题。"]]
 html_entities: 开启HTML字符实体转换
'''

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()

def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

async def predict(input, max_length, top_p, temperature, history, html_entities):
    global model, tokenizer
    for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        yield parse_text(response) if html_entities else response
    torch_gc()

class ConversationsParams(BaseModel):
    input: str
    max_length: Optional[int] = 2048
    top_p: Optional[float] = 0.7
    temperature: Optional[float] = 0.95
    history: Optional[list] = []
    html_entities: Optional[bool] = True

@app.post('/')
async def conversations(params: ConversationsParams):
    history = list(map(tuple, params.history))
    predictGenerator = predict(params.input, params.max_length, params.top_p, params.temperature, history, params.html_entities)
    return EventSourceResponse(predictGenerator)

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8010, workers=1)
