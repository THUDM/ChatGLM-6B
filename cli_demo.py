import os
import platform
import argparse
import time
from transformers import AutoTokenizer, AutoModel



parser = argparse.ArgumentParser(description='cli demo')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
parser.add_argument('--showTime', action='store_true', help='show time consuming')
parser.add_argument('--local', action='store_true',help='using local models,default path:/models/chatglm-6b')

args = parser.parse_args()

os_name = platform.system()

# mac: force use cpu
if os_name == 'Darwin':
    args.cpu = True


model_name = "THUDM/chatglm-6b"
if args.local:
    model_name = "./models/chatglm-6b"


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
if(args.cpu):
    model = model.float()
else:
    model =  model.half().cuda()
model = model.eval()



history = []
print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
while True:
    query = input("\n用户：")
    if query == "stop":
        break
    if query == "clear":
        history = []
        command = 'cls' if os_name == 'Windows' else 'clear'
        os.system(command)
        print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
        continue
    timeStart = time.perf_counter()
    response, history = model.chat(tokenizer, query, history=history)
    timeEnd = time.perf_counter()
    showTime="({timeEnd - timeStart:0.4f}s)" if args.showTime else ""

    print(f"ChatGLM-6B {showTime}：{response}")
