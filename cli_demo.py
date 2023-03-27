import os
import platform
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'


def build_prompt(history, prev_resp, count):
    cur_resp = history[count][1][len(prev_resp[0]):]
    d = cur_resp.encode('unicode_escape')
    if b'\\ufff' in d:
        return
    print(cur_resp, end='', flush=True)
    prev_resp[0] += cur_resp

def main():
    history = []
    os.system(clear_command)
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    count = 0
    while True:
        query = input("\n\n用户：")
        if query == "stop":
            break
        if query == "clear":
            history = []
            count = 0
            os.system(clear_command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        print('\nChat：', end='')
        prev_resp = [""]
        for response, history in model.stream_chat(tokenizer, query, history=history):
            build_prompt(history, prev_resp, count)
        count += 1


if __name__ == "__main__":
    main()
