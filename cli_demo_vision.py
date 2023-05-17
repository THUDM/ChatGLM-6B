import os
import platform
import signal
import sys

from transformers import AutoTokenizer, AutoModel
import readline

tokenizer = AutoTokenizer.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history, prefix):
    prompt = prefix
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    global stop_stream
    while True:
        history = []
        prefix = "欢迎使用 VisualGLM-6B 模型，输入图片路径和内容即可进行对话，clear 清空对话历史，stop 终止程序"
        print(prefix)
        image_path = input("\n请输入图片路径：")
        if image_path == "stop":
            break
        prefix = prefix + "\n" + image_path
        query = "描述这张图片。"
        while True:
            count = 0
            for response, history in model.stream_chat(tokenizer, image_path, query, history=history):
                if stop_stream:
                    stop_stream = False
                    break
                else:
                    count += 1
                    if count % 8 == 0:
                        os.system(clear_command)
                        print(build_prompt(history, prefix), flush=True)
                        signal.signal(signal.SIGINT, signal_handler)
            os.system(clear_command)
            print(build_prompt(history, prefix), flush=True)
            query = input("\n用户：")
            if query.strip() == "clear":
                break
            if query.strip() == "stop":
                sys.exit(0)


if __name__ == "__main__":
    main()
