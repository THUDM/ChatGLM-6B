import os
from transformers import AutoTokenizer, AutoModel
import signal
import platform
from stream_utils import ChatGLMStreamDecoder


tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/chatglm-6b", trust_remote_code=True)
stream_decoder = ChatGLMStreamDecoder(tokenizer.sp_tokenizer.text_tokenizer.sp)
model = AutoModel.from_pretrained(
    "THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    history = []
    global stop_stream
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            stream_decoder.end()
            stream_decoder.get()
            os.system(clear_command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        gen_kwargs = {"max_length": 2048, "do_sample": True, "top_p": 0.7,
                      "temperature": 0.95, "logits_processor": None}
        if not history:
            prompt = query
        else:
            prompt = "".join([f"[Round {i}]\n问：{q}\n答：{r}\n" for i, (q, r) in enumerate(
                history)] + [f"[Round {len(history)}]\n问：{query}\n答："])
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        print("\nChatGLM-6B：", end="")
        response = []
        for outputs in model.stream_generate(**inputs, **gen_kwargs):
            stream_decoder.put([int(outputs[0][-1])])
            new_resp = stream_decoder.get().replace("<n>", "\n")
            response.append(new_resp)
            print(new_resp, end="")
        # end of line
        stream_decoder.end()
        new_resp = stream_decoder.get().replace("<n>", "\n")
        response.append(new_resp)
        print(new_resp)
        response = "".join(response)
        history.append((query, response))


if __name__ == "__main__":
    main()
