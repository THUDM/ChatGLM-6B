#!/usr/bin/python3
import os
import platform
import requests
import json
import time

os_name = platform.system()
clear_command = "cls" if os_name == "Windows" else "clear"
stream = True
api_url = os.environ.get("ChatGLM_API", "http://127.0.0.1:8888") + "/chat"
max_history_len = 6
max_role_history_len = 6
welcome_text = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，role 设定角色，clear 清空对话历史，stop 终止程序"
failed_resp_test = "AI 我呀，有点想不通了..."
secret = os.environ.get("ChatGLM_SECRET", "721d95ac31da59fa022ec8c12f72f597")
type_wait_time = 0.05

headers = {
    "Content-Type": "application/json",
    "Authorization": secret,
}


def request_chat(data):
    response = requests.post(api_url, headers=headers, json=data)
    res = None
    if response.status_code == 200:
        res = response.json()
    else:
        print(
            "Request failed with code {}.".format(response.status_code),
            end="\n\n",
        )
    return res


def request_stream_chat(data):
    response = requests.post(api_url, headers=headers, json=data, stream=True)
    if response.status_code == 200:
        for lines in response.iter_lines(decode_unicode=True):
            if lines:
                data = json.loads(lines)
                if data.get("status") == 200:
                    if data.get("stop", True):
                        break
                    yield data
                else:
                    print(
                        f"\n\nSystem Error: Status {data.get('status')}",
                        end="\n\n",
                    )
                    break
    else:
        print(f"\n\nSystem Error: Status {response.status_code}")


def main():
    global stream
    role_history = []
    history = []
    print(welcome_text, end="\n\n")
    while True:
        query = input("\n用户：")
        print("", end="\n\n")

        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            role_history = []
            os.system(clear_command)
            print(welcome_text, end="\n\n")
            continue
        if query.strip() == "role":
            print(
                f"请输入角色设定（注意提问方式），exit 取消设定, ok 完成设定，不超过{max_role_history_len}",
                end="\n\n",
            )
            cancled = False
            new_role = []
            for i in range(1, max_role_history_len + 1):
                query = input("\n设定 " + str(i) + "：")
                if query.strip() == "exit":
                    cancled = True
                    break
                if query.strip() == "ok":
                    break
                req_data = {
                    "prompt": query,
                    "history": new_role,
                    "stream": False,
                }

                print("\n\nChatGLM-6B 记录中......", end="\n\n")

                if stream:
                    req_data["stream"] = True
                    response = None
                    for res_data in request_stream_chat(req_data):
                        if res_data:
                            response = res_data
                else:
                    response = request_chat(req_data)
                if response:
                    new_role = response.get("history", history)
                else:
                    print("该设定失败！", end="\n\n")
            if not cancled:
                role_history = new_role
                history = role_history
                print("设定角色成功！", end="\n\n")
                print(history, end="\n\n")
            continue

        if len(role_history) > 0:
            if len(history) > max_history_len + len(role_history):
                history = role_history + history[-max_history_len:]
        else:
            history = history[-max_history_len:]

        print("ChatGLM-6B：", end="")

        # stream chat
        if stream:
            completed = None
            last_stop = 0
            for res_data in request_stream_chat(
                {"prompt": query, "history": history, "stream": True}
            ):
                if res_data:
                    text = res_data.get("response", failed_resp_test)
                    print(text[last_stop:], end="", flush=True)
                    last_stop = len(text)
                    completed = res_data
            print("", end="\n\n", flush=True)

            if completed:
                history = completed.get("history", history)

        else:
            res_data = request_chat(
                {"prompt": query, "history": history, "stream": False}
            )
            if res_data:
                history = res_data.get("history", history)
                text = res_data.get("response", failed_resp_test)
                for i in range(0, len(text), 8):
                    print(text[i : i + 8], end="", flush=True)
                    time.sleep(type_wait_time)
                print("", end="\n\n", flush=True)
            else:
                print(failed_resp_test, end="\n\n")


if __name__ == "__main__":
    main()
