#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import platform
import requests
import json
import time
import signal

import readline
from rich import print, get_console

console = get_console()
print = console.print
input = console.input


os_name = platform.system()
clear_command = "cls" if os_name == "Windows" else "clear"
stream = True
api_url = (
    os.environ.get("ChatGLM_API", "http://127.0.0.1:8889")
    + "/v1/chat/completions"
)
max_history_len = 6
max_role_history_len = 6
welcome_text = "💕 欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，/role 设定角色，/clear 清空对话历史，/stop 终止程序, Ctrl+C 停止生成"
bot_name = "🤖"
user_name = "👤"
failed_resp_test = "AI 我呀，有点想不通了..."
secret = os.environ.get("ChatGLM_SECRET", "76c99b384a67d8ec6ff28d400bf91d0b")
type_wait_time = 0.05
temperature = 1
top_p = 0.7
max_tokens = 2048
stop_print = False

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {secret}",
}


def request_chat(data):
    with console.status(f"{bot_name} 思考中......\n") as status:
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
            if lines and lines.startswith("data: "):
                data_raw = lines[len("data: ") :]
                try:
                    data = json.loads(data_raw)
                    if data.get("choices")[0].get("finish_reason") == "stop":
                        break
                    yield data
                except Exception as exp:
                    print(f"\n\nSystem Error: {exp}", end="\n\n")

                if data_raw == "[DONE]":
                    break
        else:
            print(f"\n\nSystem Error: Status {response.status_code}")


def wrap_req_data(chatlist):
    return {
        "stream": stream,
        "model": "gpt-3.5-turbo",
        "messages": chatlist,
    }


def _tget(response):
    if stream:
        return response.get("choices")[0].get("delta").get("content", "")
    else:
        return response.get("choices")[0].get("message").get("content", "")


def _utext(content):
    return {"role": "user", "content": content}


def _ctext(content):
    return {"role": "assitant", "content": content}


def stop_print_signal_handler(signal, frame):
    global stop_print
    stop_print = True


def main():
    global stop_print, stream, temperature, top_p, max_tokens
    role_history = []
    history = []
    print(welcome_text, end="\n\n")
    print(
        f"▶️  参数 temperature: {temperature}, top_p: {top_p}, max_tokens: {max_tokens}, /set 设置",
        end="\n\n",
    )
    while True:
        query = input(f"\n{user_name}：")
        print("", end="\n\n")

        if query.strip() == "/set":
            _t = input("\ntemperature：")
            _tp = input("\ntop_p：")
            _mt = input("\nmax_tokens：")
            try:
                _t, _tp, _mt = float(_t), float(_tp), int(_mt)
                if _t < 0 or _t > 1:
                    raise Exception
                if _tp < 0 or _tp > 1:
                    raise Exception
                if _mt < 0 or _mt > 4096:
                    raise Exception
                temperature, top_p, max_tokens = _t, _tp, _mt
            except:
                print("输入有误", end="\n\n")
            print(f"temperature 使用值：{temperature}", end="\n")
            print(f"top_p 使用值：{top_p}", end="\n")
            print(f"max_tokens 使用值：{max_tokens}", end="\n\n")
        if query.strip() == "/stop":
            break
        if query.strip() == "/clear":
            history, role_history = [], []
            os.system(clear_command)
            print(welcome_text, end="\n\n")
            continue
        if query.strip() == "/role":
            print(
                f"请输入角色设定（注意提问方式），/exit 取消设定, /ok 完成设定，不超过{max_role_history_len}",
                end="\n\n",
            )
            cancled = False
            new_role = []
            for i in range(1, max_role_history_len + 1):
                query = input("\n设定 " + str(i) + "：")
                if query.strip() == "/exit":
                    cancled = True
                    break
                if query.strip() == "/ok":
                    break

                req_data = wrap_req_data(history + [_utext(query)])

                print(f"\n\n{bot_name} 记录中......", end="\n\n")

                response = None
                if stream:
                    completed = ""
                    for res_data in request_stream_chat(req_data):
                        if res_data:
                            completed += _tget(res_data)
                    response = completed
                else:
                    response = _tget(request_chat(req_data))
                if response:
                    new_role.append(_utext(query))
                    new_role.append(_ctext(response))
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

        # stream chat
        req_data = wrap_req_data(history + [_utext(query)])
        if stream:
            print(f"{bot_name}：", end="")
            completed = ""
            for res_data in request_stream_chat(req_data):
                if stop_print:
                    stop_print = False
                    break
                if res_data:
                    text = _tget(res_data)
                    completed += text
                    print(text, end="")
                    signal.signal(signal.SIGINT, stop_print_signal_handler)
            print("", end="\n\n")

            if completed:
                history.append(_utext(query))
                history.append(_ctext(completed))

        else:
            res_data = request_chat(req_data)
            if res_data:
                print(f"{bot_name}：", end="")
                text = _tget(res_data)
                history.append(_utext(query))
                history.append(_ctext(text))
                for i in range(0, len(text), 8):
                    print(text[i : i + 8], end="")
                    time.sleep(type_wait_time)
                print("", end="\n\n")
            else:
                print(failed_resp_test, end="\n\n")


if __name__ == "__main__":
    main()
