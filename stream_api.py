from transformers import AutoTokenizer, AutoModel
from threading import Thread
import time
import sched
from flask import Flask, request, jsonify
from multiprocessing.pool import ThreadPool

tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()

app = Flask(__name__)

handling_list_max_length = 10  # 最大处理数
waiting_list_max_length = 20  # 最大等待队列数

current_handling = []
handling_history = {}
pool = ThreadPool(handling_list_max_length)


def chat(request_id, query, history):
    global current_handling
    handling_history[request_id]["is_handled"] = True
    for response, history in model.stream_chat(tokenizer, query, history=history):
        handling_history[request_id]["response"] = response

    handling_history[request_id]["is_finished"] = True
    current_handling.remove(request_id)


@app.route('/chat', methods=['POST'])
def query():
    global current_handling
    # 获取 POST 请求中的参数
    data = request.get_json()
    request_id = data.get('request_id')
    history = data.get('history', [])
    query = data.get('query')

    # 当正在处理的请求数大于最大处理数时，返回繁忙
    if len(current_handling) > (handling_list_max_length + waiting_list_max_length):
        return jsonify({'code': 100, 'msg': 'busy now'})

    if request_id in handling_history:
        return jsonify({'code': 101, 'msg': 'request_id already exists'})

    current_handling.append(request_id)
    handling_history[request_id] = {
        "timestamp": time.time(),
        "response": "",
        "is_finished": False,
        "is_handled": False
    }

    history_data = []
    for each in history:
        history_data.append((each[0], each[1]))

    # 开启线程池进行推理
    pool.apply_async(chat, args=(request_id, query, history_data))

    # 没有匹配项返回空
    return jsonify({'code': 0, 'msg': 'start process'})


@app.route('/get_response', methods=['POST'])
def getResponse():
    data = request.get_json()
    request_id = data.get('request_id')

    if not request_id in handling_history:
        return jsonify({'code': 102, 'msg': 'request_id not exists'})

    return jsonify({'code': 0, 'msg': 'success', 'response': handling_history[request_id]})


def clearHistory():
    # 定时清楚处理history，以防堆叠
    global handling_history
    now = time.time()
    need_delete = []
    for request_id in handling_history:
        if now - handling_history[request_id]["timestamp"] > 60*60*1000 and handling_history[request_id]["is_finished"]:
            need_delete.append(request_id)
    for request_id in need_delete:
        del handling_history[request_id]

    startClean()


def startClean():
    s = sched.scheduler(time.time, time.sleep)
    s.enter(60, 1, clearHistory, ())
    s.run()


if __name__ == '__main__':
    cleanT = Thread(target=startClean)
    cleanT.start()
    app.run(debug=False, port=8000, host='0.0.0.0')
