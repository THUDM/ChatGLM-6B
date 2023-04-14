# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModel
import json
from flask import Flask, request, make_response
from gevent.pywsgi import WSGIServer
from apscheduler.schedulers.blocking import BlockingScheduler
import threading

def job():
    global user_dialogue_record
    user_dialogue_record = {}

def timer():
    scheduler = BlockingScheduler()
    scheduler.add_job(job, 'interval', minutes=30)
    # scheduler.add_job(job, 'interval', seconds=30)
    scheduler.start()


t_update = threading.Thread(target=timer)
t_update.start()

user_dialogue_record = {}

app = Flask(__name__)
@app.route("/chat", methods=['POST'])
def dialogue():
    try:
        args = request.json
        if 'user_id' not in args or 'query' not in args:
            raise Exception(400)

        user_id = str(args.get('user_id', ''))
        query = str(args.get('query',''))
        clear = str(args.get('clear', False))
        if query == "":
            response_t = "你没有输入对话内容，请输入..."
            result = {"message": response_t}
            response = make_response(json.dumps(result) + '\r\n')
            response.mimetype = 'application/json'
            return response, 200

        if user_id in user_dialogue_record:
            history = user_dialogue_record[user_id]
        else:
            history = []
            user_dialogue_record[user_id] = history

        if clear:
            history = []
            user_dialogue_record[user_id] = history

        response_t, history = model.chat(tokenizer, query, history=history)

        user_dialogue_record[user_id] = history

        result = {"message":response_t}
        response = make_response(json.dumps(result) + '\r\n')
        response.mimetype = 'application/json'
        print(user_dialogue_record)
        return response, 200
    except Exception as e:
        if str(e.__str__()).startswith('4'):
            response = make_response(json.dumps({'status': 'error'}) + '\r\n')
            response.mimetype = 'application/json'
            return response, 400
        else:
            response = make_response(json.dumps({'status': 'error'}) + '\r\n')
            response.mimetype = 'application/json'
            return response, 500



if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("./llm_6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("./llm_6b", trust_remote_code=True).half().cuda()
    model = model.eval()
    http_server = WSGIServer(('0.0.0.0', 8018), app)
    http_server.serve_forever()

