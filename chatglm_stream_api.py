#! /usr/bin/python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import ServerSentEvent, EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
from transformers import AutoTokenizer, AutoModel
import argparse
import logging
import os
import json
import sys

def getLogger(name, file_name, use_formatter=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s    %(message)s')
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    if file_name:
        handler = logging.FileHandler(file_name, encoding='utf8')
        handler.setLevel(logging.INFO)
        if use_formatter:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
            handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = getLogger('ChatGLM', 'chatlog.log')

MAX_HISTORY = 5

class ChatGLM():
    def __init__(self, model_name_or_path, quantize_level, gpu_id) -> None:
        logger.info("Start initialize model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True)
        self.model = self._model(model_name_or_path, quantize_level, gpu_id)
        self.model.eval()
        _, _ = self.model.chat(self.tokenizer, "你好", history=[])
        logger.info("Model initialization finished.")
    
    def _model(self, model_name_or_path, quantize_level, gpu_id):
        model_name = model_name_or_path
        quantize = int(quantize_level)
        model = None
        if gpu_id == '-1':
            if quantize == 8:
                print('CPU模式下量化等级只能是16或4，使用4')
                model_name = "THUDM/chatglm-6b-int4"
            elif quantize == 4:
                model_name = "THUDM/chatglm-6b-int4"
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True).float()
        else:
            gpu_ids = gpu_id.split(",")
            self.devices = ["cuda:{}".format(id) for id in gpu_ids]
            if quantize == 16:
                model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
            else:
                model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().quantize(quantize).cuda()
        return model
    
    def clear(self) -> None:
        if torch.cuda.is_available():
            for device in self.devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
    
    def answer(self, query: str, history):
        response, history = self.model.chat(self.tokenizer, query, history=history)
        history = [list(h) for h in history]
        return response, history

    def stream(self, query, history, max_length, top_p, temperature):
        if query is None or history is None:
            yield {"query": "", "response": "", "history": [], "finished": True}
        size = 0
        response = ""
        for response, history in self.model.stream_chat(self.tokenizer, query = query, history = history, max_length = max_length, top_p = top_p, temperature = temperature):
            this_response = response[size:]
            history = [list(h) for h in history]
            size = len(response)
            yield {"delta": this_response, "response": response, "finished": False}
        logger.info("Answer - {}".format(response))
        yield {"query": query, "delta": "[EOS]", "response": response, "history": history, "finished": True}


def start_server(model_name_or_path, quantize_level, http_address: str, port: int, gpu_id: str):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    bot = ChatGLM(model_name_or_path, quantize_level, gpu_id)
    
    app = FastAPI()
    app.add_middleware( CORSMiddleware,
        allow_origins = ["*"],
        allow_credentials = True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    @app.post("/chat")
    async def answer_question(arg_dict: dict):
        result = {"query": "", "response": "", "success": False}
        try:
            text = arg_dict["query"]
            ori_history = arg_dict["history"]
            logger.info("Query - {}".format(text))
            if len(ori_history) > 0:
                logger.info("History - {}".format(ori_history))
            history = ori_history[-MAX_HISTORY:]
            history = [tuple(h) for h in history] 
            response, history = bot.answer(text, history)
            logger.info("Answer - {}".format(response))
            ori_history.append((text, response))
            result = {"query": text, "response": response,
                      "history": ori_history, "success": True}
        except Exception as e:
            logger.error(f"error: {e}")
        return result

    @app.post("/stream")
    async def answer_question_stream(arg_dict: dict):
        def decorate(generator):
            for item in generator:
                yield ServerSentEvent(json.dumps(item, ensure_ascii=False), event='delta')
        result = {"query": "", "response": "", "success": False}
        try:
            query = arg_dict.get("query",None)
            ori_history = arg_dict.get("history",[])
            max_length = arg_dict.get("max_length",2048)
            top_p = arg_dict.get("top_p",0.95)
            temperature = arg_dict.get("temperature",0.01)
            logger.info("Query - {}".format(query))
            if len(ori_history) > 0:
                logger.info("History - {}".format(ori_history))
            history = ori_history[-MAX_HISTORY:]
            history = [tuple(h) for h in history]
            return EventSourceResponse(decorate(bot.stream(query, history, max_length, top_p, temperature)))
        except Exception as e:
            logger.error(f"error: {e}")
            return EventSourceResponse(decorate(bot.stream(None, None)))

    logger.info("starting server...")
    uvicorn.run(app=app, host=http_address, port=port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream API Service for ChatGLM-6B')
    parser.add_argument('--model_name_or_path', '-m', help='model name', default='THUDM/chatglm-6b')
    parser.add_argument('--device', '-d', help='device，-1 means cpu, other means gpu ids', default='0')
    parser.add_argument('--quantize', '-q', help='level of quantize, option：16 or 4', default=16)
    parser.add_argument('--host', '-H', help='host to listen', default='0.0.0.0')
    parser.add_argument('--port', '-P', help='port of this service', default=8888)
    args = parser.parse_args()
    start_server(args.model_name_or_path, args.quantize, args.host, int(args.port), args.device)
