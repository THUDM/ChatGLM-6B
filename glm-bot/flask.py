import os
import platform
from transformers import AutoTokenizer, AutoModel

from flask import Flask, request

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
def prepare_model():
    global model
    model = model.eval()

prepare_model()
model = model.eval()
preset = []
port = 7860
os_name = platform.system()
app = Flask(__name__)
history = {}
@app.route('/chatglm', methods=["GET"])
def delete_msg():
    global history
    query = request.args.get('msg')
    usrid = request.args.get('usrid')
    source = request.args.get('source')
    if query == None:
        return '请提供内容'
    if query == 'ping':
        return 'pong!服务端运行正常!'    
    if source == None:   
        return '无来源的请求，请更新插件'
    if usrid == None:
        return '请提供用户id'
    if not usrid in history:
        history[usrid] = preset
    print(f"usrid：{usrid},content：{query}")
    if query == "clear":
        history[usrid] = preset

        print(f"usrid：{usrid},清空历史")
        return '已重置当前对话'
    response, history[usrid] = model.chat(tokenizer, query, history=history[usrid])
    print(f"ChatGLM-6B：{response}")
    return response
    
if __name__ == '__main__':
    print(f"欢迎使用 ChatGLM-6B API，可通过发送GET请求到http://127.0.0.1:{port}/chatglm来调用。")
    app.run(host='0.0.0.0', port=port)  
