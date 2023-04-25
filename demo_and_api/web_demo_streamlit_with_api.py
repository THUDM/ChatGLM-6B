import streamlit as st
from streamlit_chat import message
import requests
import json

st.set_page_config(
    page_title="ChatGLM-6b 演示",
    page_icon=":robot:"
)

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2
url = "http://localhost:8000/stream_chat"


def predict(input, max_length, top_p, temperature, history=None):
    if history is None:
        history = []

    with container:
        if len(history) > 0:
            for i, (query, response) in enumerate(history):
                message(query, avatar_style="big-smile", key=str(i) + "_user")
                message(response, avatar_style="bottts", key=str(i))

        message(input, avatar_style="big-smile", key=str(len(history)) + "_user")
        st.write("AI正在回复:")
        with st.empty():
            req = {
                "prompt": input,
                "history": history,
                "max_length": max_length,
                "top_p": top_p,
                "temperature": temperature
            }
            res = requests.post(url=url,json=req,stream=True)
            for line in res.iter_lines(delimiter=b'\ndata: '):
                line = line.decode(encoding='utf-8')  
                if line.strip() == '':
                    continue;
                response_json = json.loads(json.loads(line))
                response = response_json['response']
                history = response_json['history']
                st.write(response)

    return history


container = st.container()

# create a prompt text for the text generation
prompt_text = st.text_area(label="用户命令输入",
            height = 100,
            placeholder="请在这儿输入您的命令")

max_length = st.sidebar.slider(
    'max_length', 0, 4096, 2048, step=1
)
top_p = st.sidebar.slider(
    'top_p', 0.0, 1.0, 0.6, step=0.01
)
temperature = st.sidebar.slider(
    'temperature', 0.0, 1.0, 0.95, step=0.01
)

if 'state' not in st.session_state:
    st.session_state['state'] = []

if st.button("发送", key="predict"):
    with st.spinner("AI正在思考，请稍等........"):
        # text generation
        st.session_state["state"] = predict(prompt_text, max_length, top_p, temperature, st.session_state["state"])