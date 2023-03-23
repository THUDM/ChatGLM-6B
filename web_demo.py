import os
from transformers import AutoModel, AutoTokenizer
import gradio as gr

MODEL_ID = "./model" if os.path.exists('./model') else "THUDM/chatglm-6b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).half().cuda()
model = model.eval()

MAX_TURNS = 20

WELCOME_PROMPT = [[None, "[ChatGLM-6B]：Welcome, please input text and press enter"]]


def predict(input, max_length, top_p, temperature, history):
    for _, history in model.stream_chat(
            tokenizer, input, history,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
    ):
        chatbot = []

        for query, response in history:
            chatbot.append([
                "[用户]：" + query,
                "[ChatGLM-6B]：" + response
            ])

        if len(chatbot) > MAX_TURNS:
            chatbot = chatbot[- MAX_TURNS:]

        yield history, WELCOME_PROMPT + chatbot


with gr.Blocks(title="ChatGLM-6B", css='#main-chatbot { height: 480px; }') as demo:
    input_cache = gr.State()
    history = gr.State([])

    with gr.Row():
        with gr.Column():
            pass
        with gr.Column():
            chatbot = gr.Chatbot(
                show_label=False,
                elem_id="main-chatbot"
            )
            input = gr.Textbox(
                show_label=False,
                placeholder="Input text and press enter",
                interactive=True,
            )
            with gr.Box():
                max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
        with gr.Column():
            pass

    input.submit(
        lambda x: ("", x),
        [input],
        [input, input_cache]
    ).then(
        predict,
        [input_cache, max_length, top_p, temperature, history],
        [history, chatbot],
    )

    demo.load(
        lambda: WELCOME_PROMPT,
        None,
        [chatbot]
    )

demo.queue().launch(share=False, inbrowser=True)
