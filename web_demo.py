from transformers import AutoModel, AutoTokenizer
import gradio as gr

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2


def btn_is_clickable(txt):
    if txt is not None and txt.strip() != '':
        return gr.update(interactive=True)
    else:
        return gr.update(interactive=False)


def user_message(user_message, history):
    return "", history + [[user_message, None]]


def predict(input, max_length, top_p, temperature, history):
    if len(history) > MAX_TURNS:
        history = history[-20:]
    for response, const_history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        history[-1][1] = response
        yield history


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("https://github.com/THUDM/ChatGLM-6B")
            max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
        with gr.Column(scale=4):
            chatbot = gr.Chatbot([], elem_id="chatbot").style(height=750)
            with gr.Row():
                with gr.Column(scale=4):
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="有问题就会有答案"
                    ).style(container=False)
                with gr.Column(scale=1, min_width=0):
                    btn = gr.Button("发送", interactive=False)
        # 控制按钮是否可以点击
        txt.change(btn_is_clickable, txt, btn)
        # 发送消息
        btn.click(user_message, [txt, chatbot], [txt, chatbot], queue=False).then(
            predict, [txt, max_length, top_p, temperature, chatbot], chatbot
        )

demo.queue().launch(share=False, inbrowser=True)
